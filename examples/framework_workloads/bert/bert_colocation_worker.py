#!/usr/bin/env python3
"""Scheduling-agnostic, open-loop measurement companion for the pinned BERT pair.

The original application, model and training step are not modified. This file
owns only benchmark setup, arrival timing and continuous offered training work.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import time

from bert_contract import (PROFILE, SEED, configure_cache, sha256, validate_assets,
                           wait_for_start, write_json)
from bert_workload import make_batch, step
from onnx_bert_workload import validate_export


def make_trace(count, period_ns, burst_size=1):
    if (any(type(value) is not int or value <= 0 for value in (count, period_ns, burst_size))
            or count > 1_000_000 or burst_size > count):
        raise ValueError("invalid bounded arrival trace parameters")
    return {"schema": "bert_arrivals_v1", "period_ns": period_ns, "burst_size": burst_size,
            "duration_ns": count * period_ns,
            "offsets_ns": [(index // burst_size) * burst_size * period_ns for index in range(count)]}


def trace_digest(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule_spec(schedule):
    begin = schedule["begin_ns"]
    if type(begin) is not int or begin <= 0 or type(schedule.get("preparation_required", False)) is not bool:
        raise ValueError("invalid arrival epoch")
    if "trace" not in schedule:
        trace = make_trace(schedule["requests"], schedule["period_ns"])
        return begin, trace, None
    trace = schedule["trace"]
    if (not isinstance(trace, dict) or not isinstance(trace.get("offsets_ns"), list)
            or any(type(value) is not int for value in trace["offsets_ns"])
            or type(trace.get("duration_ns")) is not int
            or trace != make_trace(len(trace["offsets_ns"]), trace["period_ns"], trace["burst_size"])
            or schedule.get("trace_sha256") != trace_digest(trace)):
        raise ValueError("arrival trace or fingerprint differs")
    drain = schedule["drain_ns"]
    if type(drain) is not int or drain < 0:
        raise ValueError("invalid drain allowance")
    return begin, trace, begin + trace["duration_ns"] + drain


def publish(path, value):
    temporary = path.with_suffix(".pending")
    write_json(temporary, value)
    temporary.replace(path)


def sleep_until(timestamp_ns):
    while (remaining := timestamp_ns - time.monotonic_ns()) > 0:
        time.sleep(remaining / 1e9)


def training_setup(args, torch, tokenizer):
    from transformers import BertForQuestionAnswering

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = BertForQuestionAnswering.from_pretrained(
        args.assets, local_files_only=True, use_safetensors=True,
        attn_implementation="eager").cuda()
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8,
                                 weight_decay=0.0, foreach=False)
    inputs, features = make_batch(torch, tokenizer, args.assets, "training")

    def execute():
        outputs = step(torch, model, inputs, optimizer)
        torch.cuda.synchronize()
        return (outputs.start_logits.detach(), outputs.end_logits.detach(), outputs.loss.detach())

    def check(results):
        for values in results:
            if any(not torch.isfinite(value).all().item() for value in values):
                raise ValueError("nonfinite training logits or loss")

    return execute, check, features, {"engine": "pytorch", "attention": "eager",
        "parameter_count": sum(value.numel() for value in model.parameters())}


def inference_setup(args, torch, tokenizer, assets):
    import numpy as np
    import onnxruntime as ort

    exported = validate_export(args.export, assets)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA EP is required")
    inputs, features = make_batch(torch, tokenizer, args.assets, "inference")
    if features != exported["inputs"]:
        raise ValueError("ONNX reference inputs differ")
    inputs = {name: value.numpy() for name, value in inputs.items()}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.export / "bert.onnx"), sess_options=options,
        providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"])
    session.disable_fallback()
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA EP initialization failed; no CPU-only fallback")
    reference = [np.asarray(exported["reference"][name], dtype=np.float32)
                 for name in exported["output_names"]]

    def execute():
        values = session.run(None, inputs)
        torch.cuda.synchronize()
        return values

    def check(results):
        for values in results:
            if len(values) != len(reference):
                raise ValueError("ONNX output inventory differs")
            for actual, expected in zip(values, reference):
                if not np.isfinite(actual).all():
                    raise ValueError("nonfinite ONNX logits")
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)

    check([execute()])
    return execute, check, features, {"engine": "onnxruntime-cuda", "onnxruntime": ort.__version__,
        "providers": session.get_providers(), "provider_options": session.get_provider_options(),
        "export_sha256": sha256(args.export / "export.json")}


def measure_training(execute, stop_file, active_file, timeout, emit=lambda event: None):
    records, results = [], []
    deadline = time.monotonic() + timeout
    begin = time.monotonic_ns()
    emit({"phase": "begin", "begin_ns": begin})
    # This is a host offered-work marker, not evidence of physical GPU execution.
    publish(active_file, {"begin_ns": begin})
    while not stop_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("training stop signal timed out")
        start = time.monotonic_ns()
        emit({"phase": "start", "index": len(records), "start_ns": start})
        results.append(execute())
        records.append({"start_ns": start, "end_ns": time.monotonic_ns()})
        emit({"phase": "complete", "index": len(records) - 1, **records[-1]})
    end = time.monotonic_ns()
    emit({"phase": "stop", "end_ns": end, "completed": len(records)})
    return records, results, begin, end


def measure_inference(execute, start_file, emit=lambda event: None):
    schedule = json.loads(start_file.read_text())
    begin, trace, deadline = schedule_spec(schedule)
    records, results = [], []
    preparation = {}
    if schedule.get("preparation_required") is True:
        prepared = time.monotonic_ns()
        if prepared >= begin:
            raise ValueError("arrival trace preparation missed the epoch")
        preparation["prepared_ns"] = prepared
    emit({"phase": "begin", "begin_ns": begin, **preparation})
    stopped_at = begin
    for index, offset in enumerate(trace["offsets_ns"]):
        arrival = begin + offset
        sleep_until(arrival)
        start = time.monotonic_ns()
        if deadline is not None and start >= deadline:
            stopped_at = start
            break
        emit({"phase": "start", "index": index, "arrival_ns": arrival, "start_ns": start})
        results.append(execute())
        records.append({"arrival_ns": arrival, "start_ns": start, "end_ns": time.monotonic_ns()})
        emit({"phase": "complete", "index": index, **records[-1]})
    # Keep the offered window identical even if the last request finishes early.
    end = max(begin + trace["duration_ns"], records[-1]["end_ns"] if records else begin, stopped_at)
    sleep_until(end)
    emit({"phase": "stop", "end_ns": end, "completed": len(records)})
    return records, results, begin, end


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "role": args.role, "profile": PROFILE[args.role],
              "seed": SEED, "dtype": "float32", "warmup": args.warmup,
              "schema": "bert_bounded_arrival_v3", "instrumented": False,
              "measurement_journal": "per-step host start/completion, no kernel trace",
              "worker_sha256": sha256(__file__)}
    try:
        assets = validate_assets(args.assets)
        configure_cache()
        import torch
        from transformers import BertTokenizerFast

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        tokenizer = BertTokenizerFast.from_pretrained(args.assets, local_files_only=True)
        setup = (training_setup(args, torch, tokenizer) if args.role == "training"
                 else inference_setup(args, torch, tokenizer, assets))
        execute, check, features, engine = setup
        report.update(assets=assets, inputs=features, **engine,
                      environment={"torch": torch.__version__, "cuda": torch.version.cuda,
                          "device": torch.cuda.get_device_name(),
                          "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
                          "packages": {item.metadata["Name"]: item.version
                                       for item in importlib.metadata.distributions()}})
        for _ in range(args.warmup):
            execute()
        torch.cuda.synchronize()
        write_json(args.output / "setup.json", {**report, "status": "prepared", "pid": os.getpid()})
        wait_for_start(args.ready_file, args.start_file, report["environment"]["device_uuid"], timeout=args.timeout)
        binding = {"schema": "bert_progress_v1", "role": args.role, "pid": os.getpid(),
                   "schedule_sha256": sha256(args.start_file), "setup_sha256": sha256(args.output / "setup.json")}
        cpu_begin = {"observed_begin_ns": time.monotonic_ns(), "process_begin_ns": time.process_time_ns()}
        with (args.output / "progress.jsonl").open("x", encoding="utf-8", buffering=1) as journal:
            def emit(event):
                if event["phase"] == "begin":
                    event = {**event, **binding}
                journal.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")

            if args.role == "training":
                schedule = json.loads(args.start_file.read_text())
                if schedule:
                    begin, end = schedule["begin_ns"], schedule["end_ns"]
                    if type(begin) is not int or type(end) is not int or not 0 < begin < end:
                        raise ValueError("invalid training-only schedule")
                    sleep_until(begin)
                measured = measure_training(execute, args.stop_file, args.active_file, args.timeout, emit)
            else:
                measured = measure_inference(execute, args.start_file, emit)
        records, results, begin, end = measured
        cpu = {**cpu_begin, "process_end_ns": time.process_time_ns(), "observed_end_ns": time.monotonic_ns(),
               "scope": "measurement routine including lead-in and drain; setup, warmup and validation excluded"}
        # All numerical checks are outside the common timing window.
        check(results)
        report.update(status="ok", records=records, measured_begin_ns=begin, measured_end_ns=end,
                      cpu=cpu,
                      journal_sha256=sha256(args.output / "progress.jsonl"),
                      validation={"checked_steps": len(results), "kind": "finite logits/loss" if args.role == "training"
                                  else "native PyTorch oracle, rtol=2e-4, atol=2e-5"})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(args.output / "result.json", report)
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "export", "output", "ready-file", "start-file", "stop-file", "active-file"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=PROFILE, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.warmup < 1 or args.timeout <= 0:
        parser.error("positive warmup and timeout required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
