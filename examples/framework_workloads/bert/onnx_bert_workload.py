#!/usr/bin/env python3
"""Run pinned BERT QA with the ordinary ONNX Runtime CUDA execution provider.

There are no scheduler APIs or custom operators in this application. Host input
and output transfers are included in each request, like the upstream benchmark.
"""

import argparse
import json
import pathlib
import time

from bert_contract import PROFILE, SEED, configure_cache, sha256, summarize, validate_assets, wait_for_start, write_json
from bert_workload import make_batch


def validate_export(directory, assets):
    manifest = json.loads((directory / "export.json").read_text())
    if (manifest.get("status") != "ok" or manifest.get("assets") != assets
            or manifest.get("seed") != SEED or manifest.get("dtype") != "float32"
            or manifest.get("opset") != 17
            or manifest.get("model_sha256") != sha256(directory / "bert.onnx")
            or manifest.get("input_names") != ["input_ids", "attention_mask", "token_type_ids"]
            or manifest.get("output_names") != ["start_logits", "end_logits"]):
        raise ValueError("missing or mismatched ONNX export provenance")
    for name in ("export_bert.py", "bert_workload.py", "bert_contract.py"):
        if manifest.get("source_sha256", {}).get(name) != sha256(pathlib.Path(__file__).with_name(name)):
            raise ValueError("ONNX export oracle source changed")
    return manifest


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "role": "inference", "engine": "onnxruntime-cuda",
              "dtype": "float32", "profile": PROFILE["inference"], "seed": SEED,
              "preemption_acceptance": False, "performance_acceptance": False,
              "iterations": args.iterations, "warmup": args.warmup,
              "instrumented": args.profile_cuda,
              "application_sha256": sha256(__file__)}
    try:
        assets = validate_assets(args.assets)
        exported = validate_export(args.export, assets)
        configure_cache()
        # Importing CUDA PyTorch also preloads its matching CUDA/cuDNN libraries
        # for ORT, per the upstream CUDA EP compatibility instructions.
        import torch
        import numpy as np
        import onnxruntime as ort
        from transformers import BertTokenizerFast

        if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA execution is required; no CPU-only fallback")
        tokenizer = BertTokenizerFast.from_pretrained(args.assets, local_files_only=True)
        inputs, features = make_batch(torch, tokenizer, args.assets, "inference")
        if features != exported["inputs"]:
            raise ValueError("ONNX and PyTorch reference inputs differ")
        inputs = {name: value.numpy() for name, value in inputs.items()}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(args.export / "bert.onnx"), sess_options=options,
                                       providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}),
                                                  "CPUExecutionProvider"])
        session.disable_fallback()
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime failed to initialize the CUDA provider")
        reference = [np.asarray(exported["reference"][name], dtype=np.float32) for name in exported["output_names"]]

        def check(values):
            if len(values) != 2:
                raise ValueError("ONNX output inventory differs")
            for actual, expected in zip(values, reference):
                if not np.isfinite(actual).all():
                    raise ValueError("nonfinite ONNX logits")
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)

        # Cross-engine numerical validation is outside the timed region.
        check(session.run(None, inputs))
        for _ in range(args.warmup):
            session.run(None, inputs)
        torch.cuda.synchronize()
        wait_for_start(args.ready_file, args.start_file, str(torch.cuda.get_device_properties(0).uuid))
        samples, outputs = [], []
        if args.profile_cuda:
            torch.cuda.profiler.start()
        begin = time.monotonic_ns()
        try:
            for _ in range(args.iterations):
                start = time.monotonic_ns()
                result = session.run(None, inputs)
                torch.cuda.synchronize()
                samples.append(time.monotonic_ns() - start)
                outputs.append(result)
        finally:
            end = time.monotonic_ns()
            if args.profile_cuda:
                torch.cuda.profiler.stop()
        for values in outputs:
            check(values)
        report.update(status="ok", assets=assets, inputs=features, export_sha256=sha256(args.export / "export.json"),
                      providers=session.get_providers(), provider_options=session.get_provider_options(),
                      environment={"onnxruntime": ort.__version__, "torch": torch.__version__,
                                   "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                                   "device_uuid": str(torch.cuda.get_device_properties(0).uuid)},
                      validation={"reference": "native PyTorch logits", "rtol": 2e-4, "atol": 2e-5,
                                  "checked_requests": len(outputs)}, samples_ns=samples,
                      timing=summarize(samples, end - begin, 1), measured_begin_ns=begin, measured_end_ns=end,
                      timing_scope="solo closed-loop request including H2D/D2H and completion; no request queue")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(args.output / "result.json", report)
    print(json.dumps({"status": report["status"], "timing": report.get("timing"), "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--export", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--ready-file", type=pathlib.Path)
    parser.add_argument("--start-file", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or bool(args.ready_file) != bool(args.start_file):
        parser.error("require nonnegative warmup, positive iterations and paired ready/start files")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
