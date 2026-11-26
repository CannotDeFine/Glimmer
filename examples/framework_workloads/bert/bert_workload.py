#!/usr/bin/env python3
"""Run full upstream BERT QA on real SQuAD inputs without scheduling APIs.

This application is identical in native and intercepted runs. Offline assets
are required. Solo closed-loop latency is not an open-loop service SLA.
"""

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import time

from bert_contract import (PROFILE, SEED, answer_tokens, configure_cache, sha256, squad_examples,
                           summarize, validate_assets, wait_for_start, write_json)


def make_batch(torch, tokenizer, assets, role):
    profile = PROFILE[role]
    dataset = assets / ("train-v1.1.json" if role == "training" else "dev-v1.1.json")
    selected = squad_examples(json.loads(dataset.read_text(encoding="utf-8")), profile["batch_size"])
    rows = []
    for example in selected:
        encoded = tokenizer(example["question"], example["context"], truncation="only_second",
                            max_length=profile["sequence_length"], stride=32,
                            padding="max_length", return_overflowing_tokens=True,
                            return_offsets_mapping=True)
        for index, offsets in enumerate(encoded["offset_mapping"]):
            span = answer_tokens(offsets, encoded.sequence_ids(index), example["answer_start"],
                                 example["answer_start"] + len(example["answer"]))
            if span is not None:
                row = {key: encoded[key][index] for key in
                       ("input_ids", "attention_mask", "token_type_ids")}
                row.update(start_positions=span[0], end_positions=span[1])
                rows.append(row)
                break
        else:
            raise ValueError(f"no complete tokenized answer for {example['id']}")
    encoded_rows = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    tensors = {name: torch.tensor([row[name] for row in rows], dtype=torch.long) for name in rows[0]}
    if role == "inference":
        tensors = {name: tensor for name, tensor in tensors.items() if not name.endswith("positions")}
    return tensors, {"example_ids": [value["id"] for value in selected],
                     "features_sha256": hashlib.sha256(encoded_rows).hexdigest()}


def step(torch, model, inputs, optimizer):
    device_inputs = {name: value.cuda() for name, value in inputs.items()}
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**device_inputs)
        outputs.loss.backward()
        optimizer.step()
        return outputs
    with torch.inference_mode():
        return model(**device_inputs)


def snapshot(torch, model, outputs, training):
    result = {"start_logits": outputs.start_logits.detach().cpu(),
              "end_logits": outputs.end_logits.detach().cpu()}
    if training:
        result["loss"] = outputs.loss.detach().cpu()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise ValueError(f"missing parameter gradient: {name}")
            result[f"gradient/{name}"] = parameter.grad.detach().cpu()
            result[f"parameter/{name}"] = parameter.detach().cpu()
    for name, value in result.items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"nonfinite numerical result: {name}")
    return result


def run(arguments):
    arguments.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "role": arguments.role, "profile": PROFILE[arguments.role],
              "seed": SEED, "dtype": "float32", "engine": "pytorch", "attention": "eager",
              "warmup": arguments.warmup, "iterations": arguments.iterations,
              "instrumented": arguments.profile_cuda or arguments.snapshot or bool(arguments.reference),
              "preemption_acceptance": False, "performance_acceptance": False,
              "application_sha256": sha256(__file__),
              "contract_sha256": sha256(pathlib.Path(__file__).with_name("bert_contract.py"))}
    try:
        manifest = validate_assets(arguments.assets)
        configure_cache()
        import torch
        from transformers import BertForQuestionAnswering, BertTokenizerFast

        if not torch.cuda.is_available():
            raise RuntimeError("a working CUDA GPU is required; no CPU fallback")
        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tokenizer = BertTokenizerFast.from_pretrained(arguments.assets, local_files_only=True)
        model = BertForQuestionAnswering.from_pretrained(
            arguments.assets, local_files_only=True, use_safetensors=True,
            attn_implementation="eager").cuda()
        training = arguments.role == "training"
        model.train(training)
        optimizer = (torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8,
                                     weight_decay=0.0, foreach=False) if training else None)
        inputs, features = make_batch(torch, tokenizer, arguments.assets, arguments.role)
        report.update(assets=manifest, inputs=features,
                      parameter_count=sum(value.numel() for value in model.parameters()),
                      environment={"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "device": torch.cuda.get_device_name(),
                                   "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
                                   "packages": {item.metadata["Name"]: item.version
                                                for item in importlib.metadata.distributions()}})
        for _ in range(arguments.warmup):
            step(torch, model, inputs, optimizer)
        torch.cuda.synchronize()
        wait_for_start(arguments.ready_file, arguments.start_file, report["environment"]["device_uuid"])
        if arguments.profile_cuda:
            torch.cuda.profiler.start()
        samples, results = [], []
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic_ns()
        try:
            for _ in range(arguments.iterations):
                start = time.monotonic_ns()
                outputs = step(torch, model, inputs, optimizer)
                torch.cuda.synchronize()
                samples.append(time.monotonic_ns() - start)
                results.append((outputs.start_logits.detach(), outputs.end_logits.detach(),
                                outputs.loss.detach() if training else None))
        finally:
            end = time.monotonic_ns()
            if arguments.profile_cuda:
                torch.cuda.profiler.stop()
        for values in results:
            if any(not torch.isfinite(value).all().item() for value in values if value is not None):
                raise ValueError("nonfinite BERT output or training loss")
        numerical = None
        if arguments.snapshot or arguments.reference:
            numerical = snapshot(torch, model, outputs, training)
        if arguments.reference:
            reference_report = json.loads((arguments.reference / "result.json").read_text())
            identity = ("role", "profile", "seed", "dtype", "engine", "attention", "warmup",
                        "iterations", "application_sha256", "contract_sha256", "inputs", "assets", "environment")
            if (reference_report.get("status") != "ok"
                    or any(reference_report.get(key) != report.get(key) for key in identity)
                    or reference_report.get("snapshot_sha256") != sha256(arguments.reference / "snapshot.pt")):
                raise ValueError("numerical reference identity mismatch")
            reference = torch.load(arguments.reference / "snapshot.pt", map_location="cpu", weights_only=True)
            if reference.keys() != numerical.keys():
                raise ValueError("numerical reference tensor inventory mismatch")
            for name, value in numerical.items():
                torch.testing.assert_close(value, reference[name], rtol=2e-4, atol=2e-5, msg=name)
            report["numerical_comparison"] = {"checks": list(numerical), "rtol": 2e-4, "atol": 2e-5}
        if arguments.snapshot:
            torch.save(numerical, arguments.output / "snapshot.pt")
            report["snapshot_sha256"] = sha256(arguments.output / "snapshot.pt")
        report.update(status="ok", validation="finite outputs" if numerical is None else "finite tensor inventory",
                      samples_ns=samples, timing=summarize(samples, end - begin, PROFILE[arguments.role]["batch_size"]),
                      timing_scope="solo closed-loop step including H2D, framework work and GPU completion; no request queue",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      measured_begin_ns=begin, measured_end_ns=end)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(arguments.output / "result.json", report)
    print(json.dumps({"status": report["status"], "role": arguments.role,
                      "timing": report.get("timing"), "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=PROFILE, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--ready-file", type=pathlib.Path)
    parser.add_argument("--start-file", type=pathlib.Path)
    arguments = parser.parse_args()
    if (arguments.warmup < 0 or arguments.iterations <= 0
            or bool(arguments.ready_file) != bool(arguments.start_file)):
        parser.error("require nonnegative warmup, positive iterations and paired ready/start files")
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
