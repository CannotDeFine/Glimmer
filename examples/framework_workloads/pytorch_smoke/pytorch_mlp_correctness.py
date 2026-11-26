#!/usr/bin/env python3
"""Compare the unchanged smoke workload's MLP steps with CPU and native GPU results.

This companion fixture imports the application's model builder and training step;
it does not modify that application or call any Glimmer API. It is a numerical
check; the optional file barrier allows the external handoff runner to establish
priority insertion independently. It is not a performance measurement.
"""

import argparse
import copy
import hashlib
import json
import os
import pathlib
import time

from pytorch_smoke import build_model, run_step, wait_for_start, write_metadata


def save_snapshot(torch, destination, snapshots):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(snapshots, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--ready-file", type=pathlib.Path)
    parser.add_argument("--start-file", type=pathlib.Path)
    arguments = parser.parse_args()
    if bool(arguments.ready_file) != bool(arguments.start_file):
        parser.error("ready/start files must be supplied together")
    for destination in (arguments.output, arguments.snapshot):
        if destination.exists() or destination.is_symlink():
            parser.error(f"output already exists: {destination}")
    if arguments.output.resolve() == arguments.snapshot.resolve():
        parser.error("output and snapshot must be different paths")
    import torch

    torch.manual_seed(817)
    torch.backends.cuda.matmul.allow_tf32 = False
    cpu = build_model(torch, 1024, 4096, "cpu", torch.float32)
    gpu = copy.deepcopy(cpu).cuda()
    inputs, targets = torch.randn(32, 1024), torch.randn(32, 1024)
    gpu_inputs, gpu_targets = inputs.cuda(), targets.cuda()
    optimizers = [torch.optim.SGD(model.parameters(), lr=1.0e-3) for model in (cpu, gpu)]
    outputs = [[], []]
    handles = [model.register_forward_hook(
        lambda _module, _inputs, result, index=index: outputs[index].append(result.detach().clone()))
        for index, model in enumerate((cpu, gpu))]
    snapshots, checks = {}, []

    if arguments.ready_file:
        torch.cuda.synchronize()
        write_metadata(arguments.ready_file, {
            "pid": os.getpid(), "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "ready_ns": time.monotonic_ns(), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        })
        wait_for_start(arguments.start_file)

    def compare(name, actual, expected):
        actual = actual.detach().cpu()
        torch.testing.assert_close(actual, expected.detach().cpu(), rtol=2e-4, atol=2e-5)
        snapshots[name] = actual.clone()
        checks.append(name)

    for step in range(2):
        losses = [run_step(torch, model, x, y, optimizer, "training", 2)
                  for model, x, y, optimizer in
                  zip((cpu, gpu), (inputs, gpu_inputs), (targets, gpu_targets), optimizers)]
        torch.cuda.synchronize()
        compare(f"loss_{step}", losses[1], losses[0])
        for index, (expected, actual) in enumerate(zip(outputs[0], outputs[1])):
            compare(f"forward_{step}_{index}", actual, expected)
        outputs[0].clear()
        outputs[1].clear()
        for (name, expected), (_, actual) in zip(cpu.named_parameters(), gpu.named_parameters()):
            compare(f"gradient_{step}_{name}", actual.grad, expected.grad)
            compare(f"update_{step}_{name}", actual, expected)
    for handle in handles:
        handle.remove()
    if arguments.reference:
        native = torch.load(arguments.reference, map_location="cpu", weights_only=True)
        if native.keys() != snapshots.keys():
            raise RuntimeError("native/preload snapshot inventories differ")
        for name, value in snapshots.items():
            torch.testing.assert_close(value, native[name], rtol=2e-4, atol=2e-5)
    save_snapshot(torch, arguments.snapshot, snapshots)
    torch.cuda.synchronize()
    application = pathlib.Path(__file__).with_name("pytorch_smoke.py")
    result = {"status": "ok", "checks": checks, "native_compared": bool(arguments.reference),
              "application_sha256": hashlib.sha256(application.read_bytes()).hexdigest(),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "rtol": 2e-4, "atol": 2e-5}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"MLP numerical checks passed: {len(checks)}; native_compared={bool(arguments.reference)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
