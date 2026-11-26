#!/usr/bin/env python3
"""Validate public PyTorch CUDA results and ordering without Glimmer APIs.

This is a correctness fixture, not a performance benchmark or proof that any
framework kernel was transformed. Supply interception externally via preload.
"""

import argparse
import copy
import importlib.metadata
import json
import pathlib
import sys


def check(torch, name, actual, expected, checks):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
                               rtol=2e-4, atol=2e-5)
    checks.append(name)


def matrix_gradients(torch, checks):
    host = [torch.randn(32, 64), torch.randn(64, 16), torch.randn(16)]
    reference = [value.clone().requires_grad_() for value in host]
    device = [value.cuda().requires_grad_() for value in host]

    def evaluate(values):
        left, right, bias = values
        output = torch.nn.functional.gelu(left @ right + bias)
        output.square().mean().backward()
        return output

    expected = evaluate(reference)
    actual = evaluate(device)
    check(torch, "matrix_forward", actual, expected, checks)
    for index, (gpu, cpu) in enumerate(zip(device, reference)):
        check(torch, f"matrix_gradient_{index}", gpu.grad, cpu.grad, checks)


def convolution_gradients(torch, checks):
    if not torch.backends.cudnn.is_available() or not torch.backends.cudnn.enabled:
        raise RuntimeError("a cuDNN-enabled CUDA installation is required")
    reference = torch.nn.Conv2d(3, 16, 3, padding=1)
    device = copy.deepcopy(reference).cuda()
    inputs = torch.randn(4, 3, 32, 32)
    cpu_input = inputs.clone().requires_grad_()
    gpu_input = inputs.cuda().requires_grad_()
    expected = torch.nn.functional.gelu(reference(cpu_input))
    actual = torch.nn.functional.gelu(device(gpu_input))
    expected.square().mean().backward()
    actual.square().mean().backward()
    check(torch, "convolution_forward", actual, expected, checks)
    check(torch, "convolution_input_gradient", gpu_input.grad, cpu_input.grad, checks)
    for name in ("weight", "bias"):
        check(torch, f"convolution_{name}_gradient", getattr(device, name).grad,
              getattr(reference, name).grad, checks)


def stream_event_and_allocator_order(torch, checks):
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    source = torch.arange(4096, dtype=torch.float32).cuda()
    producer.wait_stream(torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    event, read_done, completed = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(producer):
        buffer = source * 3 + 5
        event.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(event)
        first = buffer.clone()
        read_done.record()
    with torch.cuda.stream(producer):
        producer.wait_event(read_done)
        buffer.add_(7)
        event.record()
    with torch.cuda.stream(consumer):
        consumer.wait_event(event)
        second = buffer.clone()
        buffer.record_stream(consumer)
        completed.record()
    # Exercise allocator lifetime after the application's last host reference,
    # while preserving the public record_stream contract for another stream.
    del buffer
    scratch = [torch.empty_like(source).fill_(index) for index in range(8)]
    completed.synchronize()
    expected = torch.arange(4096, dtype=torch.float32) * 3 + 5
    check(torch, "event_first_record", first, expected, checks)
    check(torch, "event_second_record", second, expected + 7, checks)
    del scratch


def graph_replay(torch, checks):
    inputs = torch.randn(32, 64).cuda()
    weight = torch.randn(64, 16).cuda()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            inputs.matmul(weight).relu()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = inputs.matmul(weight).relu()
    for iteration in range(3):
        replacement = torch.randn(32, 64)
        inputs.copy_(replacement)
        graph.replay()
        torch.cuda.synchronize()
        expected = replacement.matmul(weight.cpu()).relu()
        check(torch, f"graph_replay_{iteration}", output, expected, checks)


def cnn_workload_gradients(torch, checks):
    """Qualify the exact optional benchmark model, outside all timed windows."""
    from pytorch_smoke import build_model, configure_cnn_backend, run_step

    configure_cnn_backend(torch, "cuda")
    reference = build_model(torch, 1024, 4096, "cpu", torch.float32, "cnn")
    device = copy.deepcopy(reference).cuda()
    cpu_inputs = torch.randn(32, 3, 32, 32, requires_grad=True)
    gpu_inputs = cpu_inputs.detach().cuda().requires_grad_()
    cpu_targets = torch.randn(32, 10)
    gpu_targets = cpu_targets.cuda()
    check(torch, "cnn_workload_forward",
          run_step(torch, device, gpu_inputs, gpu_targets, None, "inference", 1),
          run_step(torch, reference, cpu_inputs, cpu_targets, None, "inference", 1), checks)
    cpu_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    gpu_optimizer = torch.optim.SGD(device.parameters(), lr=1e-3)
    for step in range(3):
        cpu_inputs.grad = None
        gpu_inputs.grad = None
        expected = run_step(torch, reference, cpu_inputs, cpu_targets, cpu_optimizer, "training", 2)
        actual = run_step(torch, device, gpu_inputs, gpu_targets, gpu_optimizer, "training", 2)
        check(torch, f"cnn_workload_loss_{step}", actual, expected, checks)
        check(torch, f"cnn_workload_input_gradient_{step}", gpu_inputs.grad, cpu_inputs.grad, checks)
        for (name, gpu), (cpu_name, cpu) in zip(device.named_parameters(), reference.named_parameters()):
            if name != cpu_name:
                raise RuntimeError("CNN reference parameter identity differs")
            check(torch, f"cnn_workload_gradient_{step}_{name}", gpu.grad, cpu.grad, checks)
            check(torch, f"cnn_workload_update_{step}_{name}", gpu, cpu, checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--cnn-workload", action="store_true",
                        help="also qualify the exact optional CNN benchmark against CPU")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")
    result = {"status": "failed", "checks": [], "measurement_kind": "correctness_only"}
    try:
        import torch
        result.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      python_version=sys.version, packages={
                          distribution.metadata["Name"]: distribution.version
                          for distribution in importlib.metadata.distributions()
                      })
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is not a passing result")
        torch.cuda.set_device(0)
        result.update(device_name=torch.cuda.get_device_name(),
                      device_capability=list(torch.cuda.get_device_capability()),
                      cudnn_version=torch.backends.cudnn.version(),
                      compiled_architectures=torch.cuda.get_arch_list())
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        matrix_gradients(torch, result["checks"])
        convolution_gradients(torch, result["checks"])
        stream_event_and_allocator_order(torch, result["checks"])
        graph_replay(torch, result["checks"])
        if args.cnn_workload:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocated_before = torch.cuda.memory_allocated()
            cnn_workload_gradients(torch, result["checks"])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            allocated_after = torch.cuda.memory_allocated()
            result["cnn_allocator_bytes"] = {"before": allocated_before, "after": allocated_after}
            if allocated_after != allocated_before:
                raise RuntimeError("CNN did not restore live tensor allocation bytes")
            result["checks"].append("cnn_workload_tensor_release")
        torch.cuda.synchronize()
        result["status"] = "ok"
    except Exception as error:  # Preserve a failed artifact; do not claim a pass.
        result["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"pytorch_correctness status={result['status']} checks={len(result['checks'])} "
          f"output={args.output}")
    if result["status"] != "ok":
        print(result["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
