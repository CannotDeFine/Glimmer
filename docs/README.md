# Documentation

[Project home](../README.md) / Documentation

## Using Glimmer

- [Project overview](../README.md) — Capabilities and a quick start with your own application.
- [Building](BUILDING.md) — Dependencies, presets and optional CUDA components.
- [Usage](USAGE.md) — Service startup, application configuration, priorities and shutdown.
- [CUDA compatibility](CUDA_API_COVERAGE.md) — Execution-path support and limitations.
- [Architecture](ARCHITECTURE.md) — How applications, scheduling and GPU execution fit together.

## Examples

- [Example overview](../examples/README.md) — Choose a workload and execution path.
- [Priority demo](../examples/priority_demo/README.md) — Two synthetic CUDA workloads with different priorities.
- [CUDA workloads](../examples/cuda_workloads/README.md) — Allocation, memory accounting and streams.
- [PyTorch workloads](../examples/framework_workloads/pytorch_smoke/README.md) — Run a framework workload alone or with a competing process.
- [Explicit task backend](../examples/cuda_task_backend/README.md) — Submit application-owned task descriptors.
- [GEMM comparison](../examples/gemm_comparison/README.md) — Compare isolated GEMM implementations under explicit correctness and coverage checks.

## Contributing

- [Development lifecycle](DEVELOPMENT_LIFECYCLE.md) — Change scope, verification and review gates.
- [Coding guidelines](CODING_GUIDELINES.md) — Source style, ownership and error handling.
- [Testing](TESTING.md) — Host, CUDA and measurement checks.
- [Dependency management](DEPENDENCY_MANAGEMENT.md) — Reproducible dependency and SDK integration.
- [Git guidelines](GIT_GUIDELINES.md) — Logical changes and commit conventions.
- [Storage layout](STORAGE_LAYOUT.md) — Build reuse and generated-artifact retention.
- [Scripts](../scripts/README.md) — Development tools and experiment collectors.
