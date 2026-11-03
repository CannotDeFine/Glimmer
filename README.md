# Glimmer

Glimmer coordinates CUDA applications that share a GPU. It combines
priority-aware scheduling and CUDA API interception to control when covered
application work runs and how supported GPU allocations are accounted for.

## Capabilities

- Schedule work from multiple applications with priorities, weighted sharing
  and configurable concurrency.
- Account for supported GPU allocations and apply software memory limits.

## Get started

The following launches your application with shared **launch admission**:
CUDA execution stays in the application, and the service orders queued launches.
For build requirements, see [Building Glimmer](docs/BUILDING.md).
Run these commands from the repository root in terminals without existing
Glimmer or preload settings.

```sh
git submodule update --init --recursive -- 3rdparty/spdlog
cmake --preset cuda-debug
cmake --build --preset cuda-debug -j2 --target \
  glimmer_control_service glimmer_control_client glimmer_cuda_interceptor
```

Start the service in one terminal. The command creates a private directory for
its socket and prints the socket path:

```sh
glimmer_run_dir="$(mktemp -d /tmp/glimmer.XXXXXXXX)" &&
printf 'Socket path: %s/control.sock\n' "$glimmer_run_dir" &&
env -u LD_PRELOAD -u LD_AUDIT ./build/cuda-debug/bin/glimmer_control_service \
  --socket "$glimmer_run_dir/control.sock" \
  --execution-mode remote --quota-bytes 1073741824 \
  --scheduler-policy priority --max-concurrent-tasks 2 \
  --lease-timeout-ms 5000 --bind-leases-to-process
```

Once the socket exists, copy its printed path into another terminal and run
your CUDA application:

```sh
glimmer_socket="/replace/with/the/printed/socket/path"
env -u LD_AUDIT \
  LD_PRELOAD="$PWD/build/cuda-debug/lib/libglimmer_cuda_interceptor.so" \
  GLIMMER_SCHEDULER_MODE=enforce \
  GLIMMER_SCHEDULER_CONTROL_SOCKET="$glimmer_socket" \
  GLIMMER_SCHEDULER_TENANT_ID=app-a \
  GLIMMER_SCHEDULER_PRIORITY=100 \
  GLIMMER_MAX_CONCURRENT_KERNELS=2 \
  ./your_cuda_application
```

Replace the last line with your executable and arguments, or `python3 your_app.py`
in its CUDA-enabled framework environment. Launch another application with a
different tenant name and a lower priority, such as `10`, to share the same
service. Higher priorities take precedence when work is queued; running kernels
are not interrupted. The service quota above accounts for task reservations;
it is not an application allocation limit.

Let applications finish before stopping the service with Ctrl+C. The empty
private directory can then be removed with `rmdir "$glimmer_run_dir"` in the
service terminal. The
[usage guide](docs/USAGE.md) covers configuration, memory limits and shutdown.
For a ready-made synthetic workload, try the
[priority demo](examples/priority_demo/README.md).

## Scope

Glimmer is under active development and targets Linux with NVIDIA GPUs.
Compatibility depends on the execution path, CUDA version and application APIs;
see [CUDA compatibility](docs/CUDA_API_COVERAGE.md).

The quick start schedules supported launches without interrupting running
kernels. Software quotas are not hardware isolation, and scheduling priorities
are not latency guarantees.

## License

Glimmer is licensed under the [Apache License 2.0](LICENSE). Third-party
dependencies are subject to their own licenses; see
[dependency management](docs/DEPENDENCY_MANAGEMENT.md).

## Docs

- [Documentation](docs/README.md) — Guide to the available documentation.
- [Building](docs/BUILDING.md) — Requirements and build configurations.
- [Usage](docs/USAGE.md) — Run applications, configure scheduling and manage services.
- [CUDA compatibility](docs/CUDA_API_COVERAGE.md) — Execution paths, reference environment and limitations.
- [Architecture](docs/ARCHITECTURE.md) — Runtime flow and component responsibilities.
- [Priority demo](examples/priority_demo/README.md) — Run competing CUDA workloads.
- [Contributing](docs/README.md#contributing) — Development, testing and coding conventions.
