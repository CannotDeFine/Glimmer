# Building Glimmer

[Project home](../README.md) / [Documentation](../docs/README.md) / Build guide

Run commands from the repository root. Builds are Linux-only and use C++20.
The examples below describe source configuration, not a prebuilt distribution.

Reuse the fixed `build/<preset>/` directories. For a temporary custom or overlay
configuration, use the single `build/candidate/` slot. Configure rejects other
repository-local build paths; do not create a new numbered directory per run.
Keep results in `artifacts/`, separate from incremental build products. See the
[storage rules](STORAGE_LAYOUT.md#fixed-build-slots) before repurposing a slot.

## Prerequisites

- CMake 3.20 or newer, a C++20-capable GCC or Clang, and Make for the supplied
  `Unix Makefiles` presets.
- Git with submodule support: run
  `git submodule update --init --recursive -- 3rdparty/spdlog` after cloning.
- Python 3.10 or newer for Python-based checks and workload tools.
- For CUDA targets: a CUDA Toolkit. Running GPU tests additionally needs a
  compatible NVIDIA driver and GPU. CUDA workload/provider builds also use
  `nvcc`.
- The CUDA backend requires the host C++ implementation to provide an
  always-lock-free `std::atomic_ref<std::uint32_t>`; the build checks this
  requirement for mapped execution-control words.
- For CUDA C++20 source targets with NVIDIA `nvcc`: CMake 3.25.2 or newer and
  `nvcc` 12.0 or newer. The core-only build retains the CMake 3.20 minimum.

Framework environments and optional cuDNN/NCCL SDKs are not installed by the
core build. Follow the selected workload's requirements rather than reusing an
arbitrary framework environment.

The CUTLASS submodule is needed only for the optional persistent GEMM provider
or GEMM comparison. Initialize it when using either target:

```sh
git submodule update --init -- 3rdparty/cutlass
```

## Core and control plane

On a fresh build directory, the `debug` preset leaves CUDA targets disabled:

```sh
cmake --preset debug
cmake --build --preset debug -j2
ctest --preset debug --output-on-failure --no-tests=error
```

`./scripts/build.sh` is a convenience wrapper for the configure and build
commands. It does not initialize submodules or run tests. It rejects symlinked
or frozen slots, symlinked caches, nonempty slots without a regular cache and
existing non-symlink compilation databases before configuring. It does not
validate the slot's cached options. It checks the 10 GB build budget before
and after its configure and build steps, but cannot predict their peak usage.

Core outputs include `build/debug/bin/glimmer_control_service` and
`build/debug/bin/glimmer_control_client`. Their `--help` output lists the current
control-plane commands; these binaries are not the centralized CUDA execution
service.

CMake caches retain explicit settings. A preset does not reset an existing
directory to default options. Reuse compatible builds, but do not overwrite a
frozen validation baseline or assume an existing `debug` tree is CPU-only.

In the root project, when CUDA source is enabled, CMake defaults to the compiler
in the discovered Toolkit unless `CMAKE_CUDA_COMPILER` or `CUDACXX` was
explicitly set. An NVIDIA compiler and Toolkit must have matching major and
minor versions. Select both in a fresh or compatible build directory when
multiple Toolkits are installed; matching versions alone do not establish GPU
or application compatibility.

For CUDA workload targets, select the device-code architecture deliberately.
For example, configure a compatible V100 build with
`cmake --preset cuda-gpu -DCMAKE_CUDA_ARCHITECTURES=70`. Without an explicit
value, CMake may use `CUDAARCHS` or a compiler-dependent default rather than the
target GPU. The isolated GEMM comparison fixes its own architecture at 70.
Code-generation architecture does not establish actual SM participation or
saturation; see [Testing](TESTING.md#hardware-and-measurement-evidence).

## CUDA launch admission and examples

The supplied CUDA presets enable the CUDA backend and launch-admission preload:

```sh
cmake --preset cuda-debug
cmake --build --preset cuda-debug -j2
ctest --preset cuda-debug --output-on-failure --no-tests=error
```

`cuda-debug` disables hardware GPU tests and workload executables. To build
the ordinary CUDA examples and enable tests that execute on a GPU, use:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu -j2
```

The preload for these examples is `libglimmer_cuda_interceptor.so`. Continue
with [Usage](USAGE.md) to start a service and run your application, or choose
one of the [examples](../examples/README.md). Hardware tests are a separate
verification step: run `ctest --preset cuda-gpu --output-on-failure --no-tests=error` only when
the device is available for testing.

## Centralized CUDA execution (experimental)

The execution service and its frontend are separate opt-in targets; the CUDA
presets alone do not enable them. To add them to a compatible CUDA build:

```sh
cmake --preset cuda-debug \
  -DGLIMMER_BUILD_CUDA_EXECUTION_SERVICE=ON \
  -DGLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR=ON
cmake --build --preset cuda-debug -j2
ctest --preset cuda-debug --output-on-failure --no-tests=error
```

This produces `build/cuda-debug/bin/glimmer_execution_service` and
`build/cuda-debug/lib/libglimmer_execution_interceptor.so`. Do not substitute
this frontend into the launch-admission examples. It requires an authenticated
service connection and its own workload configuration; a successful build is
not proof that an arbitrary application can use it. See the
[integration requirements](USAGE.md#centralized-execution).

Additional options are deliberately independent:

| Option | Purpose and prerequisite |
| --- | --- |
| `GLIMMER_CUDNN_ROOT` | Explicit installed cuDNN SDK root; requires the CUDA backend. SDK ABI and operation coverage must match the workload. |
| `GLIMMER_NCCL_ROOT` | Explicit installed NCCL SDK root; requires the CUDA backend. |
| `GLIMMER_ENABLE_NCCL_EXECUTION` | Enables NCCL execution routes; requires the NCCL root and both execution targets. Runtime activation is separate. |
| `GLIMMER_BUILD_CUDA_PERSISTENT_GEMM` | Builds the private SM70 CUTLASS provider; requires the CUDA backend, pinned CUTLASS submodule and `nvcc`. Not a generic GPU-architecture option. |
| `GLIMMER_BUILD_GEMM_COMPARISON` | Builds the separate diagnostic GEMM comparison target; requires pinned CUTLASS and is not required for either frontend. |

## Other build configurations

For a single-configuration Debug build with GCC or Clang, enable
`GLIMMER_OPTIMIZE_EXECUTION_HOST=ON` to compile the execution service, its frontend
and their Glimmer host libraries with `-O2` while retaining Debug symbols and
assertions. At least one of the two execution targets must be enabled. For example:

```sh
cmake --preset cuda-debug \
  -DGLIMMER_BUILD_CUDA_EXECUTION_SERVICE=ON \
  -DGLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR=ON \
  -DGLIMMER_OPTIMIZE_EXECUTION_HOST=ON
cmake --build --preset cuda-debug -j2
ctest --preset cuda-debug --output-on-failure --no-tests=error
```

The option defaults to `OFF`; set it back to `OFF` and rebuild to restore ordinary
Debug compilation. It does not change third-party, test-source, assembly or CUDA
device compilation flags. Tests still link the selected production libraries.
The option requires Debug and rejects Release, RelWithDebInfo and multi-config
generators; it does not establish workload compatibility or GPU qualification.

`release` enables optimized core builds; `asan-ubsan` enables runtime sanitizers.
`lint` and `cuda-lint` enable clang-tidy and require it to be installed.
`cuda-perf` is an optimized CUDA build with profiling symbols. Configuration
names do not imply performance qualification.

List available presets with `cmake --list-presets`. See the
[development lifecycle](DEVELOPMENT_LIFECYCLE.md) for contribution checks and
[storage layout](STORAGE_LAYOUT.md) before creating or cleaning build trees.
