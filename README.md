# Glimmer

Glimmer is a Linux scheduler-based shared GPU project. It schedules work from
multiple tenants or workloads on a single physical GPU to provide fair,
controllable, and observable GPU sharing without hardware-level vGPU
partitioning.

## Build

Glimmer is a self-contained Linux C++20 project built with CMake 3.20 or
newer. Its CMake configuration uses only paths relative to the source tree, so
the repository can be moved or cloned to any location without reconfiguration.

## Prerequisites

- CMake 3.20+
- A C++20-capable compiler: GCC or Clang
- Git, including submodule support
- CUDA Toolkit and an NVIDIA driver (only for CUDA interceptor builds)

Clone the repository with its third-party dependencies:

```sh
git clone --recurse-submodules <repository-url>
```

For an existing clone, initialize them with:

```sh
git submodule update --init --recursive
```

### Build

Use the `debug` preset for day-to-day development:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

The build script runs the configure and build steps:

```sh
./scripts/build.sh
```

Additional presets are available for optimized builds, runtime sanitizers, and
static analysis:

```sh
cmake --preset release
cmake --preset asan-ubsan
cmake --preset lint
cmake --preset cuda-lint
```

The `lint` preset requires `clang-tidy`. The executable is written to `bin/`
inside the selected build directory.

Run the complete pre-commit verification with:

```sh
./scripts/check.sh
```

This runs all available build, test, sanitizer, formatting, and static-analysis
checks. The CUDA lint checks are included when the CUDA Toolkit is installed.

When developing the CUDA interceptor, use its dedicated presets. They use
separate build directories so CUDA-enabled and non-CUDA artifacts cannot share
stale CMake cache state:

```sh
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug --output-on-failure
```

For clang-tidy coverage of both the interceptor and its tests, use
`cuda-lint` or run `./scripts/check.sh`.

The CUDA GPU test preset requires a compatible NVIDIA driver and GPU:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu
ctest --preset cuda-gpu --output-on-failure
```

The root `compile_commands.json` link follows the most recently built preset.
Use `cuda-lint` last when editor diagnostics must include CUDA interceptor files
and Toolkit include paths.

## Contributing

Follow the project's [Git guidelines](docs/GIT_GUIDELINES.md) when contributing.
