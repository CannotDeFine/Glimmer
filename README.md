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
```

The `lint` preset requires `clang-tidy`. The executable is written to `bin/`
inside the selected build directory.

## Contributing

Follow the project's [Git guidelines](docs/GIT_GUIDELINES.md) when contributing.
