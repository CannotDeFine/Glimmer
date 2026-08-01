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

```sh
./scripts/build.sh
```

The equivalent CMake commands are:

```sh
cmake -S . -B build
cmake --build build
```

An out-of-tree build directory can also be placed anywhere:

```sh
cmake -S /path/to/Glimmer -B /path/to/build/glimmer
cmake --build /path/to/build/glimmer
```

The executable is written to `bin/` inside the selected build directory.

## Contributing

Follow the project's [Git guidelines](docs/GIT_GUIDELINES.md) when contributing.
