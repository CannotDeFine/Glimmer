# Building Glimmer

[Project home](../README.md) / [Documentation](README.md) / Building

Glimmer targets Linux and uses C++20. Run the commands below from the repository
root. Initialize the required source dependency after cloning:

```sh
git submodule update --init --recursive -- 3rdparty/spdlog
```

The core build requires CMake 3.20 or newer, a C++20 compiler, Git and Python 3.
The supplied presets use Make. To build and test the host components:

```sh
cmake --preset debug
cmake --build --preset debug -j2
ctest --preset debug --output-on-failure
```

To build the CUDA launch-admission library, install a compatible CUDA Toolkit:

```sh
cmake --preset cuda-debug
cmake --build --preset cuda-debug -j2 --target \
  glimmer_control_service glimmer_control_client glimmer_cuda_interceptor
```

Running CUDA applications also requires a compatible NVIDIA driver and GPU.
Continue with the [usage guide](USAGE.md) to run an application. The
[testing guide](TESTING.md) describes additional checks and GPU tests.
Existing CMake build directories retain their configured options; inspect them
before reusing or reconfiguring them. See [storage layout](STORAGE_LAYOUT.md)
for build-directory and artifact retention rules.
