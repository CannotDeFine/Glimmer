# Dependency Management

## Purpose

Dependencies must be minimal, reproducible, licensed appropriately, and easy
to update. A dependency is accepted only when the C++ standard library and
existing project code cannot meet a concrete requirement.

## Dependency classes

| Class | Examples | Integration rule |
| --- | --- | --- |
| Source dependency | `spdlog`, a future test library | Use a Git submodule under `3rdparty/<name>` pinned to an exact commit. |
| System dependency | Threads, `libdl` | Discover with CMake; do not copy system libraries into the repository. |
| Vendor SDK | CUDA Toolkit, NVML | Require an installed SDK and discover it with CMake. Do not vendor NVIDIA binaries or headers. |
| Development tool | clang-format, clang-tidy | Use only for development or CI; do not make it a runtime dependency. |

`FetchContent` must not download dependencies during a normal configure step.
Use it only after an explicit decision records why a pinned submodule or system
package is unsuitable.

## CUDA interceptor SDK dependency

The optional CUDA interceptor and explicit CUDA task backend require NVIDIA
CUDA Toolkit headers for the CUDA Driver API ABI declarations. The standalone
CUDA workload harnesses also require the Toolkit to compile their Runtime
programs. The Toolkit is a vendor SDK dependency; it is discovered with
`find_package(CUDAToolkit)` when `GLIMMER_BUILD_CUDA_INTERCEPTOR=ON`,
`GLIMMER_BUILD_CUDA_BACKEND=ON`, or `GLIMMER_BUILD_CUDA_WORKLOAD=ON`. CUDA
headers and binaries are not vendored in this repository.

The interceptor resolves `libcuda.so.1` at runtime with the Linux dynamic
loader rather than linking directly to the NVIDIA Driver. On split loader/vendor
installations, such as WSL, it also discovers the already-loaded vendor
`libcuda.so.1.1` object and resolves symbols from that object instead of the
loader shim. This keeps the standard build and no-GPU tests independent of a
CUDA installation while avoiding recursive Driver-internal lookups. CUDA
integration tests must state the Toolkit, Driver, and loader layout they
exercise when the layout is non-standard.

## Introducing a dependency

Before adding one, document:

1. The concrete capability it provides and why existing code is insufficient.
2. Its source repository, exact version or commit, and license.
3. Its dependency class and CMake integration method.
4. Its runtime, build, security, and maintenance cost.
5. The tests or build checks that verify the integration.

Record a decision under `docs/decisions/` when the dependency affects a public
API, module boundary, runtime deployment, or long-term operating model.

## Source dependency rules

- Add source dependencies as submodules under `3rdparty/` and declare them in
  `.gitmodules`.
- Pin the submodule to a reviewed commit. Do not track a branch tip.
- Configure third-party code with `EXCLUDE_FROM_ALL` unless it must be built by
  the project.
- Do not edit a dependency in place. Upstream a fix or record an explicit,
  reviewable patch strategy.
- Keep third-party warnings and static-analysis results out of Glimmer's own
  quality gate unless the dependency is intentionally maintained locally.

## Updates and removal

- Make each dependency update a dedicated commit with its reason and the old
  and new revision in the commit body when useful.
- Review upstream release notes, license changes, and compatibility notes.
- Re-run the full build, tests, formatting check, and applicable sanitizers.
- Remove unused dependencies and their CMake, documentation, and submodule
  entries in the same logical change.
