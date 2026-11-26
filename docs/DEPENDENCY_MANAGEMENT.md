# Dependency management

[Project home](../README.md) / [Documentation](../docs/README.md) / Dependencies

Dependencies must be minimal, reproducible and appropriately licensed. Add one
only when the standard library and existing project code cannot satisfy a
concrete requirement. Dependency changes require explicit authorization; this
guide does not authorize installation or edits under `3rdparty/`.

## Integration rules

| Class | Rule |
| --- | --- |
| Source dependency | Use a reviewed, exact Git submodule revision under `3rdparty/`, declared in `.gitmodules`. Do not track a branch tip. |
| System dependency | Discover with CMake. Do not copy system libraries into the repository. |
| Vendor SDK | Use an explicitly installed SDK and its matching ABI/provider. Do not vendor NVIDIA headers or binaries. |
| Development tool | Keep formatters, analyzers and profilers out of runtime dependencies. |
| Framework/workload environment | Keep optional, isolated and version-recorded; do not mutate system Python or SDK installations as part of a normal build. |

Normal configure steps must not download dependencies. `FetchContent` needs an
explicit decision explaining why a pinned submodule or system package is
unsuitable. Use `EXCLUDE_FROM_ALL` for third-party build targets unless they
must be built. Do not edit dependencies in place; upstream a fix or obtain
approval for an explicit patch strategy. Keep third-party diagnostics separate
from Glimmer-owned quality gates.

## Existing integration boundaries

CMake builds `spdlog` from its source submodule for components including the
control plane and backend; the scheduling core does not link it directly. The
project uses system threading facilities. CUDA targets, cuDNN/NCCL SDKs and
CUTLASS-based targets are optional. The pinned
CUTLASS integration checks the approved source revision and inputs; do not
bypass those checks. See the [build guide](BUILDING.md) for selection options.

Keep cuDNN major-version profiles in separate compatible builds. Do not point a
qualified build at a different SDK ABI. Actual mapped library/provider identity,
not a header version or successful link alone, determines a validation result's
scope. SDK availability does not establish complete library API support.

## Third-party notices

The source submodules retain their own notices:

- `spdlog`: [MIT license](https://github.com/gabime/spdlog/blob/989d28dd768adef702538634857942db24e1058a/LICENSE), also at `3rdparty/spdlog/LICENSE`.
- Its default bundled `{fmt}`: [license notice](https://github.com/gabime/spdlog/blob/989d28dd768adef702538634857942db24e1058a/include/spdlog/fmt/bundled/fmt.license.rst), also at `3rdparty/spdlog/include/spdlog/fmt/bundled/fmt.license.rst`.
- Optional CUTLASS: [BSD-3-Clause license](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/LICENSE.txt), also at `3rdparty/cutlass/LICENSE.txt`.

The links refer to the pinned revisions; update them when a submodule changes.
Initialize the relevant submodule to read the local files. Other system libraries
and vendor SDKs have their own terms. Review the dependencies actually used
and preserve their required notices when distributing them.

## Adding, updating or removing a dependency

Before adding one, record its concrete purpose, upstream source, exact version
or commit, license, integration method, security/runtime/build/maintenance cost
and verification plan. Update the relevant public architecture or deployment
guide for public API, module or long-term operating-model changes.

For updates, review upstream release, license and compatibility changes and
record old/new revisions. Run the applicable full build, tests, formatting and
sanitizer checks. Remove unused CMake, documentation and submodule entries in
the same logical change. Keep a dependency change independently reviewable and
commit it only on explicit request under the [Git guidelines](GIT_GUIDELINES.md).
