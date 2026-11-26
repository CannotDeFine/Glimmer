# Isolated GEMM comparison

[Project home](../../README.md) / [Documentation](../../docs/README.md) / [Examples](../README.md) / GEMM comparison

This optional target compares native cuBLAS, ordinary CUTLASS and persistent
CUTLASS on nine FP32 GEMM shapes. It does not change application interception
or production routing. The operations omit bias and split-K; they are not a
complete model or a scheduling-performance benchmark.

Both CUTLASS paths use SM70 SIMT with matched arithmetic and epilogues.
Persistent execution assigns tiles by worker stride and retains completed tiles
across a physically drained stop/resume. Audit launches are separate from
ordinary timing launches.

## Build

Requirements: Linux, a V100 with 80 SMs, CMake 3.25.2 or newer, a CUDA 12.6
toolchain and the pinned CUTLASS v3.5.1 submodule at
`f7b19de32c5d1f3cedfc735c2849f12b537522ee`.
The source and license checks run during configuration and compilation;
BSD-3-Clause terms remain in `3rdparty/cutlass/LICENSE.txt` and its headers.
No dependency is downloaded during CMake configuration.

From the repository root, use the candidate slot only if it is empty or already
contains this standalone configuration. Preserve any existing build or evidence
before repurposing the slot:

```sh
git submodule update --init 3rdparty/cutlass
cmake -S examples/gemm_comparison -B build/candidate \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
cmake --build build/candidate -j2
ctest --test-dir build/candidate --output-on-failure --no-tests=error
```

The collector currently expects the CUDA 12.6 runtime and device tools under
`/usr/local/cuda-12.6`; changing the build paths alone does not change its runtime
environment. This standalone build does not change the root compile-database link.
Alternatively, enable
`GLIMMER_BUILD_GEMM_COMPARISON=ON` in a compatible root build and build target
`glimmer_gemm_comparison`. See [storage rules](../../docs/STORAGE_LAYOUT.md)
before rebuilding or cleaning a retained configuration.

## Check correctness

Run on an otherwise idle device, without preload, profiler injection or changed
MPS/clock policy. The collector rejects foreign compute processes and retains
its exact commands, identities, outputs and failures in a fresh attempt under
`artifacts/runs/gemm-comparison/`.

```sh
python3 -B examples/gemm_comparison/run_comparison.py \
  --binary build/candidate/glimmer_gemm_comparison \
  --cutlass-root 3rdparty/cutlass --check-only
```

Checks cover beta zero/one, NaN old C for beta zero, all output elements,
unchanged operands and guards, sampled double-precision oracles, bitwise
ordinary/persistent agreement, pre-requested zero work, partial stop and
exactly-once resumption. Native comparisons use rtol=2e-4 and atol=2e-5.

Add `--device-tool memcheck` or `--device-tool synccheck` for device checks.
For host sanitizers, configure a separate compatible build with
`GLIMMER_GEMM_ENABLE_SANITIZERS=ON` and run with `--check-only --host-sanitizers`.
Neither sanitizer nor profiler runs provide ordinary timing evidence.

## Qualify full-device participation

The collector requires useful compute on all 80 SMs for every compared case.
Worker counts and occupancy capacity alone do not meet this requirement.
Untimed correctness may succeed while native coverage remains unknown; that
is not a qualified three-way comparison.

Collect counters separately before timing:

```sh
python3 -B examples/gemm_comparison/run_comparison.py \
  --binary build/candidate/glimmer_gemm_comparison \
  --cutlass-root 3rdparty/cutlass --collect-native-sm
```

The collector uses Nsight Compute at `/opt/nvidia/nsight-compute/2024.3.2/ncu`
and its adjacent sections directory; it does not discover another installation
from `PATH`. Its `ncu_sha256` records only that file's contents, not a
launcher's underlying executable or the section files. For reproducible
captures, record and verify those identities separately before and after
collection; the collector does not detect changes to them. Counter access is
also required.
If local policy requires elevated counter access, obtain authorization before
using `--sudo-ncu`; ordinary usage does not require sudo. Missing permissions,
unsupported metrics, incomplete replays or partial SM coverage fail
qualification. Do not change device policy to obtain a pass.

Every application replay must complete correctness checks. The collector
classifies full-work, pre-retreat, partial-stop and resume launches separately,
including audited and normal kernel specializations. Native coverage cannot be
inferred from CUTLASS instrumentation. Spatial participation, active-cycle
utilization, occupancy and throughput are separate observations; full SM
participation does not establish sustained saturation.

## Collect timings

Supply the successful counter report produced for the same inputs and binary:

```sh
python3 -B examples/gemm_comparison/run_comparison.py \
  --binary build/candidate/glimmer_gemm_comparison \
  --cutlass-root 3rdparty/cutlass \
  --native-sm-evidence /absolute/path/to/capture/attempt/report.json
```

Evidence must be within 24 hours and match GPU/driver, executable, embedded
source, CUTLASS and native library identities. Complete raw files must remain
unchanged. The evidence verifier does not independently verify the profiler's
underlying executable or section files. The collector revalidates coverage after
timing; stale, altered or partial evidence fails. Do not rebuild the binary
between capture and timing.

Timing uses three warm samples and 21 CUDA-event samples per mode with rotating
order. It measures complete GEMM boundaries, excluding setup, progress reset and
validation. Raw samples and medians are retained. `--diagnostic-only` explicitly
allows unqualified measurements but never turns them into formal comparisons.
A coverage pass does not establish application, priority, saturation or speedup
acceptance.
