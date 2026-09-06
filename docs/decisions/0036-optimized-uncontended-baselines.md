# 0036: Measure optimized, uncontended interception and scheduling overhead

Status: Accepted

## Context

The CUDA GPU preset inherits Debug. Existing common-window comparisons measure
sharing interference and net scheduling benefit, but do not separate preload,
memory-quota, local-admission, and remote-admission cost without competition.
Faster control requests alone have not demonstrated better inference latency.

## Decision

Add an independent `cuda-perf` preset using RelWithDebInfo, with the same CUDA
features and tests as `cuda-gpu`. Do not change the development preset in place.
The root compilation database continues to follow the most recently built
preset. Performance measurements require an optimized, non-sanitized build.

Keep a separate overhead runner in the existing PyTorch example. It reuses
the workload and the wall-clock duration accounting from ADR 0035, and runs
training and inference separately in native, preload-only, quota-only,
quota-plus-local-scheduler, and quota-plus-remote-scheduler configurations.
All quota-enabled runs use the same explicit quota. Launch batch and slot
limits are fixed; tensor batch and model mathematics do not change.

Repeat runs with rotating order and retain commands, selected environment,
build configuration, source/binary hashes, raw samples, sample counts, and
per-run and aggregate metrics. Detailed tracing is disabled. Reject invalid
measurements, missing repetitions, changed artifacts or GPU/framework identity,
and workload or service failure. Check the requested memory view outside the
measurement window. Remote runs must show completed scheduler admissions.
No passing performance threshold or causal cost decomposition is implied by
successful measurement collection.

## Scope, risk, and rollback

This change adds build and measurement support only. It does not change CUDA
ABI coverage, quota accounting, scheduler policy, launch gates, or transport.
The optional workload memory-view check runs before model setup and timing.
Existing runners retain their defaults. Reverting this preset, runner, tests,
and documentation does not revert runtime behavior.

An optimized-build prerequisite exposed during verification is fixed alongside
the baseline: libc's `nonnull` declaration allowed the compiler to remove the
existing `dlsym` null-name check. The implementation now uses a distinct source
identifier exported under the same ELF name. The test resolves that boundary
dynamically rather than calling libc's nonnull declaration with a null value.
The old optimized library fails this regression; normal symbol routing and
the exported ABI remain unchanged. This defensive rejection is not permission
for applications to violate the libc API contract.

## Verification

- Build and run CTest with the optimized preset, including real CUDA tests
  when available; run the standard project checks.
- `scripts/check.sh` builds `cuda-perf` and runs its no-GPU tests when `nvcc`
  is installed, excluding `*_gpu_test` entries. The null-name regression must
  remain part of both Debug and optimized checks.
- Standard-library tests cover configuration/environment construction, build
  rejection, metric validation, complete repetitions, identity consistency,
  failed runs, and remote-service evidence and cleanup.
- Run repeated uncontended GPU experiments with no simultaneous Glimmer build,
  test suite, or benchmark. Report unavailable hardware explicitly.

## Limitations and next step

These closed-loop MLP steps are not a serving benchmark or proof of hardware
kernel overlap. Differences between modes include their interactions; they are
not an additive decomposition of RPC, event, and driver costs. Whole-process
remote counters include initialization, warmup, and drain. Run the existing
common-window co-location comparison separately with the same optimized build.
GPU timelines and a local-admission prototype are later steps, not results of
this change. No hard deadline or general workload support is claimed.
