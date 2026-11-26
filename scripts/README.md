# Scripts

[Project home](../README.md) / [Documentation](../docs/README.md) / Scripts

This directory holds development tools and the maintained collectors behind
recorded experiments (see [storage layout](../docs/STORAGE_LAYOUT.md)). Files
stay in one flat directory because collectors import each other and
`artifact_paths` by module name and several verify exact source paths or
hashes; moving a file is not a cosmetic change.

## Development tools

| File | Purpose |
| --- | --- |
| `build.sh` | Configure and build the `debug` preset after slot and storage checks. |
| `check.sh` | Run the available build, test, format, lint and no-GPU CUDA checks. |
| `build_safety.sh` | Slot, compilation-database and build-budget guards used by the two scripts above. |
| `artifact_paths.py` | Create fresh experiment attempts outside reusable build trees. |
| `__init__.py` | Allows `import scripts.artifact_paths`. |

## Experiment collectors

Unless stated otherwise these are offline or diagnostic tools for specific
recorded experiments, not general benchmarks. Their output is evidence only
under the conditions each script enforces.

**Offline evidence audits** (no GPU, no workload):
`aggregate_sm_audit.py`, `application_priority_audit.py`,
`execution_application_receipt.py`, `execution_arrival_audit.py`,
`execution_copy_audit.py`, `execution_cycle_witness.py`,
`execution_priority_audit.py`, `execution_semantic_audit.py`,
`audit_bert_overhead_fragment.py`, `bert_timeline_audit.py`,
`bert_timeline_protocol_probe.py` (CUDA-free protocol check).

**BERT service diagnostics**: `run_bert_*.py` entry points with their helpers
`bert_device_capture.py`, `bert_device_observer.py`, `bert_host_probe.py`,
`bert_sgemm_coverage_candidates.py` and the `analyze_bert_*.py` analyzers.

**Nsight Compute capture chain**: `bert_native_ncu_protocol.py`,
`bert_ncu_*.py`, `bert_sgemm_ncu_process.py`, `run_bert_native_ncu_capture.py`,
`run_bert_sgemm_ncu_capture.py`, `run_execution_ncu_init_probe.py`,
`probe_ncu_replay_process.py`.

**CPU sampling**: `direct_cpu_window.cc` (optional preload),
`bert_direct_cpu.py`, `qualify_direct_cpu.py`, `analyze_direct_cpu.py`,
`run_bert_direct_cpu_numerical.py`.

**Host compiler pairs**: `build_execution_host_pair.py`, `execution_host_abi.cc`,
`bert_ncu_host_pair.py` and `run_bert_host_pair_*.py`.

**cuDNN RNN validation**: `run_rnn_v8_validation.py`, `rnn_v8_audit.py`,
`cudnn8_legacy_audit.py`.

## Hardware binding

The collectors were written for the earlier V100 host (80 SMs, CUDA 12.6,
Nsight Compute 2024.3.2). Scripts that check the GPU name or UUID, require 80
participating SMs, fix 320 persistent workers, match `volta_sgemm` kernels or
use fixed SDK paths refuse to run or do not apply on another GPU. Worker
geometry and these checks must be re-calibrated by measurement on a new GPU
model, not derived from its SM count. This applies to `bert_device_capture.py`,
`bert_device_observer.py`, `bert_native_ncu_protocol.py`,
`bert_ncu_current_sdk.py`, `bert_ncu_host_pair.py`, `bert_ncu_replay_exec.py`,
`bert_sgemm_coverage_candidates.py`, `execution_arrival_audit.py`,
`execution_cycle_witness.py`, `probe_ncu_replay_process.py`,
`run_execution_ncu_init_probe.py`, `run_rnn_v8_validation.py` and many
`run_bert_*.py` entry points; check a script before running it elsewhere.

## Source-layout assumptions

`run_bert_delivery_boundary.py` and `build_execution_host_pair.py` treat
`src/backend/cuda/cuda_execution_session.cc` as the only CUDA session
translation unit. The session implementation is now split into
`cuda_execution_session.cc`, `cuda_execution_session_blas.cc` and
`cuda_execution_session_dnn.cc` with private state in
`internal/cuda_execution_session_state.h`; update their object selection and
link-map checks before reusing them.

## Entry points without tests

These are run by hand and have no CTest coverage: `bert_direct_cpu.py`,
`build_execution_host_pair.py`, `probe_ncu_replay_process.py`,
`qualify_direct_cpu.py`, `analyze_direct_cpu.py`,
`run_bert_direct_cpu_numerical.py`, `run_bert_frontend_lifecycle.py`,
`run_bert_frontend_wait.py`, `run_bert_host_pair_timing.py`,
`run_bert_request_lifecycle.py`, `run_bert_service_optimization.py`,
`run_bert_stage_candidate.py`, `run_bert_stage_diagnostic.py`,
`analyze_bert_frontend_lifecycle.py`, `analyze_bert_frontend_wait.py`,
`analyze_bert_host_range_optimization.py`, `analyze_bert_request_lifecycle.py`
and `analyze_bert_service_stages.py`.
