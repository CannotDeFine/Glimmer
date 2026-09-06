#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_priority_matrix.sh [options]

Runs a fixed co-located PyTorch workload across native and priority scheduler
configurations, then writes one summary row per run to a CSV file.

Options:
  --output-dir DIR       result directory (default: temporary directory)
  --repetitions N        repetitions per configuration (default: 3)
  --iterations N         measured iterations per process (default: 20)
  --warmup N             warmup iterations per process (default: 3)
  --batch-size N         PyTorch tensor batch size (default: 32)
  --hidden-size N        MLP hidden size (default: 1024)
  --training-work-units N training work units (default: 2)
  --priority-reserved-slots N slots reserved for high-priority work (default: 0)
  --priority-threshold N minimum priority for reserved slots (default: 100)
  --inference-target-ms N per-iteration inference latency target (optional)
  --build-dir DIR        CUDA build directory (default: build/cuda-gpu)
  --python PATH          Python executable passed to the workload runner
  --trace-timings        enable launch-path timing diagnostics
  --help                 show this message

The scheduler launch batch and the PyTorch tensor batch are recorded as
separate CSV fields. The matrix keeps the tensor batch fixed while changing
only scheduler settings.
EOF
}

output_dir=""
repetitions=3
iterations=20
warmup=3
batch_size=32
hidden_size=1024
training_work_units=2
priority_reserved_slots=0
priority_threshold=100
inference_target_ms=""
build_dir="build/cuda-gpu"
python_executable=""
trace_timings=0

while (($# > 0)); do
    case "$1" in
        --output-dir|--repetitions|--iterations|--warmup|--batch-size|--hidden-size|\
        --training-work-units|--priority-reserved-slots|--priority-threshold|\
        --inference-target-ms|--build-dir|--python)
            if (($# < 2)); then
                echo "$1 requires a value" >&2
                exit 2
            fi
            case "$1" in
                --output-dir) output_dir="$2" ;;
                --repetitions) repetitions="$2" ;;
                --iterations) iterations="$2" ;;
                --warmup) warmup="$2" ;;
                --batch-size) batch_size="$2" ;;
                --hidden-size) hidden_size="$2" ;;
                --training-work-units) training_work_units="$2" ;;
                --priority-reserved-slots) priority_reserved_slots="$2" ;;
                --priority-threshold) priority_threshold="$2" ;;
                --inference-target-ms) inference_target_ms="$2" ;;
                --build-dir) build_dir="$2" ;;
                --python) python_executable="$2" ;;
            esac
            shift 2
            ;;
        --trace-timings)
            trace_timings=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
if [[ ! "$priority_reserved_slots" =~ ^[0-9]+$ ||
    ! "$priority_threshold" =~ ^[0-9]+$ ]] ||
    ! awk -v threshold="$priority_threshold" \
        'BEGIN { exit !(threshold <= 4294967295) }'; then
    echo "priority reservation values are invalid" >&2
    exit 2
fi
if [[ -n "$inference_target_ms" ]] && {
    [[ ! "$inference_target_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    ! awk -v value="$inference_target_ms" 'BEGIN { exit !(value > 0.0) }';
}; then
    echo "--inference-target-ms must be a positive decimal" >&2
    exit 2
fi

for numeric_value in "$repetitions" "$iterations" "$warmup" "$batch_size" \
    "$hidden_size" "$training_work_units"; do
    if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "numeric workload values must be positive integers" >&2
        exit 2
    fi
done

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
runner="$repo_dir/examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh"
if [[ "$build_dir" != /* ]]; then
    build_dir="$repo_dir/$build_dir"
fi

if [[ -z "$output_dir" ]]; then
    output_dir="$(mktemp -d "${TMPDIR:-/tmp}/glimmer-priority-matrix.XXXXXX")"
else
    mkdir -p "$output_dir"
    output_dir="$(cd "$output_dir" && pwd)"
fi

summary_file="$output_dir/summary.csv"
printf '%s\n' \
    "run_id,mode,repetition,tensor_batch_size,hidden_size,max_concurrent_kernels,priority_reserved_slots,priority_threshold,training_launch_batch_size,inference_launch_batch_size,training_mean_ms,training_p50_ms,training_p95_ms,training_p99_ms,training_steps_per_second,inference_mean_ms,inference_p50_ms,inference_p95_ms,inference_p99_ms,inference_steps_per_second,inference_target_ms,inference_target_miss_count,inference_target_miss_ratio,scheduler_total_queue_wait_us,scheduler_max_queue_wait_us,scheduler_total_service_time_us,scheduler_max_service_time_us,overlap_ms,device_uuid,training_pid,inference_pid,measurement_version" \
    >"$summary_file"

read_workload_metric() {
    local role="$1"
    local key="$2"
    local file="$3"
    # The runner reports common-window statistics; the worker log includes
    # its entire interval and would reintroduce the uncontended training tail.
    read_runner_metric "${role}_${key}" "${file%/*}/runner.log"
}

read_runner_metric() {
    local key="$1"
    local file="$2"
    awk -v expected_key="$key" '
        $1 == "pytorch_priority_workload" {
            for (field_idx = 1; field_idx <= NF; ++field_idx) {
                if ($field_idx ~ ("^" expected_key "=")) {
                    sub("^" expected_key "=", "", $field_idx)
                    print $field_idx
                    exit
                }
            }
        }
    ' "$file"
}

require_value() {
    local name="$1"
    local value="$2"
    if [[ -z "$value" ]]; then
        echo "missing $name in matrix result" >&2
        exit 1
    fi
    printf '%s' "$value"
}

run_configuration() {
    local run_id="$1"
    local mode="$2"
    local repetition="$3"
    local max_concurrent_kernels="$4"
    local training_launch_batch_size="$5"
    local inference_launch_batch_size="$6"
    local configuration_reserved_slots="$7"
    local configuration_priority_threshold="$8"
    local run_dir="$output_dir/$run_id"
    local runner_log="$run_dir/runner.log"
    mkdir -p "$run_dir"

    local runner_args=(
        --mode "$mode"
        --iterations "$iterations"
        --warmup "$warmup"
        --batch-size "$batch_size"
        --hidden-size "$hidden_size"
        --training-work-units "$training_work_units"
        --output-dir "$run_dir"
        --build-dir "$build_dir"
    )
    if [[ -n "$inference_target_ms" ]]; then
        runner_args+=(--inference-target-ms "$inference_target_ms")
    fi
    if [[ -n "$python_executable" ]]; then
        runner_args+=(--python "$python_executable")
    fi
    if [[ "$mode" == "priority" ]]; then
        runner_args+=(
            --max-concurrent-kernels "$max_concurrent_kernels"
            --priority-reserved-slots "$configuration_reserved_slots"
            --priority-threshold "$configuration_priority_threshold"
            --training-launch-batch-size "$training_launch_batch_size"
            --inference-launch-batch-size "$inference_launch_batch_size"
        )
    fi
    if ((trace_timings)); then
        runner_args+=(--trace-timings)
    fi

    echo "[matrix] run=$run_id mode=$mode tensor_batch_size=$batch_size max_concurrent=$max_concurrent_kernels training_launch_batch=$training_launch_batch_size"
    if ! "$runner" "${runner_args[@]}" >"$runner_log" 2>&1; then
        cat "$runner_log" >&2
        echo "matrix run failed: $run_id" >&2
        exit 1
    fi

    local training_log="$run_dir/training.log"
    local inference_log="$run_dir/inference.log"
    local training_mean_ms
    training_mean_ms="$(require_value training_mean_ms "$(read_workload_metric training mean_ms "$training_log")")"
    local training_p50_ms
    training_p50_ms="$(require_value training_p50_ms "$(read_workload_metric training p50_ms "$training_log")")"
    local training_p95_ms
    training_p95_ms="$(require_value training_p95_ms "$(read_workload_metric training p95_ms "$training_log")")"
    local training_p99_ms
    training_p99_ms="$(require_value training_p99_ms "$(read_workload_metric training p99_ms "$training_log")")"
    local training_steps_per_second
    training_steps_per_second="$(require_value training_steps_per_second "$(read_workload_metric training steps_per_second "$training_log")")"
    local inference_mean_ms
    inference_mean_ms="$(require_value inference_mean_ms "$(read_workload_metric inference mean_ms "$inference_log")")"
    local inference_p50_ms
    inference_p50_ms="$(require_value inference_p50_ms "$(read_workload_metric inference p50_ms "$inference_log")")"
    local inference_p95_ms
    inference_p95_ms="$(require_value inference_p95_ms "$(read_workload_metric inference p95_ms "$inference_log")")"
    local inference_p99_ms
    inference_p99_ms="$(require_value inference_p99_ms "$(read_workload_metric inference p99_ms "$inference_log")")"
    local inference_steps_per_second
    inference_steps_per_second="$(require_value inference_steps_per_second "$(read_workload_metric inference steps_per_second "$inference_log")")"

    local inference_target_ms_recorded
    inference_target_ms_recorded="$(require_value inference_target_ms "$(read_runner_metric inference_target_ms "$runner_log")")"
    local inference_target_miss_count
    inference_target_miss_count="$(require_value inference_target_miss_count "$(read_runner_metric inference_target_miss_count "$runner_log")")"
    local inference_target_miss_ratio
    inference_target_miss_ratio="$(require_value inference_target_miss_ratio "$(read_runner_metric inference_target_miss_ratio "$runner_log")")"
    local scheduler_total_queue_wait_us
    scheduler_total_queue_wait_us="$(require_value scheduler_total_queue_wait_us "$(read_runner_metric scheduler_total_queue_wait_us "$runner_log")")"
    local scheduler_max_queue_wait_us
    scheduler_max_queue_wait_us="$(require_value scheduler_max_queue_wait_us "$(read_runner_metric scheduler_max_queue_wait_us "$runner_log")")"
    local scheduler_total_service_time_us
    scheduler_total_service_time_us="$(require_value scheduler_total_service_time_us "$(read_runner_metric scheduler_total_service_time_us "$runner_log")")"
    local scheduler_max_service_time_us
    scheduler_max_service_time_us="$(require_value scheduler_max_service_time_us "$(read_runner_metric scheduler_max_service_time_us "$runner_log")")"

    local overlap_ms
    overlap_ms="$(require_value overlap_ms "$(read_runner_metric overlap_ms "$runner_log")")"
    local device_uuid
    device_uuid="$(require_value device_uuid "$(read_runner_metric device_uuid "$runner_log")")"
    local training_pid
    training_pid="$(require_value training_pid "$(read_runner_metric training_pid "$runner_log")")"
    local inference_pid
    inference_pid="$(require_value inference_pid "$(read_runner_metric inference_pid "$runner_log")")"

    printf '%s\n' \
        "$run_id,$mode,$repetition,$batch_size,$hidden_size,$max_concurrent_kernels,$configuration_reserved_slots,$configuration_priority_threshold,$training_launch_batch_size,$inference_launch_batch_size,$training_mean_ms,$training_p50_ms,$training_p95_ms,$training_p99_ms,$training_steps_per_second,$inference_mean_ms,$inference_p50_ms,$inference_p95_ms,$inference_p99_ms,$inference_steps_per_second,$inference_target_ms_recorded,$inference_target_miss_count,$inference_target_miss_ratio,$scheduler_total_queue_wait_us,$scheduler_max_queue_wait_us,$scheduler_total_service_time_us,$scheduler_max_service_time_us,$overlap_ms,$device_uuid,$training_pid,$inference_pid,wall_clock_v2" \
        >>"$summary_file"
}

for repetition in $(seq 1 "$repetitions"); do
    run_configuration "native-r${repetition}" native "$repetition" 0 0 0 0 0
    # The one-slot row is an ordering baseline. A reserved slot would consume
    # its entire capacity and prevent the low-priority training task from
    # making progress, so reservation is exercised by the two-slot rows below.
    run_configuration "priority-c1-b1-r${repetition}" priority "$repetition" 1 1 1 0 "$priority_threshold"
    run_configuration "priority-c2-b1-r${repetition}" priority "$repetition" 2 1 1 "$priority_reserved_slots" "$priority_threshold"
    run_configuration "priority-c2-b4-r${repetition}" priority "$repetition" 2 4 1 "$priority_reserved_slots" "$priority_threshold"
done

echo "priority_matrix status=ok repetitions=$repetitions configurations=4 rows=$((repetitions * 4)) output=$summary_file"
