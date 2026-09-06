#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_pytorch_priority.sh [options]

Runs one inference and one training process through the same CUDA device. Both
processes synchronize after CUDA warmup. The runner verifies their CUDA device
UUIDs, process IDs, and overlapping wall-clock execution intervals.

Options:
  --mode native|priority          execution mode (default: priority)
  --build-dir DIR                 CUDA build directory (default: build/cuda-gpu)
  --output-dir DIR                output directory
  --python PATH                   Python executable (default: local venv or python3)
  --iterations N                  measured iterations per process (default: 20)
  --duration-seconds N            shared wall-clock duration; overrides iterations
  --warmup N                      warmup iterations per process (default: 3)
  --batch-size N                  batch size (default: 32)
  --hidden-size N                 MLP hidden size (default: 1024)
  --training-work-units N         training work units (default: 2)
  --max-concurrent-kernels N      scheduler concurrency window (default: 1)
  --priority-reserved-slots N     slots reserved for high-priority work (default: 0)
  --priority-threshold N          minimum priority for reserved slots (default: 100)
  --adaptive-slo-target-queue-us N
                                  enable queue-wait feedback (priority mode only)
  --adaptive-slo-window N         feedback samples per window (default: 32)
  --adaptive-slo-max-reserved-slots N
                                  feedback reservation cap (optional)
  --training-launch-batch-size N  training launches per scheduler lease (default: 1)
  --inference-launch-batch-size N inference launches per scheduler lease (default: 1)
  --inference-target-ms N         per-iteration inference latency target (optional)
  --max-inference-target-miss-ratio R
                                  fail when target miss ratio exceeds R (0..1)
  --trace-timings                 enable launch-path timing diagnostics
  --trace-scheduler               enable service-side dispatch order diagnostics
  --help                          show this message
EOF
}

mode="priority"
build_dir="build/cuda-gpu"
output_dir=""
python_executable=""
iterations=20
duration_seconds=""
warmup=3
batch_size=32
hidden_size=1024
training_work_units=2
max_concurrent_kernels=1
priority_reserved_slots=0
priority_threshold=100
adaptive_slo_target_queue_us=""
adaptive_slo_window=32
adaptive_slo_max_reserved_slots=""
training_launch_batch_size=1
inference_launch_batch_size=1
inference_target_ms=""
max_inference_target_miss_ratio=""
trace_timings=0
trace_scheduler=0

while (($# > 0)); do
    case "$1" in
        --mode)
            if (($# < 2)); then
                echo "--mode requires a value" >&2
                exit 2
            fi
            mode="$2"
            shift 2
            ;;
        --build-dir)
            if (($# < 2)); then
                echo "--build-dir requires a value" >&2
                exit 2
            fi
            build_dir="$2"
            shift 2
            ;;
        --output-dir)
            if (($# < 2)); then
                echo "--output-dir requires a value" >&2
                exit 2
            fi
            output_dir="$2"
            shift 2
            ;;
        --python)
            if (($# < 2)); then
                echo "--python requires a value" >&2
                exit 2
            fi
            python_executable="$2"
            shift 2
            ;;
        --duration-seconds)
            if (($# < 2)); then
                echo "--duration-seconds requires a value" >&2
                exit 2
            fi
            duration_seconds="$2"
            shift 2
            ;;
        --iterations)
            if (($# < 2)); then
                echo "--iterations requires a value" >&2
                exit 2
            fi
            iterations="$2"
            shift 2
            ;;
        --warmup)
            if (($# < 2)); then
                echo "--warmup requires a value" >&2
                exit 2
            fi
            warmup="$2"
            shift 2
            ;;
        --batch-size)
            if (($# < 2)); then
                echo "--batch-size requires a value" >&2
                exit 2
            fi
            batch_size="$2"
            shift 2
            ;;
        --hidden-size)
            if (($# < 2)); then
                echo "--hidden-size requires a value" >&2
                exit 2
            fi
            hidden_size="$2"
            shift 2
            ;;
        --training-work-units)
            if (($# < 2)); then
                echo "--training-work-units requires a value" >&2
                exit 2
            fi
            training_work_units="$2"
            shift 2
            ;;
        --max-concurrent-kernels)
            if (($# < 2)); then
                echo "--max-concurrent-kernels requires a value" >&2
                exit 2
            fi
            max_concurrent_kernels="$2"
            shift 2
            ;;
        --priority-reserved-slots)
            if (($# < 2)); then
                echo "--priority-reserved-slots requires a value" >&2
                exit 2
            fi
            priority_reserved_slots="$2"
            shift 2
            ;;
        --priority-threshold)
            if (($# < 2)); then
                echo "--priority-threshold requires a value" >&2
                exit 2
            fi
            priority_threshold="$2"
            shift 2
            ;;
        --adaptive-slo-target-queue-us)
            if (($# < 2)); then
                echo "--adaptive-slo-target-queue-us requires a value" >&2
                exit 2
            fi
            adaptive_slo_target_queue_us="$2"
            shift 2
            ;;
        --adaptive-slo-window)
            if (($# < 2)); then
                echo "--adaptive-slo-window requires a value" >&2
                exit 2
            fi
            adaptive_slo_window="$2"
            shift 2
            ;;
        --adaptive-slo-max-reserved-slots)
            if (($# < 2)); then
                echo "--adaptive-slo-max-reserved-slots requires a value" >&2
                exit 2
            fi
            adaptive_slo_max_reserved_slots="$2"
            shift 2
            ;;
        --training-launch-batch-size)
            if (($# < 2)); then
                echo "--training-launch-batch-size requires a value" >&2
                exit 2
            fi
            training_launch_batch_size="$2"
            shift 2
            ;;
        --inference-launch-batch-size)
            if (($# < 2)); then
                echo "--inference-launch-batch-size requires a value" >&2
                exit 2
            fi
            inference_launch_batch_size="$2"
            shift 2
            ;;
        --inference-target-ms)
            if (($# < 2)); then
                echo "--inference-target-ms requires a value" >&2
                exit 2
            fi
            inference_target_ms="$2"
            shift 2
            ;;
        --max-inference-target-miss-ratio)
            if (($# < 2)); then
                echo "--max-inference-target-miss-ratio requires a value" >&2
                exit 2
            fi
            max_inference_target_miss_ratio="$2"
            shift 2
            ;;
        --trace-timings)
            trace_timings=1
            shift
            ;;
        --trace-scheduler)
            trace_scheduler=1
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
if [[ -n "$duration_seconds" ]] && {
    [[ ! "$duration_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    ! awk -v value="$duration_seconds" 'BEGIN { exit !(value > 0.0 && value <= 3600.0) }';
}; then
    echo "--duration-seconds must be a decimal in (0, 3600]" >&2
    exit 2
fi

if [[ "$mode" != "native" && "$mode" != "priority" ]]; then
    echo "invalid mode: $mode" >&2
    usage >&2
    exit 2
fi
for numeric_value in "$iterations" "$warmup" "$batch_size" "$hidden_size" \
    "$training_work_units" "$max_concurrent_kernels" "$training_launch_batch_size" \
    "$inference_launch_batch_size"; do
    if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
        echo "numeric workload values must be positive integers" >&2
        exit 2
    fi
done
if [[ -n "$inference_target_ms" ]] && {
    [[ ! "$inference_target_ms" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    ! awk -v value="$inference_target_ms" 'BEGIN { exit !(value > 0.0) }';
}; then
    echo "--inference-target-ms must be a positive decimal" >&2
    exit 2
fi
if [[ -n "$max_inference_target_miss_ratio" ]] && {
    [[ ! "$max_inference_target_miss_ratio" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    ! awk -v value="$max_inference_target_miss_ratio" \
        'BEGIN { exit !(value >= 0.0 && value <= 1.0) }';
}; then
    echo "--max-inference-target-miss-ratio must be a decimal in [0, 1]" >&2
    exit 2
fi
if [[ -n "$max_inference_target_miss_ratio" && -z "$inference_target_ms" ]]; then
    echo "--max-inference-target-miss-ratio requires --inference-target-ms" >&2
    exit 2
fi
if [[ ! "$priority_reserved_slots" =~ ^[0-9]+$ ||
    ! "$priority_threshold" =~ ^[0-9]+$ ]] ||
    ((priority_reserved_slots > max_concurrent_kernels)) ||
    ! awk -v threshold="$priority_threshold" \
        'BEGIN { exit !(threshold <= 4294967295) }'; then
    echo "priority reservation values are invalid" >&2
    exit 2
fi
if [[ -n "$adaptive_slo_target_queue_us" ]]; then
    if [[ "$mode" != "priority" || ! "$adaptive_slo_target_queue_us" =~ ^[1-9][0-9]*$ ||
        ! "$adaptive_slo_window" =~ ^[1-9][0-9]*$ ]]; then
        echo "adaptive SLO feedback requires priority mode and positive integer target/window values" >&2
        exit 2
    fi
    if [[ -n "$adaptive_slo_max_reserved_slots" &&
        ! "$adaptive_slo_max_reserved_slots" =~ ^[0-9]+$ ]]; then
        echo "--adaptive-slo-max-reserved-slots must be a non-negative integer" >&2
        exit 2
    fi
fi
if [[ -n "$adaptive_slo_max_reserved_slots" && -z "$adaptive_slo_target_queue_us" ]]; then
    echo "--adaptive-slo-max-reserved-slots requires --adaptive-slo-target-queue-us" >&2
    exit 2
fi
if [[ -n "$adaptive_slo_target_queue_us" && -n "$adaptive_slo_max_reserved_slots" &&
    "$adaptive_slo_max_reserved_slots" -lt "$priority_reserved_slots" ]]; then
    echo "adaptive SLO reservation cap must be at least --priority-reserved-slots" >&2
    exit 2
fi
if [[ -n "$adaptive_slo_target_queue_us" ]]; then
    if [[ -z "$adaptive_slo_max_reserved_slots" && "$max_concurrent_kernels" -lt 2 ]]; then
        echo "adaptive SLO feedback requires at least two concurrent kernels when no cap is provided" >&2
        exit 2
    fi
    if [[ -n "$adaptive_slo_max_reserved_slots" &&
        "$adaptive_slo_max_reserved_slots" -gt "$max_concurrent_kernels" ]]; then
        echo "adaptive SLO reservation cap cannot exceed --max-concurrent-kernels" >&2
        exit 2
    fi
fi
if [[ "$mode" == "priority" && "$priority_reserved_slots" -ge "$max_concurrent_kernels" &&
    "$priority_threshold" -gt 10 ]]; then
    echo "the co-location runner requires at least one unreserved slot for training" >&2
    echo "decrease --priority-reserved-slots or increase --max-concurrent-kernels" >&2
    exit 2
fi
if [[ "$mode" == "priority" && "$max_concurrent_kernels" -eq 1 &&
    "$training_launch_batch_size" -gt 1 ]]; then
    echo "training launch batches larger than one require at least two concurrent kernels in the co-location runner" >&2
    echo "increase --max-concurrent-kernels or use --training-launch-batch-size 1" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "$build_dir" != /* ]]; then
    build_dir="$repo_dir/$build_dir"
fi
script="$repo_dir/examples/framework_workloads/pytorch_smoke/pytorch_smoke.py"
local_python="$repo_dir/examples/framework_workloads/pytorch_smoke/.venv/bin/python"
if [[ -z "$python_executable" ]]; then
    python_executable="$local_python"
    if [[ ! -x "$python_executable" ]]; then
        python_executable="python3"
    fi
fi
if [[ "$python_executable" == */* ]]; then
    if [[ ! -x "$python_executable" ]]; then
        echo "python executable is not available: $python_executable" >&2
        exit 1
    fi
elif ! command -v "$python_executable" >/dev/null 2>&1; then
    echo "python executable is not available: $python_executable" >&2
    exit 1
fi

interceptor="$build_dir/lib/libglimmer_cuda_interceptor.so"
service="$build_dir/bin/glimmer_control_service"
client="$build_dir/bin/glimmer_control_client"
if [[ "$mode" == "priority" ]]; then
    for required_file in "$interceptor" "$service" "$client"; do
        if [[ ! -f "$required_file" ]]; then
            echo "required build artifact is missing: $required_file" >&2
            echo "build with: cmake --build --preset cuda-gpu -j2" >&2
            exit 1
        fi
    done
fi

if [[ -z "$output_dir" ]]; then
    output_dir="$repo_dir/examples/framework_workloads/pytorch_smoke/output/co-location-$mode"
fi
mkdir -p "$output_dir"

socket_dir=""
socket_path=""
service_pid=""
training_pid=""
inference_pid=""
if [[ "$mode" == "priority" ]]; then
    socket_dir="$(mktemp -d "${TMPDIR:-/tmp}/glimmer-pytorch-priority.XXXXXX")"
    socket_path="$socket_dir/control.sock"
fi

training_ready="$output_dir/training.ready"
inference_ready="$output_dir/inference.ready"
start_file="$output_dir/start.signal"
training_status_file="$output_dir/training.status"
inference_status_file="$output_dir/inference.status"
training_log="$output_dir/training.log"
inference_log="$output_dir/inference.log"
service_log="$output_dir/control-service.log"
metrics_file="$output_dir/control-metrics.txt"
training_csv="$output_dir/training.csv"
inference_csv="$output_dir/inference.csv"
rm -f "$training_ready" "$inference_ready" "$start_file" \
    "$training_status_file" "$inference_status_file" "$training_log" \
    "$inference_log" "$service_log" "$training_csv" "$inference_csv" \
    "$metrics_file"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$training_pid" ]]; then
        kill "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    if [[ -n "$inference_pid" ]]; then
        kill "$inference_pid" 2>/dev/null || true
        wait "$inference_pid" 2>/dev/null || true
    fi
    if [[ -n "$service_pid" ]]; then
        kill "$service_pid" 2>/dev/null || true
        wait "$service_pid" 2>/dev/null || true
    fi
    if [[ -n "$socket_dir" ]]; then
        rm -f "$socket_path"
        rmdir "$socket_dir" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$mode" == "priority" ]]; then
    service_arguments=(
        --socket "$socket_path"
        --quota-bytes 4294967296
        --execution-mode remote
        --lease-timeout-ms 5000
        --max-queued-tasks 256
        --max-concurrent-tasks "$max_concurrent_kernels"
        --scheduler-policy priority
        --priority-reserved-slots "$priority_reserved_slots"
        --priority-threshold "$priority_threshold"
        --bind-leases-to-process
    )
    if [[ -n "$adaptive_slo_target_queue_us" ]]; then
        service_arguments+=(--adaptive-slo-target-queue-us "$adaptive_slo_target_queue_us"
            --adaptive-slo-window "$adaptive_slo_window")
        if [[ -n "$adaptive_slo_max_reserved_slots" ]]; then
            service_arguments+=(--adaptive-slo-max-reserved-slots "$adaptive_slo_max_reserved_slots")
        fi
    fi
    if ((trace_scheduler)); then
        service_arguments+=(--trace-scheduler)
    fi
    "$service" "${service_arguments[@]}" \
        >"$service_log" 2>&1 &
    service_pid=$!
    for _ in {1..6000}; do
        if [[ -S "$socket_path" ]]; then
            break
        fi
        sleep 0.01
    done
    if [[ ! -S "$socket_path" ]]; then
        echo "control service did not create its socket" >&2
        exit 1
    fi
fi

inference_workload_arguments=()
common_workload_arguments=()
if [[ -n "$duration_seconds" ]]; then
    common_workload_arguments+=(--duration-seconds "$duration_seconds")
fi
if [[ -n "$inference_target_ms" ]]; then
    inference_workload_arguments+=(--latency-target-ms "$inference_target_ms")
fi

clean_environment=(
    -u LD_PRELOAD
    -u GLIMMER_TRACE_KERNEL_LAUNCHES
    -u GLIMMER_TRACE_MEMORY_INFO
    -u GLIMMER_TRACE_LAUNCH_TIMINGS
    -u GLIMMER_MEMORY_LIMIT_BYTES
    -u GLIMMER_TASK_MEMORY_LIMIT_BYTES
    -u GLIMMER_QUOTA_MODE
    -u GLIMMER_QUOTA_TENANT_ID
    -u GLIMMER_SCHEDULER_MODE
    -u GLIMMER_SCHEDULER_POLICY
    -u GLIMMER_SCHEDULER_PRIORITY
    -u GLIMMER_SCHEDULER_PRIORITY_THRESHOLD
    -u GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS
    -u GLIMMER_SCHEDULER_WEIGHT
    -u GLIMMER_SCHEDULER_TENANT_ID
    -u GLIMMER_SCHEDULER_CONTROL_SOCKET
    -u GLIMMER_MAX_CONCURRENT_KERNELS
    -u GLIMMER_SCHEDULER_BATCH_SIZE
)

training_environment=("${clean_environment[@]}")
inference_environment=("${clean_environment[@]}")
if [[ "$mode" == "priority" ]]; then
    training_environment+=(
        GLIMMER_SCHEDULER_MODE=enforce
        GLIMMER_SCHEDULER_POLICY=priority
        GLIMMER_SCHEDULER_PRIORITY=10
        GLIMMER_SCHEDULER_PRIORITY_THRESHOLD="$priority_threshold"
        GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS="$priority_reserved_slots"
        GLIMMER_SCHEDULER_CONTROL_SOCKET="$socket_path"
        GLIMMER_MAX_CONCURRENT_KERNELS="$max_concurrent_kernels"
        GLIMMER_SCHEDULER_BATCH_SIZE="$training_launch_batch_size"
        LD_PRELOAD="$interceptor"
    )
    inference_environment+=(
        GLIMMER_SCHEDULER_MODE=enforce
        GLIMMER_SCHEDULER_POLICY=priority
        GLIMMER_SCHEDULER_PRIORITY=100
        GLIMMER_SCHEDULER_PRIORITY_THRESHOLD="$priority_threshold"
        GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS="$priority_reserved_slots"
        GLIMMER_SCHEDULER_CONTROL_SOCKET="$socket_path"
        GLIMMER_MAX_CONCURRENT_KERNELS="$max_concurrent_kernels"
        GLIMMER_SCHEDULER_BATCH_SIZE="$inference_launch_batch_size"
        LD_PRELOAD="$interceptor"
    )
    if ((trace_timings)); then
        training_environment+=(GLIMMER_TRACE_LAUNCH_TIMINGS=1)
        inference_environment+=(GLIMMER_TRACE_LAUNCH_TIMINGS=1)
    fi
fi

env "${training_environment[@]}" \
    "$python_executable" "$script" \
    --role training \
    "${common_workload_arguments[@]}" \
    --iterations "$iterations" \
    --warmup "$warmup" \
    --batch-size "$batch_size" \
    --hidden-size "$hidden_size" \
    --work-units "$training_work_units" \
    --ready-file "$training_ready" \
    --start-file "$start_file" \
    --status-file "$training_status_file" \
    --output "$training_csv" \
    >"$training_log" 2>&1 &
training_pid=$!

env "${inference_environment[@]}" \
    "$python_executable" "$script" \
    --role inference \
    "${common_workload_arguments[@]}" \
    --iterations "$iterations" \
    --warmup "$warmup" \
    --batch-size "$batch_size" \
    --hidden-size "$hidden_size" \
    --work-units 1 \
    --ready-file "$inference_ready" \
    --start-file "$start_file" \
    --status-file "$inference_status_file" \
    --output "$inference_csv" \
    "${inference_workload_arguments[@]}" \
    >"$inference_log" 2>&1 &
inference_pid=$!

process_is_running() {
    local pid="$1"
    local state_field=""
    if [[ ! -r "/proc/$pid/stat" ]]; then
        return 1
    fi
    read -r _ _ state_field _ < "/proc/$pid/stat" || return 1
    [[ "$state_field" != "Z" ]]
}

wait_for_file() {
    local file="$1"
    local pid="$2"
    local log="$3"
    for _ in {1..6000}; do
        if [[ -f "$file" ]]; then
            return 0
        fi
        if ! process_is_running "$pid"; then
            wait "$pid" 2>/dev/null || true
            echo "workload process exited before writing: $file" >&2
            cat "$log" >&2
            return 1
        fi
        sleep 0.01
    done
    echo "timed out waiting for: $file" >&2
    return 1
}

wait_for_file "$training_ready" "$training_pid" "$training_log"
wait_for_file "$inference_ready" "$inference_pid" "$inference_log"
if [[ -n "$duration_seconds" ]]; then
    "$python_executable" "$(dirname "$script")/measurements.py" "$output_dir" \
        --start-duration-seconds "$duration_seconds"
else
    touch "$start_file"
fi

training_status=0
inference_status=0
wait "$training_pid" || training_status=$?
wait "$inference_pid" || inference_status=$?
if ((training_status != 0 || inference_status != 0)); then
    echo "PyTorch co-location workload failed: training=$training_status inference=$inference_status" >&2
    cat "$training_log" "$inference_log" >&2
    exit 1
fi

wait_for_file "$training_status_file" "$training_pid" "$training_log"
wait_for_file "$inference_status_file" "$inference_pid" "$inference_log"

read_field() {
    local field="$1"
    local file="$2"
    awk -F= -v key="$field" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$file"
}

training_uuid="$(read_field device_uuid "$training_status_file")"
inference_uuid="$(read_field device_uuid "$inference_status_file")"
training_index="$(read_field device_index "$training_status_file")"
inference_index="$(read_field device_index "$inference_status_file")"
training_pid_recorded="$(read_field pid "$training_status_file")"
inference_pid_recorded="$(read_field pid "$inference_status_file")"
training_start="$(read_field start_ns "$training_status_file")"
inference_start="$(read_field start_ns "$inference_status_file")"
training_end="$(read_field end_ns "$training_status_file")"
inference_end="$(read_field end_ns "$inference_status_file")"
training_validation="$(read_field validation "$training_status_file")"
inference_validation="$(read_field validation "$inference_status_file")"
training_mean_ms="$(read_field mean_ms "$training_status_file")"
training_p50_ms="$(read_field p50_ms "$training_status_file")"
training_p95_ms="$(read_field p95_ms "$training_status_file")"
training_p99_ms="$(read_field p99_ms "$training_status_file")"
training_steps_per_second="$(read_field steps_per_second "$training_status_file")"
inference_mean_ms="$(read_field mean_ms "$inference_status_file")"
inference_p50_ms="$(read_field p50_ms "$inference_status_file")"
inference_p95_ms="$(read_field p95_ms "$inference_status_file")"
inference_p99_ms="$(read_field p99_ms "$inference_status_file")"
inference_steps_per_second="$(read_field steps_per_second "$inference_status_file")"
inference_target_ms_recorded="$(read_field latency_target_ms "$inference_status_file")"
inference_target_miss_count="$(read_field latency_target_miss_count "$inference_status_file")"
inference_target_miss_ratio="$(read_field latency_target_miss_ratio "$inference_status_file")"

scheduler_total_queue_wait_us=0
scheduler_max_queue_wait_us=0
scheduler_total_service_time_us=0
scheduler_max_service_time_us=0
if [[ "$mode" == "priority" ]]; then
    if ! "$client" --socket "$socket_path" metrics >"$metrics_file"; then
        echo "failed to query scheduler metrics" >&2
        cat "$metrics_file" >&2 || true
        exit 1
    fi
    "$python_executable" "$(dirname "$script")/measurements.py" "$output_dir" --check-scheduler
    scheduler_total_queue_wait_us="$(awk '$2 == "METRICS" { print $14; exit }' "$metrics_file")"
    scheduler_max_queue_wait_us="$(awk '$2 == "METRICS" { print $15; exit }' "$metrics_file")"
    scheduler_total_service_time_us="$(awk '$2 == "METRICS" { print $16; exit }' "$metrics_file")"
    scheduler_max_service_time_us="$(awk '$2 == "METRICS" { print $17; exit }' "$metrics_file")"
    if [[ ! "$scheduler_total_queue_wait_us" =~ ^[0-9]+$ ||
        ! "$scheduler_max_queue_wait_us" =~ ^[0-9]+$ ||
        ! "$scheduler_total_service_time_us" =~ ^[0-9]+$ ||
        ! "$scheduler_max_service_time_us" =~ ^[0-9]+$ ]]; then
        echo "scheduler metrics response was malformed" >&2
        cat "$metrics_file" >&2
        exit 1
    fi
fi

if [[ -z "$training_uuid" || "$training_uuid" != "$inference_uuid" || "$training_uuid" == "unknown" ]]; then
    echo "co-location verification failed: CUDA device UUIDs differ or are unavailable" >&2
    exit 1
fi
if [[ "$training_index" != "$inference_index" || "$training_pid_recorded" == "$inference_pid_recorded" ]]; then
    echo "co-location verification failed: device indexes or process IDs are invalid" >&2
    exit 1
fi

latest_start=$((training_start > inference_start ? training_start : inference_start))
earliest_end=$((training_end < inference_end ? training_end : inference_end))
overlap_ns=0
if ((earliest_end > latest_start)); then
    overlap_ns=$((earliest_end - latest_start))
fi
overlap_ms="$(awk -v nanoseconds="$overlap_ns" 'BEGIN { printf "%.3f", nanoseconds / 1000000.0 }')"
if ((overlap_ns <= 0 || training_validation != 1 || inference_validation != 1)); then
    echo "co-location verification failed: no valid overlapping execution interval" >&2
    exit 1
fi

minimum_coverage=0
if [[ -n "$duration_seconds" ]]; then
    minimum_coverage=0.95
fi
"$python_executable" "$(dirname "$script")/measurements.py" "$output_dir" \
    --minimum-coverage "$minimum_coverage"
common_values="$("$python_executable" -c '
import json, sys
data = json.load(open(sys.argv[1]))
keys = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "steps_per_second")
values = [data[role][key] for role in ("training", "inference") for key in keys]
values += [data["inference"][key] for key in ("latency_target_miss_count", "latency_target_miss_ratio")]
values += [data["overlap_ms"], data["training"]["coverage"], data["inference"]["coverage"],
           data["training"]["samples"], data["inference"]["samples"]]
print(" ".join(str(value) for value in values))
' "$output_dir/colocation.json")"
read -r training_mean_ms training_p50_ms training_p95_ms training_p99_ms training_steps_per_second \
    inference_mean_ms inference_p50_ms inference_p95_ms inference_p99_ms inference_steps_per_second \
    inference_target_miss_count inference_target_miss_ratio overlap_ms training_coverage \
    inference_coverage training_samples inference_samples <<< "$common_values"

if [[ -n "$max_inference_target_miss_ratio" ]] && ! awk \
    -v actual="$inference_target_miss_ratio" \
    -v maximum="$max_inference_target_miss_ratio" \
    'BEGIN { exit !(actual <= maximum) }'; then
    echo "inference latency target miss ratio exceeded: actual=$inference_target_miss_ratio maximum=$max_inference_target_miss_ratio" >&2
    exit 1
fi

device_name="$(read_field device_name "$training_status_file")"
echo "pytorch_priority_workload mode=$mode status=ok colocated=1 same_gpu_uuid=1 overlap_ms=$overlap_ms max_concurrent_kernels=$max_concurrent_kernels priority_reserved_slots=$priority_reserved_slots priority_threshold=$priority_threshold adaptive_slo_target_queue_us=${adaptive_slo_target_queue_us:-0} adaptive_slo_window=$adaptive_slo_window adaptive_slo_max_reserved_slots=${adaptive_slo_max_reserved_slots:-0} training_launch_batch_size=$training_launch_batch_size inference_launch_batch_size=$inference_launch_batch_size training_mean_ms=$training_mean_ms training_p50_ms=$training_p50_ms training_p95_ms=$training_p95_ms training_p99_ms=$training_p99_ms training_steps_per_second=$training_steps_per_second inference_mean_ms=$inference_mean_ms inference_p50_ms=$inference_p50_ms inference_p95_ms=$inference_p95_ms inference_p99_ms=$inference_p99_ms inference_steps_per_second=$inference_steps_per_second inference_target_ms=$inference_target_ms_recorded inference_target_miss_count=$inference_target_miss_count inference_target_miss_ratio=$inference_target_miss_ratio scheduler_total_queue_wait_us=$scheduler_total_queue_wait_us scheduler_max_queue_wait_us=$scheduler_max_queue_wait_us scheduler_total_service_time_us=$scheduler_total_service_time_us scheduler_max_service_time_us=$scheduler_max_service_time_us device_index=$training_index device_uuid=$training_uuid device_name=$device_name training_pid=$training_pid_recorded inference_pid=$inference_pid_recorded"
echo "pytorch_priority_measurement version=wall_clock_v2 duration_seconds=${duration_seconds:-0} training_coverage=$training_coverage inference_coverage=$inference_coverage training_samples=$training_samples inference_samples=$inference_samples"
echo "pytorch_priority_workload output_dir=$output_dir"
