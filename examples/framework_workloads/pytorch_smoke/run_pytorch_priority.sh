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
  --warmup N                      warmup iterations per process (default: 3)
  --batch-size N                  batch size (default: 32)
  --hidden-size N                 MLP hidden size (default: 1024)
  --training-work-units N         training work units (default: 2)
  --max-concurrent-kernels N      scheduler concurrency window (default: 1)
  --training-launch-batch-size N  training launches per scheduler lease (default: 1)
  --inference-launch-batch-size N inference launches per scheduler lease (default: 1)
  --trace-timings                 enable launch-path timing diagnostics
  --help                          show this message
EOF
}

mode="priority"
build_dir="build/cuda-gpu"
output_dir=""
python_executable=""
iterations=20
warmup=3
batch_size=32
hidden_size=1024
training_work_units=2
max_concurrent_kernels=1
training_launch_batch_size=1
inference_launch_batch_size=1
trace_timings=0

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
if [[ "$mode" == "priority" ]]; then
    for required_file in "$interceptor" "$service"; do
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
training_csv="$output_dir/training.csv"
inference_csv="$output_dir/inference.csv"
rm -f "$training_ready" "$inference_ready" "$start_file" \
    "$training_status_file" "$inference_status_file" "$training_log" \
    "$inference_log" "$service_log" "$training_csv" "$inference_csv"

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
    "$service" \
        --socket "$socket_path" \
        --quota-bytes 4294967296 \
        --execution-mode remote \
        --lease-timeout-ms 5000 \
        --max-queued-tasks 256 \
        --max-concurrent-tasks "$max_concurrent_kernels" \
        --scheduler-policy priority \
        --bind-leases-to-process \
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
        GLIMMER_SCHEDULER_CONTROL_SOCKET="$socket_path"
        GLIMMER_MAX_CONCURRENT_KERNELS="$max_concurrent_kernels"
        GLIMMER_SCHEDULER_BATCH_SIZE="$training_launch_batch_size"
        LD_PRELOAD="$interceptor"
    )
    inference_environment+=(
        GLIMMER_SCHEDULER_MODE=enforce
        GLIMMER_SCHEDULER_POLICY=priority
        GLIMMER_SCHEDULER_PRIORITY=100
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
    --iterations "$iterations" \
    --warmup "$warmup" \
    --batch-size "$batch_size" \
    --hidden-size "$hidden_size" \
    --work-units 1 \
    --ready-file "$inference_ready" \
    --start-file "$start_file" \
    --status-file "$inference_status_file" \
    --output "$inference_csv" \
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
touch "$start_file"

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

device_name="$(read_field device_name "$training_status_file")"
echo "pytorch_priority_workload mode=$mode status=ok colocated=1 same_gpu_uuid=1 overlap_ms=$overlap_ms max_concurrent_kernels=$max_concurrent_kernels training_launch_batch_size=$training_launch_batch_size inference_launch_batch_size=$inference_launch_batch_size device_index=$training_index device_uuid=$training_uuid device_name=$device_name training_pid=$training_pid_recorded inference_pid=$inference_pid_recorded"
echo "pytorch_priority_workload output_dir=$output_dir"
