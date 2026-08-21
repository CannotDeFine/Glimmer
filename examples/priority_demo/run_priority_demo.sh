#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_priority_demo.sh [--mode priority|native] [--build-dir DIR]
                            [--output-dir DIR] [--no-plot] [--trace]

Runs a mixed training/inference workload. The default priority mode routes
launches through the transparent CUDA scheduler; native mode leaves CUDA
untouched. The default output directory is
examples/priority_demo/output/<mode> and contains per-process CSV traces, logs,
and an SVG plot.
EOF
}

mode="priority"
build_dir="build/cuda-gpu"
output_dir=""
make_plot=1
trace_kernel_launches=0

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
        --no-plot)
            make_plot=0
            shift
            ;;
        --trace)
            trace_kernel_launches=1
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

if [[ "$mode" != "priority" && "$mode" != "native" ]]; then
    echo "invalid mode: $mode" >&2
    usage >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$build_dir" != /* ]]; then
    build_dir="$repo_dir/$build_dir"
fi

workload="$build_dir/examples/priority_demo/glimmer_cuda_priority_workload"
service="$build_dir/bin/glimmer_control_service"
interceptor="$build_dir/lib/libglimmer_cuda_interceptor.so"
required_files=("$workload")
if [[ "$mode" == "priority" ]]; then
    required_files+=("$service" "$interceptor")
fi
for required_file in "${required_files[@]}"; do
    if [[ ! -x "$required_file" && ! -f "$required_file" ]]; then
        echo "required build artifact is missing: $required_file" >&2
        echo "build with: cmake --build --preset cuda-gpu --target glimmer_cuda_priority_demo -j2" >&2
        exit 1
    fi
done

if [[ -z "$output_dir" ]]; then
    output_dir="$repo_dir/examples/priority_demo/output/$mode"
    mkdir -p "$output_dir"
else
    mkdir -p "$output_dir"
fi

socket_dir=""
socket_path=""
if [[ "$mode" == "priority" ]]; then
    socket_dir="$(mktemp -d "${TMPDIR:-/tmp}/glimmer-priority-socket.XXXXXX")"
    socket_path="$socket_dir/control.sock"
fi
start_file="$output_dir/start.signal"
service_pid=""
training_pid=""
inference_pid=""

rm -f \
    "$output_dir/control-service.log" \
    "$output_dir/training.csv" \
    "$output_dir/training.log" \
    "$output_dir/training.ready" \
    "$output_dir/inference.csv" \
    "$output_dir/inference.log" \
    "$output_dir/inference.ready" \
    "$start_file" \
    "$output_dir/priority_trace.svg"

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
        rmdir "$socket_dir" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$mode" == "priority" ]]; then
    "$service" \
        --socket "$socket_path" \
        --quota-bytes 8388608 \
        --execution-mode remote \
        --max-concurrent-tasks 2 \
        --max-queued-tasks 128 \
        --scheduler-policy priority \
        >"$output_dir/control-service.log" 2>&1 &
    service_pid=$!

    for _ in {1..100}; do
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

common_environment=()
if [[ "$mode" == "priority" ]]; then
    common_environment=(
        GLIMMER_MAX_CONCURRENT_KERNELS=2
        GLIMMER_SCHEDULER_MODE=enforce
        GLIMMER_SCHEDULER_POLICY=priority
        GLIMMER_SCHEDULER_CONTROL_SOCKET="$socket_path"
        LD_PRELOAD="$interceptor"
    )
    if ((trace_kernel_launches)); then
        common_environment+=(GLIMMER_TRACE_KERNEL_LAUNCHES=1)
    fi
fi
training_environment=("${common_environment[@]}")
inference_environment=("${common_environment[@]}")
if [[ "$mode" == "priority" ]]; then
    training_environment+=(GLIMMER_SCHEDULER_PRIORITY=10)
    inference_environment+=(GLIMMER_SCHEDULER_PRIORITY=100)
fi

env "${training_environment[@]}" \
    "$workload" \
    --role training \
    --kernel model \
    --matrix-size 2048 \
    --iterations 40 \
    --work-units 2 \
    --interval-us 3000 \
    --ready-file "$output_dir/training.ready" \
    --start-file "$start_file" \
    --output "$output_dir/training.csv" \
    >"$output_dir/training.log" 2>&1 &
training_pid=$!

env "${inference_environment[@]}" \
    "$workload" \
    --role inference \
    --kernel model \
    --matrix-size 1024 \
    --iterations 10 \
    --work-units 1 \
    --interval-us 500 \
    --ready-file "$output_dir/inference.ready" \
    --start-file "$start_file" \
    --output "$output_dir/inference.csv" \
    >"$output_dir/inference.log" 2>&1 &
inference_pid=$!

for _ in {1..2000}; do
    if [[ -f "$output_dir/training.ready" && -f "$output_dir/inference.ready" ]]; then
        break
    fi
    sleep 0.001
done
if [[ ! -f "$output_dir/training.ready" || ! -f "$output_dir/inference.ready" ]]; then
    echo "priority workloads did not finish CUDA initialization" >&2
    exit 1
fi
touch "$start_file"

training_status=0
inference_status=0
wait "$training_pid" || training_status=$?
wait "$inference_pid" || inference_status=$?
if ((training_status != 0 || inference_status != 0)); then
    echo "priority demo failed: training=$training_status inference=$inference_status" >&2
    exit 1
fi

if ((make_plot)); then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required to generate the SVG trace" >&2
        exit 1
    fi
    python3 "$repo_dir/examples/priority_demo/plot_priority_trace.py" \
        --input "$output_dir/training.csv" "$output_dir/inference.csv" \
        --output "$output_dir/priority_trace.svg"
fi

echo "priority_demo mode=$mode status=ok output_dir=$output_dir"
