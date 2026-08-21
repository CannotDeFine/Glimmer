#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_pytorch_smoke.sh [options]

Options:
  --mode native|observe|enforce  CUDA execution mode (default: native)
  --role inference|training      workload role (default: inference)
  --build-dir DIR                CUDA build directory (default: build/cuda-gpu)
  --python PATH                  Python executable (default: local .venv)
  --output FILE                  CSV output path
  --trace                        enable allocation-free launch tracing
  --trace-timings                enable launch-path timing diagnostics
  --help                         show this message

Additional arguments after '--' are passed to pytorch_smoke.py.
EOF
}

mode="native"
role="inference"
build_dir="build/cuda-gpu"
python_executable=""
output_file=""
trace=0
trace_timings=0
python_args=()

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
        --role)
            if (($# < 2)); then
                echo "--role requires a value" >&2
                exit 2
            fi
            role="$2"
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
        --python)
            if (($# < 2)); then
                echo "--python requires a value" >&2
                exit 2
            fi
            python_executable="$2"
            shift 2
            ;;
        --output)
            if (($# < 2)); then
                echo "--output requires a value" >&2
                exit 2
            fi
            output_file="$2"
            shift 2
            ;;
        --trace)
            trace=1
            shift
            ;;
        --trace-timings)
            trace_timings=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            python_args+=("$@")
            break
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$mode" != "native" && "$mode" != "observe" && "$mode" != "enforce" ]]; then
    echo "invalid mode: $mode" >&2
    usage >&2
    exit 2
fi
if [[ "$role" != "inference" && "$role" != "training" ]]; then
    echo "invalid role: $role" >&2
    usage >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "$build_dir" != /* ]]; then
    build_dir="$repo_dir/$build_dir"
fi
script="$repo_dir/examples/framework_workloads/pytorch_smoke/pytorch_smoke.py"
interceptor="$build_dir/lib/libglimmer_cuda_interceptor.so"
local_python="$repo_dir/examples/framework_workloads/pytorch_smoke/.venv/bin/python"
if [[ -z "$python_executable" && -x "$local_python" ]]; then
    python_executable="$local_python"
fi
if [[ -z "$python_executable" ]]; then
    python_executable="python3"
fi
if [[ "$mode" != "native" && ! -f "$interceptor" ]]; then
    echo "required interceptor is missing: $interceptor" >&2
    echo "build with: cmake --build --preset cuda-gpu --target glimmer_cuda_interceptor -j2" >&2
    exit 1
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

if [[ -z "$output_file" ]]; then
    output_file="$repo_dir/examples/framework_workloads/pytorch_smoke/output/${mode}-${role}.csv"
fi
mkdir -p "$(dirname "$output_file")"

common_args=(
    "$script"
    --role "$role"
    --output "$output_file"
    "${python_args[@]}"
)
if ((trace)); then
    trace_environment=(GLIMMER_TRACE_KERNEL_LAUNCHES=1)
else
    trace_environment=(-u GLIMMER_TRACE_KERNEL_LAUNCHES)
fi
if ((trace_timings)); then
    timing_environment=(GLIMMER_TRACE_LAUNCH_TIMINGS=1)
else
    timing_environment=(-u GLIMMER_TRACE_LAUNCH_TIMINGS)
fi

clean_environment=(
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
)

case "$mode" in
    native)
        env "${clean_environment[@]}" "${timing_environment[@]}" -u LD_PRELOAD \
            "$python_executable" "${common_args[@]}"
        ;;
    observe)
        env "${clean_environment[@]}" "${trace_environment[@]}" "${timing_environment[@]}" \
            GLIMMER_SCHEDULER_MODE=observe \
            LD_PRELOAD="$interceptor" "$python_executable" "${common_args[@]}"
        ;;
    enforce)
        env "${clean_environment[@]}" "${trace_environment[@]}" "${timing_environment[@]}" \
            GLIMMER_SCHEDULER_MODE=enforce \
            GLIMMER_MAX_CONCURRENT_KERNELS=2 \
            GLIMMER_SCHEDULER_POLICY=weighted_rr \
            LD_PRELOAD="$interceptor" "$python_executable" "${common_args[@]}"
        ;;
esac

echo "pytorch_smoke_runner mode=$mode role=$role status=ok output=$output_file"
