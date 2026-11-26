#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: setup_pytorch_env.sh [--python PATH] [--cuda cu126|cu132]

Creates the ignored .venv directory next to this script and installs the
versioned PyTorch requirements for the selected CUDA profile (default: cu132).
Use cu126 for Volta GPUs or hosts with a CUDA 12.6-compatible driver.
Requires Python 3.10 or newer; pins pip 26.2.1 inside the virtual environment.
EOF
}

python_bootstrap="python3"
cuda_profile="cu132"
while (($# > 0)); do
    case "$1" in
        --python)
            if (($# < 2)); then
                echo "--python requires a value" >&2
                exit 2
            fi
            python_bootstrap="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        --cuda)
            if (($# < 2)); then
                echo "--cuda requires a value" >&2
                exit 2
            fi
            cuda_profile="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$cuda_profile" != "cu126" && "$cuda_profile" != "cu132" ]]; then
    echo "invalid CUDA profile: $cuda_profile" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$script_dir/.venv"
requirements_file="$script_dir/requirements-$cuda_profile.txt"

if [[ "$python_bootstrap" == */* ]]; then
    if [[ ! -x "$python_bootstrap" ]]; then
        echo "python executable is not available: $python_bootstrap" >&2
        exit 1
    fi
elif ! command -v "$python_bootstrap" >/dev/null 2>&1; then
    echo "python executable is not available: $python_bootstrap" >&2
    exit 1
fi

env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
    "$python_bootstrap" -m venv "$venv_dir"
pip_command=(
    env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u PIP_TARGET -u PIP_PREFIX
    -u PIP_ROOT -u PIP_LOG -u PYTHONPATH -u PYTHONHOME
    PIP_CONFIG_FILE=/dev/null PIP_USER=0 PYTHONNOUSERSITE=1
    "$venv_dir/bin/python" -m pip --cache-dir "$venv_dir/.cache/pip"
    --timeout 120 --retries 5 --disable-pip-version-check
)
"${pip_command[@]}" install --index-url https://pypi.org/simple pip==26.2.1
"${pip_command[@]}" install --requirement "$requirements_file"

echo "pytorch_env status=ok cuda=$cuda_profile python=$venv_dir/bin/python"
