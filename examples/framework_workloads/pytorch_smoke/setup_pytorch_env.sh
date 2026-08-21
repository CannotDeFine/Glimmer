#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: setup_pytorch_env.sh [--python PATH]

Creates the ignored .venv directory next to this script and installs the
versioned CUDA 13.2 PyTorch requirements from requirements-cu132.txt.
EOF
}

python_bootstrap="python3"
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
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$script_dir/.venv"
requirements_file="$script_dir/requirements-cu132.txt"

if [[ "$python_bootstrap" == */* ]]; then
    if [[ ! -x "$python_bootstrap" ]]; then
        echo "python executable is not available: $python_bootstrap" >&2
        exit 1
    fi
elif ! command -v "$python_bootstrap" >/dev/null 2>&1; then
    echo "python executable is not available: $python_bootstrap" >&2
    exit 1
fi

"$python_bootstrap" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install --requirement "$requirements_file"

echo "pytorch_env status=ok python=$venv_dir/bin/python"
