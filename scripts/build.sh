#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' 'Usage: ./scripts/build.sh' \
        'Configure and build the debug preset; inspect the cache and storage first.'
}

if (( $# )); then
    if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi
    printf '%s\n' 'ERROR: unexpected arguments.' >&2
    usage >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/3rdparty/spdlog/CMakeLists.txt" ]]; then
    printf '%s\n' 'Missing 3rdparty/spdlog. Run: git submodule update --init --recursive -- 3rdparty/spdlog' >&2
    exit 1
fi

# shellcheck source=scripts/build_safety.sh
source "${ROOT_DIR}/scripts/build_safety.sh"
check_build_slot "${ROOT_DIR}" debug
check_compile_database "${ROOT_DIR}"
check_build_budget "${ROOT_DIR}"

cmake --preset debug
check_build_budget "${ROOT_DIR}"
cmake --build --preset debug
check_build_budget "${ROOT_DIR}"
