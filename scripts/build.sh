#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/debug"

cd "${ROOT_DIR}"

if [[ ! -f "${ROOT_DIR}/3rdparty/spdlog/CMakeLists.txt" ]]; then
    if command -v git >/dev/null 2>&1 && [[ -e "${ROOT_DIR}/.git" ]]; then
        git -C "${ROOT_DIR}" submodule update --init --recursive
    else
        echo "Missing 3rdparty/spdlog. Clone with --recurse-submodules." >&2
        exit 1
    fi
fi

cmake --preset debug
cmake --build --preset debug

# clangd usually looks for compile_commands.json in the project root.  The CMake
# target creates a link; this copy step also refreshes a regular-file fallback
# when a tool or generator has replaced that link.
if [[ -f "${BUILD_DIR}/compile_commands.json" ]]; then
    cmake -E copy_if_different \
        "${BUILD_DIR}/compile_commands.json" \
        "${ROOT_DIR}/compile_commands.json"
fi
