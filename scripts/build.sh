#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [[ ! -f "${ROOT_DIR}/3rdparty/spdlog/CMakeLists.txt" ]]; then
    if command -v git >/dev/null 2>&1 && [[ -e "${ROOT_DIR}/.git" ]]; then
        git -C "${ROOT_DIR}" submodule update --init --recursive
    else
        echo "Missing 3rdparty/spdlog. Clone with --recurse-submodules." >&2
        exit 1
    fi
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}"

# clangd usually looks for compile_commands.json in the project root.  Copying
# instead of creating a symbolic link keeps this script usable on more systems.
if [[ -f "${BUILD_DIR}/compile_commands.json" ]]; then
    cmake -E copy_if_different \
        "${BUILD_DIR}/compile_commands.json" \
        "${ROOT_DIR}/compile_commands.json"
fi
