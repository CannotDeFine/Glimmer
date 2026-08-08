#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

run_preset() {
    local preset="$1"

    echo "==> Configuring ${preset}"
    cmake --preset "${preset}"
    echo "==> Building ${preset}"
    cmake --build --preset "${preset}"
    echo "==> Checking formatting (${preset})"
    cmake --build --preset "${preset}" --target format-check
    echo "==> Running tests (${preset})"
    ctest --preset "${preset}" --output-on-failure
}

echo "==> Checking repository whitespace"
git diff --check

run_preset debug

echo "==> Running sanitizer checks"
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" \
    ctest --preset asan-ubsan --output-on-failure

if command -v clang-tidy >/dev/null 2>&1; then
    run_preset lint
else
    echo "WARNING: clang-tidy is not installed; skipping lint preset." >&2
fi

if command -v nvcc >/dev/null 2>&1 && [[ -f /usr/local/cuda/include/cuda.h ]]; then
    run_preset cuda-lint
else
    echo "WARNING: CUDA Toolkit is unavailable; skipping cuda-lint preset." >&2
fi

echo "All available checks passed."
