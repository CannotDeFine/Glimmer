#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    COLOR_BOLD=$'\033[1m'
    COLOR_CYAN=$'\033[1;36m'
    COLOR_GREEN=$'\033[1;32m'
    COLOR_YELLOW=$'\033[1;33m'
    COLOR_RESET=$'\033[0m'
else
    COLOR_BOLD=''
    COLOR_CYAN=''
    COLOR_GREEN=''
    COLOR_YELLOW=''
    COLOR_RESET=''
fi

section() {
    local title="$1"
    local title_length=${#title}
    local index
    local content_width=50
    local left_padding=$(( (content_width - title_length - 2) / 2 ))
    local right_padding=$(( content_width - title_length - 2 - left_padding ))

    printf '\n%s%s╭' "${COLOR_BOLD}" "${COLOR_CYAN}"
    for ((index = 0; index < content_width; index++)); do
        printf '─'
    done
    printf '╮%s\n' "${COLOR_RESET}"

    printf '%s%s│' "${COLOR_BOLD}" "${COLOR_CYAN}"
    for ((index = 0; index < left_padding; index++)); do
        printf ' '
    done
    printf ' %s ' "${title}"
    for ((index = 0; index < right_padding; index++)); do
        printf ' '
    done
    printf '│%s\n' "${COLOR_RESET}"

    printf '%s%s╰' "${COLOR_BOLD}" "${COLOR_CYAN}"
    for ((index = 0; index < content_width; index++)); do
        printf '─'
    done
    printf '╯%s\n' "${COLOR_RESET}"
}

run_preset() {
    local preset="$1"

    section "${preset}: configure"
    cmake --preset "${preset}"
    section "${preset}: build"
    cmake --build --preset "${preset}"
    if command -v clang-format >/dev/null 2>&1; then
        section "${preset}: format check"
        cmake --build --preset "${preset}" --target format-check
    else
        printf '%sWARNING: clang-format is not installed; skipping %s format check.%s\n' \
            "${COLOR_YELLOW}" "${preset}" "${COLOR_RESET}" >&2
    fi
    section "${preset}: tests"
    ctest --preset "${preset}" --output-on-failure
}

section "repository: whitespace check"
git diff --check

if command -v shellcheck >/dev/null 2>&1; then
    section "repository: shellcheck"
    shellcheck scripts/*.sh
else
    printf '%sWARNING: shellcheck is not installed; skipping script analysis.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

run_preset debug

section "asan-ubsan: configure and build"
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" \
    ctest --preset asan-ubsan --output-on-failure

if command -v clang-tidy >/dev/null 2>&1; then
    run_preset lint
else
    printf '%sWARNING: clang-tidy is not installed; skipping lint preset.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

if command -v nvcc >/dev/null 2>&1 && command -v clang-tidy >/dev/null 2>&1; then
    run_preset cuda-lint
else
    printf '%sWARNING: CUDA Toolkit or clang-tidy is unavailable; skipping cuda-lint preset.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

printf '\n%s%sAll available checks passed.%s\n' \
    "${COLOR_BOLD}" "${COLOR_GREEN}" "${COLOR_RESET}"
