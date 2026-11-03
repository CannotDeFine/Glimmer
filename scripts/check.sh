#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: ./scripts/check.sh' \
        'Run available build, test, formatting and lint checks using multiple preset directories.' \
        'Inspect existing caches and storage before running; see docs/TESTING.md.'
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
# shellcheck source=scripts/build_safety.sh
source "${ROOT_DIR}/scripts/build_safety.sh"

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

check_preset_cache() {
    local preset="$1"
    local directory="${ROOT_DIR}/build/${preset}"
    local cache="${directory}/CMakeCache.txt"
    local value

    check_build_slot "${ROOT_DIR}" "${preset}"
    [[ -f "${cache}" ]] || return 0
    value=$(sed -n 's/^GLIMMER_ENABLE_CUDA_GPU_TESTS:[^=]*=//p' "${cache}")
    case "${value^^}" in
        ''|0|OFF|NO|FALSE|N|IGNORE|NOTFOUND|*-NOTFOUND) ;;
        *)
            printf 'ERROR: GPU tests are enabled in %s; refusing to change the cache.\n' \
                "${cache}" >&2
            return 1
            ;;
    esac
}

run_preset() {
    local preset="$1"
    shift
    local -a ctest_filter=()
    if [[ "${preset}" != "debug" ]]; then
        ctest_filter=(-LE '^no-instrumented-code$')
    fi

    check_build_budget "${ROOT_DIR}"
    section "${preset}: configure"
    cmake --preset "${preset}" -DGLIMMER_ENABLE_CUDA_GPU_TESTS=OFF
    check_build_budget "${ROOT_DIR}"
    section "${preset}: build"
    cmake --build --preset "${preset}"
    check_build_budget "${ROOT_DIR}"
    # The target checks the same source tree in every preset; run it once.
    if [[ "${preset}" == "debug" ]]; then
        if command -v clang-format >/dev/null 2>&1; then
            section "${preset}: format check"
            cmake --build --preset "${preset}" --target format-check
            check_build_budget "${ROOT_DIR}"
        else
            printf '%sWARNING: clang-format not found on PATH; skipping format check.%s\n' \
                "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
        fi
    fi
    section "${preset}: tests"
    CUDA_VISIBLE_DEVICES='' ctest --preset "${preset}" --output-on-failure --no-tests=error \
        "${ctest_filter[@]}" "$@"
    check_build_budget "${ROOT_DIR}"
}

# Check all selected slots before configuring any of them. An enabled cache may
# be an intentional hardware setup; leave the choice to its owner.
check_compile_database "${ROOT_DIR}"
check_preset_cache debug
check_preset_cache asan-ubsan
check_build_budget "${ROOT_DIR}"
if command -v clang-tidy >/dev/null 2>&1; then
    check_preset_cache lint
fi
if command -v nvcc >/dev/null 2>&1 && command -v clang-tidy >/dev/null 2>&1; then
    check_preset_cache cuda-lint
fi
if command -v nvcc >/dev/null 2>&1; then
    check_preset_cache cuda-perf
fi

section "repository: whitespace check"
git diff --check
git diff --cached --check

if command -v shellcheck >/dev/null 2>&1; then
    section "repository: shellcheck"
    # Include tracked and new source scripts, but not ignored environments or
    # generated output. NUL records preserve spaces and newlines in paths;
    # pipefail also makes a failed Git inventory fail the gate.
    git ls-files --cached --others --exclude-standard -z -- \
        ':(glob)scripts/**/*.sh' ':(glob)examples/**/*.sh' |
        while IFS= read -r -d '' script; do
            if [[ -f "${script}" && ! -L "${script}" ]]; then
                shellcheck -x "${script}"
            fi
        done
else
    printf '%sWARNING: shellcheck not found on PATH; skipping script analysis.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

run_preset debug

section "asan-ubsan: configure and build"
check_build_budget "${ROOT_DIR}"
cmake --preset asan-ubsan -DGLIMMER_ENABLE_CUDA_GPU_TESTS=OFF
check_build_budget "${ROOT_DIR}"
cmake --build --preset asan-ubsan
check_build_budget "${ROOT_DIR}"
if [[ -z "${ASAN_OPTIONS:-}" ]]; then
    printf '%sWARNING: ASan leak detection is disabled by default; set ASAN_OPTIONS to change it.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi
CUDA_VISIBLE_DEVICES='' ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}" \
    ctest --preset asan-ubsan --output-on-failure --no-tests=error -LE '^no-instrumented-code$'
check_build_budget "${ROOT_DIR}"

if command -v clang-tidy >/dev/null 2>&1; then
    run_preset lint
else
    printf '%sWARNING: clang-tidy not found on PATH; skipping lint preset.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

if command -v nvcc >/dev/null 2>&1 && command -v clang-tidy >/dev/null 2>&1; then
    run_preset cuda-lint
else
    printf '%sWARNING: nvcc or clang-tidy not found on PATH; skipping cuda-lint preset.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

if command -v nvcc >/dev/null 2>&1; then
    # The CMake option is the primary GPU-test gate; some names lack this suffix.
    # Keep optimized no-GPU regressions in the normal gate without requiring a GPU.
    run_preset cuda-perf -E '_gpu_test$'
else
    printf '%sWARNING: nvcc not found on PATH; skipping optimized CUDA checks.%s\n' \
        "${COLOR_YELLOW}" "${COLOR_RESET}" >&2
fi

printf '\n%s%sInvoked checks exited successfully; review skipped checks and warnings above.%s\n' \
    "${COLOR_BOLD}" "${COLOR_GREEN}" "${COLOR_RESET}"
