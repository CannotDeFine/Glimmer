#!/usr/bin/env bash

# Shared, read-only preflight for repository build slots.
check_build_slot() {
    local root="$1" preset="$2"
    local directory="${root}/build/${preset}"
    local cache="${directory}/CMakeCache.txt"
    local entry

    if [[ -L "${root}/build" || -L "${directory}" || -L "${cache}" ]]; then
        printf 'ERROR: %s uses a symlink; refusing to reconfigure it.\n' "${directory}" >&2
        return 1
    fi
    if [[ -e "${directory}/.glimmer-frozen" || -L "${directory}/.glimmer-frozen" ]]; then
        printf 'ERROR: %s is frozen; refusing to reconfigure it.\n' "${directory}" >&2
        return 1
    fi
    if [[ -e "${directory}" && ! -d "${directory}" ]]; then
        printf 'ERROR: %s is not a directory; refusing to reconfigure it.\n' "${directory}" >&2
        return 1
    fi
    if [[ -d "${directory}" && ! -f "${cache}" ]]; then
        if ! entry=$(find "${directory}" -mindepth 1 -maxdepth 1 -print -quit); then
            printf 'ERROR: cannot inspect %s; refusing to reconfigure it.\n' "${directory}" >&2
            return 1
        fi
        if [[ -n "${entry}" ]]; then
            printf 'ERROR: %s is nonempty without a regular CMake cache; refusing to reconfigure it.\n' \
                "${directory}" >&2
            return 1
        fi
    fi
}

check_compile_database() {
    local link="$1/compile_commands.json"
    if [[ -e "${link}" && ! -L "${link}" ]]; then
        printf 'ERROR: refusing to replace a non-symlink compile database: %s\n' "${link}" >&2
        return 1
    fi
}

check_build_budget() {
    local directory="$1/build"
    local usage allocated_bytes
    if [[ -L "${directory}" || ( -e "${directory}" && ! -d "${directory}" ) ]]; then
        printf 'ERROR: cannot safely inspect build storage: %s\n' "${directory}" >&2
        return 1
    fi
    [[ -d "${directory}" ]] || return 0
    if ! usage=$(du -s -B1 -- "${directory}"); then
        printf 'ERROR: cannot measure build storage: %s\n' "${directory}" >&2
        return 1
    fi
    read -r allocated_bytes _ <<< "${usage}"
    if [[ ! "${allocated_bytes}" =~ ^[0-9]+$ ]]; then
        printf 'ERROR: invalid build storage measurement: %s\n' "${directory}" >&2
        return 1
    fi
    # GNU du emits an unsigned decimal with no leading zeros. Eleven digits
    # means at least 10,000,000,000 bytes, without shell arithmetic overflow.
    if ((${#allocated_bytes} > 10)); then
        printf 'ERROR: build storage is %s bytes; it must remain below 10000000000 bytes.\n' \
            "${allocated_bytes}" >&2
        return 1
    fi
}
