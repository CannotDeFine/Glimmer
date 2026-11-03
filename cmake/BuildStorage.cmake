include_guard(GLOBAL)

# This is a configure-time workflow guard, not a filesystem access boundary.
function(glimmer_check_build_storage)
    get_filename_component(project_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." REALPATH)
    file(REAL_PATH "${CMAKE_SOURCE_DIR}" source_dir)
    file(REAL_PATH "${CMAKE_BINARY_DIR}" binary_dir)
    if(NOT "${CMAKE_BINARY_DIR}" STREQUAL "${binary_dir}")
        message(FATAL_ERROR "Build-directory symlink aliases are not supported; use a canonical path.")
    endif()
    cmake_path(IS_PREFIX binary_dir "${project_root}" NORMALIZE contains_project)
    if(contains_project OR "${binary_dir}" STREQUAL "${source_dir}")
        message(FATAL_ERROR "In-source and ancestor builds are not supported.")
    endif()
    if(EXISTS "${binary_dir}/.glimmer-frozen" OR IS_SYMLINK "${binary_dir}/.glimmer-frozen")
        message(FATAL_ERROR "This build is frozen evidence and must not be reconfigured.")
    endif()

    cmake_path(IS_PREFIX project_root "${binary_dir}" NORMALIZE inside_project)
    if(inside_project)
        file(READ "${project_root}/CMakePresets.json" presets)
        string(JSON count LENGTH "${presets}" configurePresets)
        math(EXPR last "${count} - 1")
        set(allowed "${project_root}/build/candidate")
        foreach(index RANGE ${last})
            string(JSON hidden ERROR_VARIABLE hidden_error GET "${presets}" configurePresets ${index} hidden)
            if(NOT hidden_error AND hidden)
                continue()
            endif()
            string(JSON name GET "${presets}" configurePresets ${index} name)
            list(APPEND allowed "${project_root}/build/${name}")
        endforeach()
        if(NOT binary_dir IN_LIST allowed)
            message(FATAL_ERROR
                "Repository builds must use build/<configure-preset> or build/candidate. "
                "Numbered and legacy build directories are read-only evidence. "
                "See docs/STORAGE_LAYOUT.md; do not create another full build for a run.")
        endif()
    endif()

    # Never switch the reusable slot between the root and an overlay silently.
    if(EXISTS "${binary_dir}/CMakeCache.txt")
        file(STRINGS "${binary_dir}/CMakeCache.txt" home REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
        if(home)
            string(REGEX REPLACE "^CMAKE_HOME_DIRECTORY:INTERNAL=" "" home "${home}")
            file(REAL_PATH "${home}" cached_source)
            if(NOT "${cached_source}" STREQUAL "${source_dir}")
                message(FATAL_ERROR
                    "Build slot belongs to another source tree. Preserve its evidence and "
                    "clean verified disposable outputs before reusing it.")
            endif()
        endif()
    endif()
endfunction()
