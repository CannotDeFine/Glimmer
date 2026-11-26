include_guard(GLOBAL)

# This header-only dependency never executes upstream CMake or downloads code.
function(glimmer_validate_cutlass root)
    if(NOT EXISTS "${root}/include/cutlass/gemm/device/gemm.h" OR
       NOT EXISTS "${root}/LICENSE.txt")
        message(FATAL_ERROR
            "CUTLASS is missing. Initialize the pinned 3rdparty/cutlass submodule.")
    endif()
    find_package(Git REQUIRED)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${root}" rev-parse --verify HEAD
        OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE revision_status TIMEOUT 15
    )
    if(NOT revision_status STREQUAL "0" OR
       NOT revision STREQUAL "f7b19de32c5d1f3cedfc735c2849f12b537522ee")
        message(FATAL_ERROR "CUTLASS must be pinned to the reviewed v3.5.1 revision")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${root}" status --porcelain
            --untracked-files=all -- include LICENSE.txt
        OUTPUT_VARIABLE changes RESULT_VARIABLE include_status TIMEOUT 15
    )
    if(NOT include_status STREQUAL "0" OR NOT changes STREQUAL "")
        message(FATAL_ERROR "The pinned CUTLASS headers and license must be unchanged")
    endif()
    set(GLIMMER_CUTLASS_REVISION "${revision}" PARENT_SCOPE)
    set(GIT_EXECUTABLE "${GIT_EXECUTABLE}" PARENT_SCOPE)
endfunction()

function(glimmer_add_cutlass_headers root)
    file(REAL_PATH "${root}" resolved_root)
    glimmer_validate_cutlass("${resolved_root}")
    if(TARGET Glimmer::cutlass_headers)
        get_target_property(existing_root glimmer_cutlass_headers GLIMMER_CUTLASS_ROOT)
        if(NOT existing_root STREQUAL resolved_root)
            message(FATAL_ERROR "Conflicting CUTLASS roots in the same build")
        endif()
    else()
        add_library(glimmer_cutlass_headers INTERFACE)
        add_library(Glimmer::cutlass_headers ALIAS glimmer_cutlass_headers)
        target_include_directories(glimmer_cutlass_headers SYSTEM INTERFACE
            "${resolved_root}/include")
        set_property(TARGET glimmer_cutlass_headers PROPERTY
            GLIMMER_CUTLASS_ROOT "${resolved_root}")
        add_custom_target(glimmer_verify_cutlass
            COMMAND "${CMAKE_COMMAND}"
                "-DGLIMMER_CUTLASS_ROOT=${resolved_root}"
                "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            COMMENT "Verifying pinned CUTLASS headers and license"
            VERBATIM)
        add_dependencies(glimmer_cutlass_headers glimmer_verify_cutlass)
    endif()
    set(GLIMMER_CUTLASS_REVISION "${GLIMMER_CUTLASS_REVISION}" PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
    glimmer_validate_cutlass("${GLIMMER_CUTLASS_ROOT}")
endif()
