# An explicit installed SDK only. No configure-time download or vendor copying.
function(glimmer_add_cudnn_sdk root)
    set(GLIMMER_CUDNN_ABI_PROFILE "9" CACHE STRING "Owned cuDNN ABI major (8 or 9)")
    set_property(CACHE GLIMMER_CUDNN_ABI_PROFILE PROPERTY STRINGS 8 9)
    if(NOT GLIMMER_CUDNN_ABI_PROFILE MATCHES "^[89]$")
        message(FATAL_ERROR "GLIMMER_CUDNN_ABI_PROFILE must select exactly 8 or 9")
    endif()
    if(NOT IS_ABSOLUTE "${root}" OR NOT EXISTS "${root}/include/cudnn.h")
        message(FATAL_ERROR "GLIMMER_CUDNN_ROOT must name an absolute installed cuDNN SDK root")
    endif()
    file(REAL_PATH "${root}" _glimmer_cudnn_root)
    if(TARGET Glimmer::cudnn_headers)
        get_target_property(_glimmer_cudnn_selected glimmer_cudnn_headers GLIMMER_SDK_ROOT)
        if(NOT _glimmer_cudnn_selected STREQUAL _glimmer_cudnn_root)
            message(FATAL_ERROR "Only one cuDNN SDK root may be selected in a build")
        endif()
        return()
    endif()
    if(NOT EXISTS "${root}/include/cudnn_version.h")
        message(FATAL_ERROR "The installed cuDNN SDK is missing cudnn_version.h")
    endif()
    foreach(_glimmer_cudnn_part MAJOR MINOR PATCHLEVEL)
        file(STRINGS "${root}/include/cudnn_version.h" _glimmer_cudnn_line
            REGEX "^#define[ \t]+CUDNN_${_glimmer_cudnn_part}[ \t]+[0-9]+")
        if(NOT _glimmer_cudnn_line MATCHES
                "^#define[ \t]+CUDNN_${_glimmer_cudnn_part}[ \t]+([0-9]+)([ \t]|$)")
            message(FATAL_ERROR "The installed cuDNN SDK has an invalid version header")
        endif()
        set(_glimmer_cudnn_${_glimmer_cudnn_part} "${CMAKE_MATCH_1}")
    endforeach()
    set(_glimmer_cudnn_version
        "${_glimmer_cudnn_MAJOR}.${_glimmer_cudnn_MINOR}.${_glimmer_cudnn_PATCHLEVEL}")
    if(NOT _glimmer_cudnn_MAJOR EQUAL GLIMMER_CUDNN_ABI_PROFILE OR
            (GLIMMER_CUDNN_ABI_PROFILE STREQUAL "8" AND
             NOT _glimmer_cudnn_version VERSION_EQUAL "8.9.7") OR
            (GLIMMER_CUDNN_ABI_PROFILE STREQUAL "9" AND
             _glimmer_cudnn_version VERSION_LESS "9.10.2"))
        message(FATAL_ERROR "The selected owned cuDNN profile requires 8.9.7 or >=9.10.2 major9 headers")
    endif()
    # Do not retain discovery from a previously selected SDK root.
    set(_glimmer_cudnn_library "_glimmer_cudnn_library-NOTFOUND")
    unset(_glimmer_cudnn_library CACHE)
    find_library(_glimmer_cudnn_library NAMES cudnn "libcudnn.so.${GLIMMER_CUDNN_ABI_PROFILE}"
        PATHS "${root}/lib" "${root}/lib64" NO_DEFAULT_PATH)
    if(NOT _glimmer_cudnn_library)
        message(FATAL_ERROR "No installed cuDNN library under GLIMMER_CUDNN_ROOT")
    endif()
    add_library(glimmer_cudnn_headers INTERFACE)
    add_library(Glimmer::cudnn_headers ALIAS glimmer_cudnn_headers)
    set_target_properties(glimmer_cudnn_headers PROPERTIES GLIMMER_SDK_ROOT "${_glimmer_cudnn_root}")
    target_include_directories(glimmer_cudnn_headers SYSTEM INTERFACE "${root}/include")
    if(NOT CMAKE_READELF)
        find_program(CMAKE_READELF NAMES readelf REQUIRED)
    endif()
    file(GLOB _glimmer_cudnn_providers LIST_DIRECTORIES FALSE
        "${root}/lib/libcudnn*.so" "${root}/lib/libcudnn*.so.*"
        "${root}/lib64/libcudnn*.so" "${root}/lib64/libcudnn*.so.*")
    list(SORT _glimmer_cudnn_providers)
    if(NOT _glimmer_cudnn_providers)
        message(FATAL_ERROR "The cuDNN SDK has no explicit provider ELF identities")
    endif()
    foreach(_glimmer_provider IN LISTS _glimmer_cudnn_providers)
        execute_process(COMMAND "${CMAKE_READELF}" -d "${_glimmer_provider}"
            OUTPUT_VARIABLE _glimmer_provider_dynamic RESULT_VARIABLE _glimmer_provider_status
            ERROR_QUIET TIMEOUT 15)
        if(NOT _glimmer_provider_status EQUAL 0 OR NOT _glimmer_provider_dynamic MATCHES
                "\\(SONAME\\)[^\n]*\\[libcudnn[^]]*\\.so\\.${GLIMMER_CUDNN_ABI_PROFILE}\\]")
            message(FATAL_ERROR "cuDNN provider SONAME does not match selected ABI: ${_glimmer_provider}")
        endif()
    endforeach()
    include("${PROJECT_SOURCE_DIR}/cmake/CudnnProfileProviders.cmake")
    set(_glimmer_cudnn_identity_dir "${PROJECT_BINARY_DIR}/generated/cudnn-production")
    glimmer_write_cudnn_profile_providers(
        "${_glimmer_cudnn_identity_dir}/cudnn_profile_providers.h"
        "${CMAKE_READELF}" ${_glimmer_cudnn_providers})
    target_include_directories(glimmer_cudnn_headers INTERFACE "${_glimmer_cudnn_identity_dir}")
    target_compile_definitions(glimmer_cudnn_headers INTERFACE GLIMMER_HAS_CUDNN=1)
    add_library(Glimmer::cudnn SHARED IMPORTED GLOBAL)
    set_target_properties(Glimmer::cudnn PROPERTIES
        IMPORTED_LOCATION "${_glimmer_cudnn_library}"
        INTERFACE_LINK_LIBRARIES Glimmer::cudnn_headers)
    if(GLIMMER_CUDNN_ABI_PROFILE STREQUAL "8")
        # Legacy entry points live in distinct original SDK8 components. Only
        # composition/SDK inputs link them; the preload/backend use headers.
        foreach(_glimmer_component ops_infer ops_train cnn_infer cnn_train adv_infer adv_train)
            unset(_glimmer_split_library CACHE)
            set(_glimmer_split_library "_glimmer_split_library-NOTFOUND")
            find_library(_glimmer_split_library
                NAMES "cudnn_${_glimmer_component}" "libcudnn_${_glimmer_component}.so.8"
                PATHS "${root}/lib" "${root}/lib64" NO_DEFAULT_PATH)
            if(NOT _glimmer_split_library)
                message(FATAL_ERROR "SDK8 legacy component is missing: ${_glimmer_component}")
            endif()
            target_link_libraries(Glimmer::cudnn INTERFACE "${_glimmer_split_library}")
        endforeach()
        unset(_glimmer_split_library CACHE)
    endif()
    unset(_glimmer_cudnn_library CACHE)
endfunction()
