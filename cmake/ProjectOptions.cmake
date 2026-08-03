add_library(glimmer_project_options INTERFACE)
add_library(Glimmer::project_options ALIAS glimmer_project_options)

target_compile_features(glimmer_project_options INTERFACE cxx_std_20)

option(GLIMMER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(GLIMMER_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(GLIMMER_ENABLE_CLANG_TIDY "Enable clang-tidy during compilation" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(glimmer_project_options INTERFACE
        -Wall
        -Wextra
        -Wpedantic
    )

    if(GLIMMER_WARNINGS_AS_ERRORS)
        target_compile_options(glimmer_project_options INTERFACE -Werror)
    endif()

    if(GLIMMER_ENABLE_SANITIZERS)
        target_compile_options(glimmer_project_options INTERFACE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(glimmer_project_options INTERFACE
            -fsanitize=address,undefined
        )
    endif()
endif()

if(GLIMMER_ENABLE_CLANG_TIDY)
    find_program(GLIMMER_CLANG_TIDY NAMES clang-tidy REQUIRED)
endif()

function(glimmer_enable_clang_tidy target)
    if(GLIMMER_ENABLE_CLANG_TIDY)
        set_property(TARGET ${target} PROPERTY
            CXX_CLANG_TIDY
            "${GLIMMER_CLANG_TIDY};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
        )
    endif()
endfunction()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
