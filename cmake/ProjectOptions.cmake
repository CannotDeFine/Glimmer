add_library(glimmer_project_options INTERFACE)
add_library(Glimmer::project_options ALIAS glimmer_project_options)

target_compile_features(glimmer_project_options INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(glimmer_project_options INTERFACE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
