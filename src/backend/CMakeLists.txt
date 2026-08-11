add_library(glimmer_backend
    simulated_backend.cc
)
add_library(Glimmer::backend ALIAS glimmer_backend)

set_target_properties(glimmer_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(glimmer_backend
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(glimmer_backend
    PUBLIC
        Glimmer::core
        Glimmer::project_options
)

glimmer_enable_clang_tidy(glimmer_backend)
