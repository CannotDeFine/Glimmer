add_library(glimmer_core
    adaptive_slo.cc
    quota_ledger.cc
    scheduler.cc
    scheduler_mode.cc
)
add_library(Glimmer::core ALIAS glimmer_core)

set_target_properties(glimmer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(glimmer_core
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(glimmer_core
    PUBLIC
        Glimmer::project_options
)

glimmer_enable_clang_tidy(glimmer_core)
