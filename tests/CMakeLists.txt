add_executable(glimmer_core_quota_ledger_test core/quota_ledger_test.cc)

target_link_libraries(glimmer_core_quota_ledger_test
    PRIVATE
        Glimmer::core
)

glimmer_enable_clang_tidy(glimmer_core_quota_ledger_test)

add_test(NAME glimmer_core_quota_ledger_test COMMAND glimmer_core_quota_ledger_test)

add_executable(glimmer_core_scheduler_test core/scheduler_test.cc)

target_link_libraries(glimmer_core_scheduler_test
    PRIVATE
        Glimmer::core
)

glimmer_enable_clang_tidy(glimmer_core_scheduler_test)

add_test(NAME glimmer_core_scheduler_test COMMAND glimmer_core_scheduler_test)

add_executable(glimmer_backend_simulated_test backend/simulated_backend_test.cc)

target_link_libraries(glimmer_backend_simulated_test
    PRIVATE
        Glimmer::backend
)

glimmer_enable_clang_tidy(glimmer_backend_simulated_test)

add_test(NAME glimmer_backend_simulated_test COMMAND glimmer_backend_simulated_test)

add_executable(glimmer_control_process_memory_quota_test control/process_memory_quota_test.cc)

target_link_libraries(glimmer_control_process_memory_quota_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_process_memory_quota_test)

add_test(NAME glimmer_control_process_memory_quota_test
    COMMAND glimmer_control_process_memory_quota_test)

add_executable(glimmer_control_shared_memory_quota_test
    control/shared_memory_quota_test.cc
)

target_link_libraries(glimmer_control_shared_memory_quota_test
    PRIVATE
        Glimmer::control
)

glimmer_enable_clang_tidy(glimmer_control_shared_memory_quota_test)

add_test(NAME glimmer_control_shared_memory_quota_test
    COMMAND glimmer_control_shared_memory_quota_test)
