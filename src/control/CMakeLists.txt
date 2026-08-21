add_library(glimmer_control
    composite_quota.cc
    device_capacity_quota.cc
    launch_gate.cc
    process_memory_quota.cc
    shared_memory_quota.cc
    task_admission.cc
    task_endpoint.cc
    task_protocol.cc
    unix_socket_client.cc
    unix_socket_server.cc
)
add_library(Glimmer::control ALIAS glimmer_control)

set_target_properties(glimmer_control PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(glimmer_control
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(glimmer_control
    PUBLIC
        Glimmer::core
        Threads::Threads
)

glimmer_enable_clang_tidy(glimmer_control)
