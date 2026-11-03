add_library(glimmer_control
    execution_protocol.cc
    internal/control_request_sequence.cc
    internal/local_control_wake.cc
    internal/owned_control_exchange.cc
    internal/shared_control_channel.cc
    execution_socket.cc
    internal/process_identity.cc
    execution_dnn_legacy_protocol.cc
    execution_dnn_protocol.cc
    execution_dnn_backend_protocol.cc
    execution_nccl_protocol.cc
    execution_blas_protocol.cc
    execution_function_protocol.cc
    execution_jit_protocol.cc
    execution_pointer_protocol.cc
    execution_process_memory_protocol.cc
    execution_graph_protocol.cc
    native_epoch_protocol.cc
    task_execution_protocol.cc
    task_execution_admission.cc
    task_execution_endpoint.cc
    shared_task_control_client.cc
    internal/execution_admission_state.cc
    internal/execution_cooperative_admission.cc
    internal/execution_native_epoch_admission.cc
    composite_quota.cc
    device_capacity_quota.cc
    internal/socket_line_reader.cc
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
    PRIVATE
        spdlog::spdlog
)

glimmer_enable_clang_tidy(glimmer_control)
