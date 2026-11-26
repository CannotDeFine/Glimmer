cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/../BuildStorage.cmake")
glimmer_check_build_storage()
project(GlimmerServiceOptimizationDiagnostic LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../ExecutionServiceDiagnosticOverlay/CMakeLists.txt")

# Keep the observer and all imported baseline libraries unchanged. This is a
# controlled compiler intervention, not a production build configuration.
foreach(source runtime.cc host_owner.cc profile_archive.cc async_audit_log.cc
               nccl_actor.cc nccl_output_effects.cc main.cc)
    set_source_files_properties("${PROJECT_SOURCE_DIR}/src/execution_service/${source}"
        TARGET_DIRECTORY glimmer_execution_service_runtime
        PROPERTIES COMPILE_OPTIONS "-O2")
endforeach()
