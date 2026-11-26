#pragma once

#include <cuda.h>
#include <nvml.h>

#include "internal/diagnostics.h"
#include "internal/process_call_gate.h"
#include "glimmer/core/scheduler.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <functional>

namespace glimmer::control {
struct LaunchGateTiming;
}

namespace glimmer::interceptor {

struct CudaCooperativeReleaseResult;
class CudaNativeGraphExecutable;
class DriverCooperativeRouter;
struct PersistentSgemmProblem;
struct PersistentBlasContext {
    CUcontext context{};
    unsigned long long id{};
    bool operator==(const PersistentBlasContext&) const = default;
};
// Read-only observation; CUDA's context ID prevents address-reuse qualification.
// Requires a normal shared Call and creates no context or frontend.
[[nodiscard]] std::optional<PersistentBlasContext> persistent_blas_current_context() noexcept;
// Nullopt means no application work was accepted; preserve the original cuBLAS
// call. A returned status belongs to an accepted path and must never be replayed.
// Caller owns a normal shared Call and has qualified the cuBLAS handle/scalars.
[[nodiscard]] std::optional<CUresult> try_intercept_persistent_sgemm(
    CUcontext handle_context, PersistentSgemmProblem problem, CUdeviceptr a, CUdeviceptr b,
    CUdeviceptr c, CUdeviceptr bias = 0, CUdeviceptr workspace = 0,
    std::size_t workspace_bytes = 0);
struct CudaNativeEpochConfiguration;
enum class CudaSourceCaptureKind;
[[nodiscard]] CUresult intercept_driver_capture(CUstream stream, bool is_ptds,
                                                CudaSourceCaptureKind kind,
                                                const std::function<CUresult()>& original);
// Borrowed process owner for trusted source assembly; no frontend initialization.
// Requires an admitted shared call; the borrow must stay inside its lifetime.
// Explicit process reset/fork teardown ends the process owner's lifetime.
[[nodiscard]] DriverCooperativeRouter* cooperative_source_router() noexcept;
// Trusted private startup only, before process-owner initialization. Requires
// a normal shared Call and owns its Transition. One-shot; no SDK/actor creation.
// Ordinary applications do not supply this policy; default construction is off.
[[nodiscard]] CUresult configure_cooperative_native_epochs(
    const CudaNativeEpochConfiguration& configuration) noexcept;
// CPU-only private diagnostic under an admitted Call. Counts exact submitted
// release generations in one context; never initializes an owner or calls SDK.
[[nodiscard]] std::optional<std::size_t> cooperative_submitted_release_count(
    CUcontext context) noexcept;
[[nodiscard]] CUresult quarantine_cooperative_source_call(CUresult error) noexcept;
[[nodiscard]] CUresult cooperative_graph_executable(CUgraphExec executable,
                                                    CudaNativeGraphExecutable& output) noexcept;
// Shared CPU-only admission prerequisite for covered public process transitions.
// Thread-bound, lexical lifetime; trusted private/native callbacks bypass entry.
[[nodiscard]] ProcessCallGate& cooperative_process_call_gate() noexcept;
class CooperativeCallScope final {
   public:
    CooperativeCallScope();
    CooperativeCallScope(const CooperativeCallScope&) = delete;
    CooperativeCallScope& operator=(const CooperativeCallScope&) = delete;

   private:
    std::optional<ProcessCallGate::Call> call_;
};
[[nodiscard]] CUresult intercept_cooperative_native_boundary(
    const std::function<CUresult()>& operation);
[[nodiscard]] CUresult intercept_cooperative_memory_boundary(
    const std::function<CUresult()>& operation);
[[nodiscard]] CUresult intercept_cooperative_module_boundary(
    const std::function<CUresult()>& operation);
// Stream factories are metadata while the process retains source ownership.
[[nodiscard]] CUresult intercept_driver_stream_create(const std::function<CUresult()>& original);
[[nodiscard]] CUresult intercept_graph_instantiation(
    CUgraphExec* output, CUgraph graph, bool should_prepare_effects,
    const std::function<CUresult(CUgraphExec*)>& original);
[[nodiscard]] CUresult intercept_graph_mutation(CUgraphExec executable, bool is_destroy,
                                                const std::function<CUresult()>& original);
[[nodiscard]] CUresult intercept_graph_event_update(CUgraphExec executable, CUgraphNode node,
                                                    CUevent event, bool is_record,
                                                    const std::function<CUresult()>& original);
[[nodiscard]] CUresult intercept_graph_node_update(CUgraphExec executable, CUgraphNode node,
                                                   std::optional<CUgraphNodeType> expected_type,
                                                   const std::function<CUresult()>& original);
[[nodiscard]] CUresult intercept_graph_update(CUgraphExec executable, CUgraph graph,
                                              std::optional<CUgraphNode> child_node,
                                              const std::function<CUresult()>& original);
[[nodiscard]] void* resolve_native_driver_function(const char* name,
                                                   bool per_thread_default_stream = false,
                                                   int cuda_version = 12600);
[[nodiscard]] CUresult intercept_driver_copy(CUdeviceptr destination, CUdeviceptr source,
                                             void* host_pointer, std::size_t memory_bytes,
                                             CUstream stream, unsigned int kind,
                                             bool per_thread_default_stream);
[[nodiscard]] bool cooperative_driver_preview_requested() noexcept;
[[nodiscard]] CUresult intercept_driver_uva_copy(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream,
                                                 bool per_thread_default_stream,
                                                 const std::function<CUresult()>& original);
// Trusted progress-pool hooks: use the same quota ledger even in a Runtime
// scope. Call only at an initialized, physically safe native context boundary.
[[nodiscard]] CUresult allocate_cooperative_progress(CUdeviceptr* pointer,
                                                     std::size_t memory_bytes) noexcept;
[[nodiscard]] CudaCooperativeReleaseResult release_cooperative_progress(
    CUdeviceptr pointer) noexcept;

[[nodiscard]] CUresult intercept_module_load_data(CUmodule* module, const void* image,
                                                  bool uses_extended_loader,
                                                  unsigned int option_count, CUjit_option* options,
                                                  void** option_values);
[[nodiscard]] CUresult intercept_module_get_function(CUfunction* function, CUmodule module,
                                                     const char* name);
[[nodiscard]] CUresult intercept_module_unload(CUmodule module);
[[nodiscard]] CUresult intercept_library_load_data(CUlibrary* library, const void* image,
                                                   CUjit_option* options, void** option_values,
                                                   unsigned int option_count,
                                                   CUlibraryOption* library_options,
                                                   void** library_option_values,
                                                   unsigned int library_option_count);
[[nodiscard]] CUresult intercept_library_load_from_file(CUlibrary* library, const char* path,
                                                        CUjit_option* options, void** option_values,
                                                        unsigned int option_count,
                                                        CUlibraryOption* library_options,
                                                        void** library_option_values,
                                                        unsigned int library_option_count);
[[nodiscard]] CUresult intercept_library_get_module(CUmodule* module, CUlibrary library);
[[nodiscard]] CUresult intercept_library_unload(CUlibrary library);

[[nodiscard]] CUresult intercept_init(unsigned int flags);
[[nodiscard]] CUresult intercept_event_create(CUevent* event, unsigned int flags);
[[nodiscard]] CUresult intercept_event_record(CUevent event, CUstream stream);
[[nodiscard]] CUresult intercept_event_query(CUevent event);
[[nodiscard]] CUresult intercept_event_destroy(CUevent event);
[[nodiscard]] CUresult intercept_event_record_ptsz(CUevent event, CUstream stream);
[[nodiscard]] CUresult intercept_event_record_with_flags(CUevent event, CUstream stream,
                                                         unsigned int flags);
[[nodiscard]] CUresult intercept_event_record_with_flags_ptsz(CUevent event, CUstream stream,
                                                              unsigned int flags);
[[nodiscard]] CUresult intercept_event_synchronize(CUevent event);
[[nodiscard]] CUresult intercept_event_elapsed_time(float* milliseconds, CUevent start,
                                                    CUevent end);
[[nodiscard]] CUresult intercept_stream_wait_event(CUstream stream, CUevent event,
                                                   unsigned int flags);
[[nodiscard]] CUresult intercept_stream_wait_event_ptsz(CUstream stream, CUevent event,
                                                        unsigned int flags);
[[nodiscard]] bool launch_scheduling_is_enforced() noexcept;
// A missing value prohibits admission; true delegates capture construction.
[[nodiscard]] std::optional<bool> launch_stream_is_capturing(
    CUstream stream, bool per_thread_default_stream) noexcept;
[[nodiscard]] bool launch_timing_is_enabled() noexcept;
[[nodiscard]] std::optional<glimmer::core::TaskId> acquire_launch_slot(
    glimmer::control::LaunchGateTiming* timing = nullptr) noexcept;
void fail_launch_slot(glimmer::core::TaskId task_id) noexcept;
[[nodiscard]] bool track_launch_completion(glimmer::core::TaskId task_id, CUstream stream) noexcept;
[[nodiscard]] bool finish_launch_batch_call(glimmer::core::TaskId task_id) noexcept;
void report_kernel_launch_observed(const KernelLaunchObservation& observation) noexcept;
void report_graph_launch_observed(const char* api_name, const void* stream) noexcept;
void report_memory_info_observed(const char* api_name, std::int32_t device,
                                 std::uint64_t total_bytes, std::uint64_t free_bytes,
                                 std::uint64_t physical_total_bytes,
                                 std::uint64_t physical_free_bytes) noexcept;
void report_launch_timing_observed(glimmer::core::TaskId task_id,
                                   const glimmer::control::LaunchGateTiming& timing,
                                   std::uint64_t cuda_launch_nanoseconds,
                                   std::uint64_t event_tracking_nanoseconds,
                                   std::uint32_t cuda_launch_status,
                                   bool event_tracking_succeeded) noexcept;
[[nodiscard]] CUresult intercept_launch_kernel(CUfunction function, unsigned int grid_dim_x,
                                               unsigned int grid_dim_y, unsigned int grid_dim_z,
                                               unsigned int block_dim_x, unsigned int block_dim_y,
                                               unsigned int block_dim_z,
                                               unsigned int shared_memory_bytes, CUstream stream,
                                               void** kernel_parameters, void** extra,
                                               bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_graph_launch(CUgraphExec graph_exec, CUstream stream,
                                              bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_launch_kernel_ex(const CUlaunchConfig* config, CUfunction function,
                                                  void** kernel_parameters, void** extra,
                                                  bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes);
[[nodiscard]] CUresult intercept_mem_alloc_managed(CUdeviceptr* device_pointer,
                                                   std::size_t memory_bytes, unsigned int flags);
[[nodiscard]] CUresult intercept_mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                                 std::size_t width_bytes, std::size_t height,
                                                 unsigned int element_size_bytes);
[[nodiscard]] CUresult intercept_mem_alloc_async(CUdeviceptr* device_pointer,
                                                 std::size_t memory_bytes, CUstream stream,
                                                 bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                           std::size_t memory_bytes,
                                                           CUmemoryPool pool, CUstream stream,
                                                           bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_create(CUmemGenericAllocationHandle* handle,
                                            std::size_t memory_bytes,
                                            const CUmemAllocationProp* prop,
                                            unsigned long long flags);
[[nodiscard]] CUresult intercept_mem_release(CUmemGenericAllocationHandle handle);
[[nodiscard]] CUresult intercept_mem_address_reserve(CUdeviceptr* device_pointer,
                                                     std::size_t memory_bytes,
                                                     std::size_t alignment,
                                                     CUdeviceptr requested_address,
                                                     unsigned long long flags);
[[nodiscard]] CUresult intercept_mem_address_free(CUdeviceptr device_pointer,
                                                  std::size_t memory_bytes);
[[nodiscard]] CUresult intercept_mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                         std::size_t offset, CUmemGenericAllocationHandle handle,
                                         unsigned long long flags);
[[nodiscard]] CUresult intercept_mem_map_array_async(CUarrayMapInfo* map_info_list,
                                                     unsigned int count, CUstream stream);
[[nodiscard]] CUresult intercept_mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes);
[[nodiscard]] CUresult intercept_mem_set_access(CUdeviceptr device_pointer,
                                                std::size_t memory_bytes,
                                                const CUmemAccessDesc* access_descriptors,
                                                std::size_t descriptor_count);
[[nodiscard]] CUresult intercept_mem_get_address_range(CUdeviceptr* base_pointer,
                                                       std::size_t* memory_bytes,
                                                       CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_mem_get_access(unsigned long long* flags,
                                                const CUmemLocation* location,
                                                CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_mem_export_to_shareable_handle(
    void* shareable_handle, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handle_type, unsigned long long flags);
[[nodiscard]] CUresult intercept_mem_import_from_shareable_handle(
    CUmemGenericAllocationHandle* handle, void* os_handle, CUmemAllocationHandleType handle_type);
[[nodiscard]] CUresult intercept_ipc_get_mem_handle(CUipcMemHandle* handle,
                                                    CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_ipc_open_mem_handle(CUdeviceptr* device_pointer,
                                                     CUipcMemHandle handle, unsigned int flags);
[[nodiscard]] CUresult intercept_ipc_close_mem_handle(CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_import_external_memory(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc);
[[nodiscard]] CUresult intercept_external_memory_get_mapped_buffer(
    CUdeviceptr* device_pointer, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc);
[[nodiscard]] CUresult intercept_external_memory_get_mapped_mipmapped_array(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc);
[[nodiscard]] CUresult intercept_destroy_external_memory(CUexternalMemory external_memory);
[[nodiscard]] CUresult intercept_array_create(CUarray* array,
                                              const CUDA_ARRAY_DESCRIPTOR* descriptor);
[[nodiscard]] CUresult intercept_array_3d_create(CUarray* array,
                                                 const CUDA_ARRAY3D_DESCRIPTOR* descriptor);
[[nodiscard]] CUresult intercept_array_destroy(CUarray array);
[[nodiscard]] CUresult intercept_mipmapped_array_create(CUmipmappedArray* mipmap,
                                                        const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                        unsigned int level_count);
[[nodiscard]] CUresult intercept_mipmapped_array_destroy(CUmipmappedArray mipmap);
[[nodiscard]] CUresult intercept_graphics_unregister_resource(CUgraphicsResource resource);
[[nodiscard]] CUresult intercept_graphics_subresource_get_mapped_array(CUarray* array,
                                                                       CUgraphicsResource resource,
                                                                       unsigned int array_index,
                                                                       unsigned int mip_level);
[[nodiscard]] CUresult intercept_graphics_resource_get_mapped_mipmapped_array(
    CUmipmappedArray* mipmap, CUgraphicsResource resource);
[[nodiscard]] CUresult intercept_graphics_resource_get_mapped_pointer(CUdeviceptr* device_pointer,
                                                                      std::size_t* size,
                                                                      CUgraphicsResource resource);
[[nodiscard]] CUresult intercept_graphics_resource_set_map_flags(CUgraphicsResource resource,
                                                                 unsigned int flags);
[[nodiscard]] CUresult intercept_graphics_map_resources(unsigned int count,
                                                        CUgraphicsResource* resources,
                                                        CUstream stream);
[[nodiscard]] CUresult intercept_graphics_unmap_resources(unsigned int count,
                                                          CUgraphicsResource* resources,
                                                          CUstream stream);
[[nodiscard]] CUresult intercept_mem_get_allocation_granularity(
    std::size_t* granularity, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option);
[[nodiscard]] CUresult intercept_mem_get_allocation_properties(CUmemAllocationProp* prop,
                                                               CUmemGenericAllocationHandle handle);
[[nodiscard]] CUresult intercept_mem_retain_allocation_handle(CUmemGenericAllocationHandle* handle,
                                                              void* device_pointer);
[[nodiscard]] CUresult intercept_mem_pool_trim_to(CUmemoryPool pool, std::size_t min_bytes_to_keep);
[[nodiscard]] CUresult intercept_mem_pool_set_attribute(CUmemoryPool pool,
                                                        CUmemPool_attribute attribute, void* value);
[[nodiscard]] CUresult intercept_mem_pool_get_attribute(CUmemoryPool pool,
                                                        CUmemPool_attribute attribute, void* value);
[[nodiscard]] CUresult intercept_mem_pool_set_access(CUmemoryPool pool,
                                                     const CUmemAccessDesc* access_descriptors,
                                                     std::size_t descriptor_count);
[[nodiscard]] CUresult intercept_mem_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                                     CUmemLocation* location);
[[nodiscard]] CUresult intercept_mem_pool_create(CUmemoryPool* pool,
                                                 const CUmemPoolProps* properties);
[[nodiscard]] CUresult intercept_mem_pool_destroy(CUmemoryPool pool);
[[nodiscard]] CUresult intercept_device_get_mem_pool(CUmemoryPool* pool, CUdevice device);
[[nodiscard]] CUresult intercept_device_set_mem_pool(CUdevice device, CUmemoryPool pool);
[[nodiscard]] CUresult intercept_device_get_default_mem_pool(CUmemoryPool* pool, CUdevice device);
[[nodiscard]] CUresult intercept_mem_get_default_mem_pool(CUmemoryPool* pool,
                                                          CUmemLocation* location,
                                                          CUmemAllocationType allocation_type);
[[nodiscard]] CUresult intercept_mem_get_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                                  CUmemAllocationType allocation_type);
[[nodiscard]] CUresult intercept_mem_set_mem_pool(CUmemLocation* location,
                                                  CUmemAllocationType allocation_type,
                                                  CUmemoryPool pool);
[[nodiscard]] CUresult intercept_mem_pool_export_to_shareable_handle(
    void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handle_type,
    unsigned long long flags);
[[nodiscard]] CUresult intercept_mem_pool_import_from_shareable_handle(
    CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handle_type,
    unsigned long long flags);
[[nodiscard]] CUresult intercept_mem_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                                         CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_mem_pool_import_pointer(CUdeviceptr* pointer_out,
                                                         CUmemoryPool pool,
                                                         CUmemPoolPtrExportData* share_data);
[[nodiscard]] nvmlReturn_t intercept_nvml_init();
[[nodiscard]] nvmlReturn_t intercept_nvml_init_with_flags(unsigned int flags);
[[nodiscard]] nvmlReturn_t intercept_nvml_shutdown();
[[nodiscard]] nvmlReturn_t intercept_nvml_device_get_count(unsigned int* device_count);
[[nodiscard]] nvmlReturn_t intercept_nvml_device_get_handle_by_index(unsigned int index,
                                                                     nvmlDevice_t* device);
[[nodiscard]] nvmlReturn_t intercept_nvml_device_get_index(nvmlDevice_t device,
                                                           unsigned int* index);
[[nodiscard]] nvmlReturn_t intercept_nvml_device_get_memory_info(nvmlDevice_t device,
                                                                 nvmlMemory_t* memory);
[[nodiscard]] nvmlReturn_t intercept_nvml_device_get_memory_info_v2(nvmlDevice_t device,
                                                                    nvmlMemory_v2_t* memory);
[[nodiscard]] CUresult intercept_mem_free(CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_mem_free_async(CUdeviceptr device_pointer, CUstream stream,
                                                bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes);
[[nodiscard]] CUresult intercept_device_total_mem(std::size_t* total_bytes, CUdevice device);
[[nodiscard]] CUresult intercept_context_get_current(CUcontext* context);
[[nodiscard]] CUresult intercept_context_get_device(CUdevice* device);
[[nodiscard]] CUresult intercept_context_destroy(CUcontext context, bool is_modern = true);
[[nodiscard]] CUresult intercept_primary_context_reset(CUdevice device, bool is_modern);
[[nodiscard]] CUresult intercept_primary_context_release(CUdevice device, bool is_modern);
[[nodiscard]] CUresult intercept_stream_get_device(CUstream stream, CUdevice* device,
                                                   bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_get_context(CUstream stream, CUcontext* context,
                                                    bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_query(CUstream stream, bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_synchronize(CUstream stream,
                                                    bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_destroy(CUstream stream);
[[nodiscard]] CUresult intercept_context_synchronize();
[[nodiscard]] CUresult intercept_get_proc_address(const char* symbol, void** function_pointer,
                                                  int cuda_version, cuuint64_t flags);
[[nodiscard]] CUresult intercept_get_proc_address_v2(const char* symbol, void** function_pointer,
                                                     int cuda_version, cuuint64_t flags,
                                                     CUdriverProcAddressQueryResult* symbol_status);

}  // namespace glimmer::interceptor
