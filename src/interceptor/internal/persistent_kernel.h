#pragma once

#include "driver_dispatch.h"
#include "ptx_transform.h"
#include "qualified_cubin.h"

#include <cstdint>
#include <array>
#include <atomic>
#include <mutex>
#include <memory>
#include <unordered_map>

namespace glimmer::interceptor {

struct RuntimeKernelVariant {
    CUkernel kernel = nullptr;
    std::size_t parameter_count = 0;
    PtxExecutionKind execution_kind = PtxExecutionKind::kStatic;
    std::vector<PtxParameterLayout> parameter_layout{};
    CUfunction function = nullptr;
    // Borrowed SDK identity, never an owned module or unload permission.
    CUmodule original_module = nullptr;
    CUfunction original_function = nullptr;
};

struct PersistentLaunchShape {
    unsigned int logical_blocks = 0;
    unsigned int block_threads = 0;
};

struct CooperativeKernelVariant {
    CUcontext context{};
    CUfunction function{};
    std::vector<PtxParameterLayout> parameter_layout;
    PersistentLaunchShape shape;
    unsigned int worker_blocks{};
    // Trusted private argument indices for pool-owned mapped audit storage.
    struct AuditParameters {
        std::size_t first_tile;
        std::size_t tile_counts;
        // Only a qualified provider may write the adjacent data-dependent word.
        bool has_compute_witness{};
    };
    std::optional<AuditParameters> audit{};
};

struct OwnedSgemmCode {
    CUfunction function{};
    unsigned int worker_blocks{};
    std::shared_ptr<void> resources;
};

struct MappedBinaryKernelVariant {
    CUcontext context{};
    CUfunction function{};
    // Retains the private library through queued execution. Does not authorize
    // using a function after its context is destroyed or bypass physical drain.
    std::shared_ptr<void> resources;
    QualifiedAddMapping mapping{QualifiedAddMapping::kSingleCta};
};

// Distinct from the mapped-add ABI. Code ownership alone does not qualify a
// GEMM launch's parameters, allocations or physical batch independence.
struct MappedGemmKernelVariant {
    CUcontext context{};
    CUfunction function{};
    std::shared_ptr<void> resources;
    QualifiedGemmMapping mapping{QualifiedGemmMapping::kPlane};
};

// Process-local metadata only. Driver calls never run under mutex_. Callers
// must observe CUDA's normal module/context lifetime and concurrency rules.
class PersistentKernelRuntime {
   public:
    PersistentKernelRuntime(const DriverDispatch& driver, std::uint32_t worker_blocks,
                            bool should_trace, bool should_prepare_cooperative = false) noexcept;

    [[nodiscard]] CUresult load_module(CUmodule* module, const void* image,
                                       bool uses_extended_loader = false,
                                       unsigned int option_count = 0,
                                       CUjit_option* options = nullptr,
                                       void** option_values = nullptr);
    [[nodiscard]] CUresult get_function(CUfunction* function, CUmodule module, const char* name);
    // Source ownership permits original SDK issue and owned CPU metadata only.
    [[nodiscard]] CUresult load_source_module(CUmodule* module, const void* image,
                                              bool uses_extended_loader, unsigned int option_count,
                                              CUjit_option* options, void** option_values);
    [[nodiscard]] CUresult get_source_function(CUfunction* function, CUmodule module,
                                               const char* name);
    // Original unload and CPU retirement only; private cleanup needs normal proof.
    [[nodiscard]] CUresult unload_source_module(CUmodule module);
    // Readiness is CPU-only. Cold preparation requires a normal native boundary.
    [[nodiscard]] bool module_source_is_prepared(CUfunction function) const;
    [[nodiscard]] bool module_lookup_is_prepared(CUmodule module, const char* name) const;
    void prepare_module_source(CUfunction function);
    // CPU-only lookup by the original successful source function association.
    // Not public launch selection; caller owes context/stream/shape validation.
    [[nodiscard]] std::optional<MappedBinaryKernelVariant> mapped_binary_kernel(
        CUfunction original_function) const;
    [[nodiscard]] std::optional<MappedGemmKernelVariant> mapped_gemm_kernel(
        CUfunction original_function) const;
    [[nodiscard]] CUresult unload_module(CUmodule module);
    // Source observations only: original library/module ownership stays in CUDA.
    // Call after the exact original operation succeeds; no native calls here.
    void observe_library_source(CUlibrary library, const void* image) noexcept;
    void observe_library_module(CUlibrary library, CUmodule module, CUcontext context) noexcept;
    void retire_library_source(CUlibrary library) noexcept;
    void forget_context(CUcontext context);
    // Checked CPU retirement plus context-independent library unload. Caller
    // serializes native lifetime; never binds or queries old context resources.
    [[nodiscard]] CUresult forget_context_checked(CUcontext context);

    // Cold preparation requires a drained native boundary in this exact current
    // context. Warm lookup is metadata-only; the caller owes launch qualification.
    [[nodiscard]] CUresult prepare_owned_sgemm(CUcontext context);
    [[nodiscard]] std::optional<OwnedSgemmCode> owned_sgemm(CUcontext context) const;

    [[nodiscard]] std::optional<PersistentLaunchShape> validated_launch_shape(
        std::array<unsigned int, 3> grid, std::array<unsigned int, 3> block,
        PtxExecutionKind kind) const;

    // Owned metadata only; never retains caller arguments or attempts a launch.
    // The frontend must own progress/private ABI and fence module destruction.
    [[nodiscard]] std::optional<CooperativeKernelVariant> cooperative_kernel(
        CUfunction function, std::array<unsigned int, 3> grid, std::array<unsigned int, 3> block,
        unsigned int shared_memory_bytes, CUstream stream, bool per_thread_default_stream) const;

    // A missing value means no launch was attempted: use the native Driver path.
    [[nodiscard]] std::optional<CUresult> try_launch(
        CUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
        unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
        unsigned int block_dim_z, unsigned int shared_memory_bytes, CUstream stream,
        void** kernel_parameters, void** extra, bool per_thread_default_stream);

    [[nodiscard]] std::optional<RuntimeKernelVariant> runtime_kernel(
        CUcontext context, const void* host_function, const std::string& name,
        const std::string& binary_path, const std::string& tool_path,
        CUfunction original_function = nullptr);

    // Metadata-only readiness query. A public cold caller must obtain the
    // process native-boundary ownership before invoking runtime_kernel().
    [[nodiscard]] bool runtime_cooperative_source_is_prepared(CUcontext context,
                                                              const std::string& binary_path) const;

    [[nodiscard]] CUresult launch(CUfunction function, unsigned int grid_dim_x,
                                  unsigned int grid_dim_y, unsigned int grid_dim_z,
                                  unsigned int block_dim_x, unsigned int block_dim_y,
                                  unsigned int block_dim_z, unsigned int shared_memory_bytes,
                                  CUstream stream, void** kernel_parameters, void** extra,
                                  bool per_thread_default_stream);

   private:
    struct BinarySourceRecord {
        std::vector<std::byte> image;
        bool is_gemm{};
    };
    static std::shared_ptr<const BinarySourceRecord> snapshot_binary_source(const void* image);
    struct DeferredModuleRecord {
        std::once_flag initialization;
        CUcontext context{};
        CUmodule module{};
        PersistentPtx ptx;
        std::vector<CUfunction> functions;
        CUlibrary library{};
        std::atomic<bool> is_prepared{false};
        bool is_original_only{};
        bool is_retired{};
        bool can_unload{};
        bool has_loaded_originals{};
        bool is_binary{};
        CUlibrary original_library{};
        std::shared_ptr<const BinarySourceRecord> binary_source;
    };
    struct ModuleRecord {
        CUcontext context = nullptr;
        std::vector<PtxKernelVariant> kernels;
        std::shared_ptr<DeferredModuleRecord> deferred{};
    };
    struct FunctionRecord {
        CUmodule module = nullptr;
        CUcontext context = nullptr;
        CUfunction persistent = nullptr;
        std::size_t parameter_count = 0;
        PtxExecutionKind execution_kind = PtxExecutionKind::kStatic;
        std::vector<PtxParameterLayout> parameter_layout{};
        std::shared_ptr<DeferredModuleRecord> deferred{};
        std::size_t deferred_index{};
    };
    struct SourceRecord {
        std::once_flag initialization;
        std::optional<std::vector<std::string>> images;
    };
    struct RuntimeFunctionRecord {
        std::once_flag initialization;
        CUcontext context = nullptr;
        const void* host_function = nullptr;
        std::string name;
        std::string binary_path;
        CUlibrary library = nullptr;
        RuntimeKernelVariant variant;
    };
    struct RuntimeLibraryOwner {
        CUlibrary library = nullptr;
        bool can_unload = false;
    };
    struct OwnedSgemmRecord {
        CUcontext context{};
        CUlibrary library{};
        CUfunction function{};
        CUresult status{CUDA_ERROR_NOT_READY};
        bool can_unload{};
        bool is_retired{};
        std::atomic<bool> is_prepared{false};
    };
    struct RuntimeCooperativeSourceRecord {
        std::once_flag initialization;
        CUcontext context = nullptr;
        std::string binary_path;
        std::atomic<bool> is_prepared{false};
        std::vector<RuntimeLibraryOwner> libraries;
        std::unordered_map<std::string, RuntimeKernelVariant> kernels;
    };

    struct RuntimeSourceLocation {
        const std::string& binary_path;
        const std::string& tool_path;
    };
    [[nodiscard]] std::optional<RuntimeKernelVariant> runtime_cooperative_kernel(
        CUcontext context, const std::string& name, RuntimeSourceLocation location,
        CUfunction original_function);

    void forget_module(CUmodule module);
    void forget_module_locked(CUmodule module);
    void retire_library_locked(CUlibrary library);
    CUresult clean_retired_modules();

    const DriverDispatch& driver_;
    std::uint32_t worker_blocks_;
    bool should_trace_;
    bool should_prepare_cooperative_;
    mutable std::mutex mutex_;
    std::unordered_map<CUmodule, ModuleRecord> modules_;
    std::unordered_map<CUfunction, FunctionRecord> functions_;
    std::vector<std::shared_ptr<DeferredModuleRecord>> deferred_modules_;
    std::unordered_map<CUlibrary, std::shared_ptr<const BinarySourceRecord>> library_sources_;
    std::unordered_map<std::string, std::shared_ptr<SourceRecord>> runtime_sources_;
    std::vector<std::shared_ptr<RuntimeFunctionRecord>> runtime_functions_;
    std::vector<std::shared_ptr<RuntimeCooperativeSourceRecord>> runtime_cooperative_sources_;
    std::vector<std::shared_ptr<OwnedSgemmRecord>> owned_sgemm_records_;
};

}  // namespace glimmer::interceptor
