#include "persistent_kernel.h"

#include "diagnostics.h"
#include "embedded_ptx.h"
#include "qualified_cubin_source.h"
#include "persistent_sgemm.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

namespace glimmer::interceptor {
namespace {

constexpr std::size_t kMaxCooperativeSources = 256;

PtxKernelVariant binary_metadata(bool is_gemm) {
    // Name/layout metadata only. Binary owners are excluded from PTX selection.
    if (is_gemm)
        return {kQualifiedGemmEntry, kQualifiedGemmEntry, 1, {{176, 8}}, PtxExecutionKind::kStatic};
    return {kQualifiedAddEntry,
            kQualifiedAddEntry,
            3,
            {{4, 4}, {4, 4}, {24, 8}},
            PtxExecutionKind::kStatic};
}

bool supports_qualified_binary(const DriverDispatch& driver) {
    CUdevice device{};
    int major{}, minor{};
    return driver.context_get_device(&device) == CUDA_SUCCESS &&
           driver.device_get_attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                       device) == CUDA_SUCCESS &&
           driver.device_get_attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                       device) == CUDA_SUCCESS &&
           major == 7 && minor == 0;
}

std::optional<std::string_view> textual_ptx(const void* image) {
    if (image == nullptr) {
        return std::nullopt;
    }
    const auto* text = static_cast<const char*>(image);
    // ELF cubins and CUDA fatbinary headers do not begin with these bytes.
    // The caller still owes the Driver a valid, readable module image.
    if (text[0] != '.' && text[0] != '/' && text[0] != '\n' && text[0] != '\r' && text[0] != '\t' &&
        text[0] != ' ') {
        return std::nullopt;
    }
    const auto length = ::strnlen(text, kMaxPersistentPtxBytes + 1);
    if (length > kMaxPersistentPtxBytes) {
        return std::nullopt;
    }
    return std::string_view{text, length};
}

void trace_persistent_launch(unsigned int logical_blocks, unsigned int worker_blocks) noexcept {
    std::array<char, 160> message{};
    const auto written = std::snprintf(message.data(), message.size(),
                                       "[glimmer] persistent_kernel logical_blocks=%u "
                                       "worker_blocks=%u\n",
                                       logical_blocks, worker_blocks);
    if (written > 0 && static_cast<std::size_t>(written) < message.size()) {
        report_diagnostic(message.data());
    }
}

}  // namespace

PersistentKernelRuntime::PersistentKernelRuntime(const DriverDispatch& driver,
                                                 std::uint32_t worker_blocks, bool should_trace,
                                                 bool should_prepare_cooperative) noexcept
    : driver_(driver),
      worker_blocks_(worker_blocks),
      should_trace_(should_trace),
      should_prepare_cooperative_(should_prepare_cooperative) {}

std::optional<PersistentLaunchShape> PersistentKernelRuntime::validated_launch_shape(
    std::array<unsigned int, 3> grid, std::array<unsigned int, 3> block,
    PtxExecutionKind kind) const {
    if (kind != PtxExecutionKind::kStatic && kind != PtxExecutionKind::kBlockStatic &&
        kind != PtxExecutionKind::kPreemptive) {
        return std::nullopt;
    }
    std::uint64_t logical_blocks = 1;
    std::uint64_t block_threads = 1;
    for (std::size_t axis = 0; axis < grid.size(); ++axis) {
        if (grid[axis] == 0 || block[axis] == 0 ||
            grid[axis] > std::numeric_limits<unsigned int>::max() / logical_blocks ||
            block[axis] > 1024 / block_threads) {
            return std::nullopt;
        }
        logical_blocks *= grid[axis];
        block_threads *= block[axis];
    }
    if (worker_blocks_ == 0 ||
        (kind != PtxExecutionKind::kPreemptive && logical_blocks <= worker_blocks_)) {
        return std::nullopt;
    }
    if (kind == PtxExecutionKind::kStatic) {
        if (grid[1] != 1 || grid[2] != 1 || block[1] != 1 || block[2] != 1 ||
            grid[0] > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
            return std::nullopt;
        }
    } else {
        CUdevice device = 0;
        if (driver_.context_get_device(&device) != CUDA_SUCCESS) {
            return std::nullopt;
        }
        constexpr std::array<CUdevice_attribute, 3> k_grid_limits{
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
            CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z};
        constexpr std::array<CUdevice_attribute, 3> k_block_limits{
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z};
        for (std::size_t axis = 0; axis < grid.size(); ++axis) {
            int grid_limit = 0;
            int block_limit = 0;
            if (driver_.device_get_attribute(&grid_limit, k_grid_limits[axis], device) !=
                    CUDA_SUCCESS ||
                driver_.device_get_attribute(&block_limit, k_block_limits[axis], device) !=
                    CUDA_SUCCESS ||
                grid_limit <= 0 || block_limit <= 0 ||
                grid[axis] > static_cast<unsigned int>(grid_limit) ||
                (axis == 0 && (kind == PtxExecutionKind::kPreemptive
                                   ? std::min<std::uint64_t>(worker_blocks_, logical_blocks)
                                   : worker_blocks_) > static_cast<unsigned int>(grid_limit)) ||
                block[axis] > static_cast<unsigned int>(block_limit)) {
                return std::nullopt;
            }
        }
        int thread_limit = 0;
        if (driver_.device_get_attribute(&thread_limit, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                         device) != CUDA_SUCCESS ||
            thread_limit <= 0 || block_threads > static_cast<unsigned int>(thread_limit)) {
            return std::nullopt;
        }
    }
    return PersistentLaunchShape{static_cast<unsigned int>(logical_blocks),
                                 static_cast<unsigned int>(block_threads)};
}

CUresult PersistentKernelRuntime::load_module(CUmodule* module, const void* image,
                                              bool uses_extended_loader, unsigned int option_count,
                                              CUjit_option* options, void** option_values) {
    const auto load = [&](CUmodule* output, const void* input) {
        return uses_extended_loader ? driver_.module_load_data_ex(output, input, option_count,
                                                                  options, option_values)
                                    : driver_.module_load_data(output, input);
    };
    const auto load_original = [&] {
        const auto result = load(module, image);
        if (result == CUDA_SUCCESS && module != nullptr) {
            forget_module(*module);
        }
        return result;
    };
    std::optional<PersistentPtx> transformed;
    CUcontext context = nullptr;
    if (module != nullptr && option_count == 0 && worker_blocks_ > 0 &&
        driver_.has_persistent_kernel_api() &&
        driver_.context_get_current(&context) == CUDA_SUCCESS && context != nullptr) {
        try {
            const auto ptx = textual_ptx(image);
            if (ptx.has_value()) {
                transformed = should_prepare_cooperative_
                                  ? transform_persistent_ptx(*ptx, PtxExecutionKind::kPreemptive)
                                  : transform_transparent_ptx(*ptx);
            }
        } catch (...) {
            // Host metadata allocation must not prevent a native module load.
        }
    }
    if (!transformed.has_value()) {
        if (should_prepare_cooperative_) {
            return load_source_module(module, image, uses_extended_loader, option_count, options,
                                      option_values);
        }
        return load_original();
    }
    CUmodule candidate = nullptr;
    if (load(&candidate, transformed->image.c_str()) != CUDA_SUCCESS) {
        if (should_trace_ && should_prepare_cooperative_) {
            report_diagnostic("[glimmer] cooperative PTX JIT rejected; using original module\n");
        }
        return load_original();
    }
    *module = candidate;
    forget_module(candidate);
    try {
        std::scoped_lock lock(mutex_);
        modules_.insert_or_assign(candidate,
                                  ModuleRecord{context, std::move(transformed->kernels)});
    } catch (...) {
        // Both entries are in the same module. Without metadata, original
        // functions remain usable and the Driver still owns all module storage.
    }
    return CUDA_SUCCESS;
}

CUresult PersistentKernelRuntime::get_function(CUfunction* function, CUmodule module,
                                               const char* name) {
    bool is_deferred{};
    {
        const std::scoped_lock lock(mutex_);
        const auto found = modules_.find(module);
        is_deferred = found != modules_.end() && found->second.deferred;
    }
    if (is_deferred) {
        return get_source_function(function, module, name);
    }
    const auto result = driver_.module_get_function(function, module, name);
    if (result != CUDA_SUCCESS || function == nullptr || name == nullptr) {
        return result;
    }
    try {
        std::optional<PtxKernelVariant> variant;
        CUcontext context = nullptr;
        {
            std::scoped_lock lock(mutex_);
            functions_.erase(*function);
            const auto found = modules_.find(module);
            if (found == modules_.end()) {
                return result;
            }
            context = found->second.context;
            for (const auto& kernel : found->second.kernels) {
                if (kernel.original_name == name) {
                    variant = kernel;
                    break;
                }
            }
        }
        CUfunction persistent = nullptr;
        if (variant.has_value() &&
            driver_.module_get_function(&persistent, module, variant->persistent_name.c_str()) ==
                CUDA_SUCCESS &&
            persistent != nullptr) {
            std::scoped_lock lock(mutex_);
            functions_.insert_or_assign(
                *function, FunctionRecord{module, context, persistent, variant->parameter_count,
                                          variant->execution_kind, variant->parameter_layout});
        }
    } catch (...) {
        // The caller already received the original Driver function.
    }
    return result;
}

std::shared_ptr<const PersistentKernelRuntime::BinarySourceRecord>
PersistentKernelRuntime::snapshot_binary_source(const void* image) {
    auto mapped = snapshot_qualified_add_cubin(image, QualifiedAddMapping::kRange);
    const bool is_gemm = !mapped;
    if (is_gemm) mapped = snapshot_qualified_gemm_cubin(image, QualifiedGemmMapping::kRange);
    return mapped ? std::make_shared<const BinarySourceRecord>(
                        BinarySourceRecord{std::move(*mapped), is_gemm})
                  : nullptr;
}

CUresult PersistentKernelRuntime::load_source_module(CUmodule* module, const void* image,
                                                     bool uses_extended_loader,
                                                     unsigned int option_count,
                                                     CUjit_option* options, void** option_values) {
    std::shared_ptr<DeferredModuleRecord> deferred;
    ModuleRecord metadata;
    bool can_snapshot_binary = false;
    try {
        if (should_prepare_cooperative_ && module && option_count == 0 && worker_blocks_ > 0 &&
            driver_.has_persistent_kernel_api() && driver_.has_runtime_persistent_api() &&
            driver_.context_get_current(&metadata.context) == CUDA_SUCCESS && metadata.context) {
            can_snapshot_binary = true;
            const auto text = textual_ptx(image);
            auto transformed = text ? transform_persistent_ptx(*text, PtxExecutionKind::kPreemptive)
                                    : std::nullopt;
            if (transformed) {
                deferred = std::make_shared<DeferredModuleRecord>();
                deferred->context = metadata.context;
                deferred->ptx = std::move(*transformed);
                deferred->functions.resize(deferred->ptx.kernels.size());
                deferred->is_original_only = true;
                metadata.kernels = deferred->ptx.kernels;
                metadata.deferred = deferred;
            }
        }
    } catch (...) {
        deferred.reset();
    }
    const auto result =
        uses_extended_loader
            ? driver_.module_load_data_ex(module, image, option_count, options, option_values)
            : driver_.module_load_data(module, image);
    if (result != CUDA_SUCCESS || !module) {
        return result;
    }
    forget_module(*module);
    if (!deferred && can_snapshot_binary && *module) {
        try {
            auto source = snapshot_binary_source(image);
            if (source) {
                deferred = std::make_shared<DeferredModuleRecord>();
                deferred->context = metadata.context;
                deferred->binary_source = std::move(source);
                deferred->is_binary = true;
                deferred->is_original_only = true;
                deferred->ptx.kernels.push_back(binary_metadata(deferred->binary_source->is_gemm));
                deferred->functions.resize(1);
                metadata.kernels = deferred->ptx.kernels;
                metadata.deferred = deferred;
            }
        } catch (...) {
            deferred.reset();
        }
    }
    if (deferred && *module) {
        try {
            const std::scoped_lock lock(mutex_);
            if (deferred_modules_.size() < kMaxCooperativeSources) {
                deferred->module = *module;
                deferred->is_retired = true;
                deferred_modules_.push_back(deferred);
                modules_.insert_or_assign(*module, std::move(metadata));
                deferred->is_retired = false;
            }
        } catch (...) {
            // Original success stays authoritative; no private CUDA owner exists.
        }
    }
    return result;
}

CUresult PersistentKernelRuntime::get_source_function(CUfunction* function, CUmodule module,
                                                      const char* name) {
    const auto result = driver_.module_get_function(function, module, name);
    if (result != CUDA_SUCCESS || !function || !*function || !name) {
        return result;
    }
    try {
        const std::scoped_lock lock(mutex_);
        functions_.erase(*function);
        const auto found = modules_.find(module);
        if (found == modules_.end() || !should_prepare_cooperative_) {
            return result;
        }
        auto& metadata = found->second;
        const auto kernel =
            std::find_if(metadata.kernels.begin(), metadata.kernels.end(),
                         [&](const auto& item) { return item.original_name == name; });
        if (kernel == metadata.kernels.end()) {
            return result;
        }
        if (!metadata.deferred) {
            if (deferred_modules_.size() >= kMaxCooperativeSources) {
                return result;
            }
            auto owner = std::make_shared<DeferredModuleRecord>();
            owner->context = metadata.context;
            owner->module = module;
            owner->ptx.kernels = metadata.kernels;
            owner->functions.resize(metadata.kernels.size());
            deferred_modules_.push_back(owner);
            metadata.deferred = std::move(owner);
        }
        const auto index = static_cast<std::size_t>(kernel - metadata.kernels.begin());
        functions_.insert_or_assign(
            *function, FunctionRecord{module, metadata.context, nullptr, kernel->parameter_count,
                                      kernel->execution_kind, kernel->parameter_layout,
                                      metadata.deferred, index});
    } catch (...) {
        // Never fail a successful original lookup because CPU metadata is absent.
    }
    return result;
}

bool PersistentKernelRuntime::module_source_is_prepared(CUfunction function) const {
    const std::scoped_lock lock(mutex_);
    const auto found = functions_.find(function);
    return found == functions_.end() || !found->second.deferred ||
           found->second.deferred->is_prepared.load(std::memory_order_acquire);
}

bool PersistentKernelRuntime::module_lookup_is_prepared(CUmodule module, const char* name) const {
    const std::scoped_lock lock(mutex_);
    const auto found = modules_.find(module);
    return name && found != modules_.end() && found->second.deferred &&
           found->second.deferred->is_prepared.load(std::memory_order_acquire) &&
           found->second.deferred->has_loaded_originals &&
           std::any_of(found->second.kernels.begin(), found->second.kernels.end(),
                       [&](const auto& kernel) { return kernel.original_name == name; });
}

void PersistentKernelRuntime::prepare_module_source(CUfunction function) {
    std::shared_ptr<DeferredModuleRecord> record;
    {
        const std::scoped_lock lock(mutex_);
        const auto found = functions_.find(function);
        if (found != functions_.end()) {
            record = found->second.deferred;
        }
    }
    if (!record || record->is_prepared.load(std::memory_order_acquire)) {
        return;
    }
    std::call_once(record->initialization, [&] {
        // Catch internally: partial native acquisition must never be retried.
        try {
            clean_retired_modules();
            CUcontext context{};
            if (driver_.context_get_current(&context) == CUDA_SUCCESS &&
                context == record->context && driver_.has_function_load() &&
                (!record->is_binary || supports_qualified_binary(driver_))) {
                bool has_loaded_originals = true;
                for (const auto& kernel : record->ptx.kernels) {
                    CUfunction original{};
                    if (driver_.module_get_function(&original, record->module,
                                                    kernel.original_name.c_str()) != CUDA_SUCCESS ||
                        !original || driver_.function_load(original) != CUDA_SUCCESS) {
                        has_loaded_originals = false;
                    }
                }
                record->has_loaded_originals = has_loaded_originals;
                bool can_lookup = has_loaded_originals;
                if (can_lookup && record->is_original_only) {
                    const void* image =
                        record->is_binary
                            ? static_cast<const void*>(record->binary_source->image.data())
                            : record->ptx.image.c_str();
                    const auto status = driver_.library_load_data(&record->library, image);
                    can_lookup = status == CUDA_SUCCESS && record->library;
                    record->can_unload = can_lookup;
                }
                if (can_lookup) {
                    for (std::size_t index = 0; index < record->ptx.kernels.size(); ++index) {
                        const auto& name = record->ptx.kernels[index].persistent_name;
                        CUfunction native_function{};
                        if (record->is_original_only) {
                            CUkernel kernel{};
                            if (driver_.library_get_kernel(&kernel, record->library,
                                                           name.c_str()) != CUDA_SUCCESS ||
                                !kernel ||
                                driver_.kernel_get_function(&native_function, kernel) !=
                                    CUDA_SUCCESS) {
                                continue;
                            }
                        } else if (driver_.module_get_function(&native_function, record->module,
                                                               name.c_str()) != CUDA_SUCCESS) {
                            continue;
                        }
                        if (native_function &&
                            driver_.function_load(native_function) == CUDA_SUCCESS) {
                            record->functions[index] = native_function;
                        }
                    }
                }
            }
        } catch (...) {
            std::fill(record->functions.begin(), record->functions.end(), nullptr);
        }
        record->is_prepared.store(true, std::memory_order_release);
    });
}

std::optional<MappedBinaryKernelVariant> PersistentKernelRuntime::mapped_binary_kernel(
    CUfunction original_function) const {
    const std::scoped_lock lock(mutex_);
    const auto found = functions_.find(original_function);
    if (found == functions_.end() || !found->second.deferred) return std::nullopt;
    const auto& owner = found->second.deferred;
    if (!owner->is_binary || !owner->binary_source || owner->binary_source->is_gemm ||
        !owner->is_prepared.load(std::memory_order_acquire) || owner->functions.size() != 1 ||
        !owner->functions[0])
        return std::nullopt;
    return MappedBinaryKernelVariant{owner->context, owner->functions[0], owner,
                                     QualifiedAddMapping::kRange};
}

std::optional<MappedGemmKernelVariant> PersistentKernelRuntime::mapped_gemm_kernel(
    CUfunction original_function) const {
    const std::scoped_lock lock(mutex_);
    const auto found = functions_.find(original_function);
    if (found == functions_.end() || !found->second.deferred) return std::nullopt;
    const auto& owner = found->second.deferred;
    if (!owner->is_binary || !owner->binary_source || !owner->binary_source->is_gemm ||
        !owner->is_prepared.load(std::memory_order_acquire) || owner->functions.size() != 1 ||
        !owner->functions[0])
        return std::nullopt;
    return MappedGemmKernelVariant{owner->context, owner->functions[0], owner,
                                   QualifiedGemmMapping::kRange};
}

CUresult PersistentKernelRuntime::clean_retired_modules() {
    CUresult status = CUDA_SUCCESS;
    std::vector<std::shared_ptr<DeferredModuleRecord>> retired;
    {
        const std::scoped_lock lock(mutex_);
        retired.reserve(deferred_modules_.size());
        for (const auto& record : deferred_modules_) {
            if (!record->is_retired) {
                continue;
            }
            // External module/context lifetime serialization excludes preparation.
            retired.push_back(record);
            record->is_retired = false;
        }
    }
    for (const auto& record : retired) {
        // Registry and this cleanup batch own two references after source
        // retirement. Binary variant leases may still protect private work.
        if (record->is_binary && record.use_count() > 2) {
            const std::scoped_lock lock(mutex_);
            record->is_retired = true;
            continue;
        }
        if (record->library && record->can_unload) {
            record->can_unload = false;
            const auto result = driver_.library_unload(record->library);
            if (result == CUDA_SUCCESS) {
                record->library = nullptr;
            } else {
                if (status == CUDA_SUCCESS) {
                    status = result;
                }
                report_diagnostic("[glimmer] deferred cooperative module cleanup uncertain\n");
            }
        }
    }
    const std::scoped_lock lock(mutex_);
    std::erase_if(deferred_modules_, [&](const auto& record) {
        return std::find(retired.begin(), retired.end(), record) != retired.end() &&
               !record->library;
    });
    return status;
}

CUresult PersistentKernelRuntime::unload_source_module(CUmodule module) {
    const auto result = driver_.module_unload(module);
    if (result == CUDA_SUCCESS) {
        forget_module(module);
    }
    return result;
}

CUresult PersistentKernelRuntime::unload_module(CUmodule module) {
    const auto result = unload_source_module(module);
    if (result == CUDA_SUCCESS) {
        clean_retired_modules();
    }
    return result;
}

void PersistentKernelRuntime::forget_module(CUmodule module) {
    std::scoped_lock lock(mutex_);
    forget_module_locked(module);
}

void PersistentKernelRuntime::forget_module_locked(CUmodule module) {
    const auto found = modules_.find(module);
    if (found != modules_.end() && found->second.deferred) {
        found->second.deferred->is_retired = true;
    }
    modules_.erase(module);
    std::erase_if(functions_,
                  [module](const auto& entry) { return entry.second.module == module; });
}

void PersistentKernelRuntime::retire_library_locked(CUlibrary library) {
    for (const auto& [module, record] : modules_) {
        (void)module;
        if (record.deferred && record.deferred->original_library == library)
            record.deferred->is_retired = true;
    }
    std::erase_if(modules_, [&](const auto& item) {
        return item.second.deferred && item.second.deferred->original_library == library;
    });
    std::erase_if(functions_, [&](const auto& item) {
        return item.second.deferred && item.second.deferred->original_library == library;
    });
    library_sources_.erase(library);
}

void PersistentKernelRuntime::retire_library_source(CUlibrary library) noexcept {
    if (!library) return;
    try {
        const std::scoped_lock lock(mutex_);
        retire_library_locked(library);
    } catch (...) {
        // CPU retirement does not issue or repeat native cleanup.
    }
}

void PersistentKernelRuntime::observe_library_source(CUlibrary library,
                                                     const void* image) noexcept {
    if (!library) return;
    std::shared_ptr<const BinarySourceRecord> pending;
    try {
        if (should_prepare_cooperative_ && worker_blocks_ && image)
            pending = std::make_shared<const BinarySourceRecord>();
    } catch (...) {
        // Even a failed optional allocation must retire a reused source handle.
    }
    try {
        {
            const std::scoped_lock lock(mutex_);
            retire_library_locked(library);
            if (!pending || library_sources_.size() >= kMaxCooperativeSources) return;
            library_sources_.insert_or_assign(library, pending);
        }
        auto source = snapshot_binary_source(image);
        const std::scoped_lock lock(mutex_);
        const auto found = library_sources_.find(library);
        // A concurrent unload or handle replacement cannot publish this old load.
        if (found == library_sources_.end() || found->second != pending) return;
        if (source)
            found->second = std::move(source);
        else
            library_sources_.erase(found);
    } catch (...) {
        // Optional CPU metadata never changes original library load success.
    }
}

void PersistentKernelRuntime::observe_library_module(CUlibrary library, CUmodule module,
                                                     CUcontext context) noexcept {
    if (!module) return;
    try {
        std::shared_ptr<const BinarySourceRecord> source;
        {
            const std::scoped_lock lock(mutex_);
            const auto found = library_sources_.find(library);
            if (context && found != library_sources_.end() && !found->second->image.empty())
                source = found->second;
            const auto existing = modules_.find(module);
            if (source && existing != modules_.end() && existing->second.context == context &&
                existing->second.deferred &&
                existing->second.deferred->original_library == library &&
                existing->second.deferred->binary_source == source)
                return;
            forget_module_locked(module);
        }
        if (!source) return;
        auto deferred = std::make_shared<DeferredModuleRecord>();
        deferred->context = context;
        deferred->module = module;
        deferred->original_library = library;
        deferred->binary_source = source;
        deferred->is_binary = true;
        deferred->is_original_only = true;
        deferred->is_retired = true;
        deferred->ptx.kernels.push_back(binary_metadata(source->is_gemm));
        deferred->functions.resize(1);
        ModuleRecord metadata{context, deferred->ptx.kernels, deferred};
        const std::scoped_lock lock(mutex_);
        const auto found = library_sources_.find(library);
        if (found == library_sources_.end() || found->second != source ||
            deferred_modules_.size() >= kMaxCooperativeSources)
            return;
        // Another successful observation may have installed this association
        // while metadata was allocated. Do not orphan its private owner.
        const auto existing = modules_.find(module);
        if (existing != modules_.end()) return;
        deferred_modules_.push_back(deferred);
        modules_.emplace(module, std::move(metadata));
        deferred->is_retired = false;
    } catch (...) {
        // Original module success remains authoritative; no private CUDA here.
    }
}

void PersistentKernelRuntime::forget_context(CUcontext context) {
    (void)forget_context_checked(context);
}

CUresult PersistentKernelRuntime::forget_context_checked(CUcontext context) {
    if (!context) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUresult status = CUDA_SUCCESS;
    std::vector<std::shared_ptr<RuntimeFunctionRecord>> forgotten;
    std::vector<std::shared_ptr<RuntimeCooperativeSourceRecord>> cooperative_forgotten;
    std::vector<std::shared_ptr<OwnedSgemmRecord>> sgemm_forgotten;
    {
        std::scoped_lock lock(mutex_);
        for (const auto& [module, record] : modules_) {
            (void)module;
            if (record.context == context && record.deferred) {
                record.deferred->is_retired = true;
            }
        }
        std::erase_if(modules_,
                      [context](const auto& entry) { return entry.second.context == context; });
        std::erase_if(functions_,
                      [context](const auto& entry) { return entry.second.context == context; });
        for (const auto& entry : runtime_functions_) {
            if (entry->context == context) {
                forgotten.push_back(entry);
                entry->context = nullptr;
            }
        }
        for (const auto& entry : runtime_cooperative_sources_) {
            if (context != nullptr && entry->context == context) {
                cooperative_forgotten.push_back(entry);
                // Retire before cleanup. A failed unload keeps an unreachable
                // metadata owner, not permission to retry with a reused context.
                entry->context = nullptr;
            }
        }
        for (const auto& entry : owned_sgemm_records_) {
            if (entry->context == context) {
                sgemm_forgotten.push_back(entry);
                entry->is_retired = true;
            }
        }
    }
    for (const auto& entry : sgemm_forgotten) {
        // The registry and this cleanup snapshot are the only allowed owners.
        // A retained queued lease is not permission to unload a live library.
        if (entry.use_count() != 2) {
            status = CUDA_ERROR_NOT_READY;
            continue;
        }
        if (entry->library && !entry->can_unload) {
            status = CUDA_ERROR_UNKNOWN;
            continue;
        }
        if (entry->library) {
            entry->can_unload = false;
            const auto result = driver_.library_unload(entry->library);
            if (result == CUDA_SUCCESS)
                entry->library = nullptr;
            else
                status = result;
        }
    }
    for (const auto& entry : forgotten) {
        if (entry->library != nullptr) {
            const auto result = driver_.library_unload(entry->library);
            if (result == CUDA_SUCCESS) {
                entry->library = nullptr;
            } else {
                if (status == CUDA_SUCCESS) {
                    status = result;
                }
                report_diagnostic("[glimmer] private Runtime PTX library cleanup failed\n");
            }
        }
    }
    for (const auto& entry : cooperative_forgotten) {
        for (auto& owner : entry->libraries) {
            if (owner.library != nullptr && owner.can_unload) {
                owner.can_unload = false;
                const auto result = driver_.library_unload(owner.library);
                if (result == CUDA_SUCCESS) {
                    owner.library = nullptr;
                } else {
                    if (status == CUDA_SUCCESS) {
                        status = result;
                    }
                    report_diagnostic("[glimmer] cooperative Runtime library cleanup uncertain\n");
                }
            }
        }
    }
    const auto modules_status = clean_retired_modules();
    if (status == CUDA_SUCCESS) {
        status = modules_status;
    }
    const std::scoped_lock lock(mutex_);
    std::erase_if(owned_sgemm_records_, [&](const auto& entry) {
        return std::find(sgemm_forgotten.begin(), sgemm_forgotten.end(), entry) !=
                   sgemm_forgotten.end() &&
               !entry->library;
    });
    std::erase_if(runtime_functions_, [&](const auto& entry) {
        return std::find(forgotten.begin(), forgotten.end(), entry) != forgotten.end() &&
               !entry->library;
    });
    std::erase_if(runtime_cooperative_sources_, [&](const auto& entry) {
        // Only inspect this call's retired owners. Another context may be
        // cleaning its native libraries concurrently outside the registry lock.
        return std::find(cooperative_forgotten.begin(), cooperative_forgotten.end(), entry) !=
                   cooperative_forgotten.end() &&
               std::all_of(entry->libraries.begin(), entry->libraries.end(),
                           [](const auto& owner) { return owner.library == nullptr; });
    });
    return status;
}

std::optional<OwnedSgemmCode> PersistentKernelRuntime::owned_sgemm(CUcontext context) const {
    const std::lock_guard lock(mutex_);
    for (const auto& entry : owned_sgemm_records_) {
        if (entry->context == context && !entry->is_retired &&
            entry->is_prepared.load(std::memory_order_acquire) && entry->status == CUDA_SUCCESS)
            return OwnedSgemmCode{entry->function, worker_blocks_, entry};
    }
    return std::nullopt;
}

CUresult PersistentKernelRuntime::prepare_owned_sgemm(CUcontext context) {
    if (!context || !should_prepare_cooperative_ || !driver_.has_runtime_persistent_api())
        return CUDA_ERROR_NOT_SUPPORTED;
    CUcontext current{};
    const auto context_status = driver_.context_get_current(&current);
    if (context_status != CUDA_SUCCESS) return context_status;
    if (current != context) return CUDA_ERROR_INVALID_CONTEXT;
    if (!supports_qualified_binary(driver_)) return CUDA_ERROR_NOT_SUPPORTED;
    std::shared_ptr<OwnedSgemmRecord> record;
    {
        const std::lock_guard lock(mutex_);
        for (const auto& entry : owned_sgemm_records_) {
            if (entry->context == context && !entry->is_retired)
                return entry->is_prepared.load(std::memory_order_acquire) ? entry->status
                                                                          : CUDA_ERROR_NOT_READY;
        }
        if (owned_sgemm_records_.size() >= kMaxCooperativeSources) return CUDA_ERROR_OUT_OF_MEMORY;
        record = std::make_shared<OwnedSgemmRecord>();
        record->context = context;
        owned_sgemm_records_.push_back(record);
    }
    // The caller's process transition serializes cold preparation and teardown.
    // No SDK operation runs under the metadata mutex above.
    auto result = driver_.library_load_data(&record->library, persistent_sgemm_ptx());
    record->can_unload = result == CUDA_SUCCESS && record->library;
    if (result == CUDA_SUCCESS && !record->library) result = CUDA_ERROR_UNKNOWN;
    CUkernel kernel{};
    if (result == CUDA_SUCCESS)
        result = driver_.library_get_kernel(&kernel, record->library, kPersistentSgemmName);
    if (result == CUDA_SUCCESS && !kernel) result = CUDA_ERROR_UNKNOWN;
    if (result == CUDA_SUCCESS) result = driver_.kernel_get_function(&record->function, kernel);
    if (result == CUDA_SUCCESS && !record->function) result = CUDA_ERROR_UNKNOWN;
    int resident{};
    if (result == CUDA_SUCCESS) result = driver_.occupancy(&resident, record->function, 256, 0);
    if (result == CUDA_SUCCESS && resident <= 0) result = CUDA_ERROR_NOT_SUPPORTED;
    if (result != CUDA_SUCCESS && record->can_unload) {
        record->can_unload = false;
        const auto unloaded = driver_.library_unload(record->library);
        if (unloaded == CUDA_SUCCESS)
            record->library = nullptr;
        else
            result = unloaded;
    }
    record->status = result;
    record->is_prepared.store(true, std::memory_order_release);
    return result;
}

std::optional<RuntimeKernelVariant> PersistentKernelRuntime::runtime_kernel(
    CUcontext context, const void* host_function, const std::string& name,
    const std::string& binary_path, const std::string& tool_path, CUfunction original_function) {
    if (!driver_.has_runtime_persistent_api()) {
        return std::nullopt;
    }
    if (should_prepare_cooperative_) {
        return runtime_cooperative_kernel(context, name, {binary_path, tool_path},
                                          original_function);
    }
    std::shared_ptr<SourceRecord> source;
    std::shared_ptr<RuntimeFunctionRecord> record;
    {
        std::scoped_lock lock(mutex_);
        auto& cached_source = runtime_sources_[binary_path];
        if (cached_source == nullptr) {
            cached_source = std::make_shared<SourceRecord>();
        }
        source = cached_source;
        for (const auto& candidate : runtime_functions_) {
            if (candidate->context == context && candidate->host_function == host_function &&
                candidate->name == name && candidate->binary_path == binary_path) {
                record = candidate;
                break;
            }
        }
        if (record == nullptr) {
            record = std::make_shared<RuntimeFunctionRecord>();
            record->context = context;
            record->host_function = host_function;
            record->name = name;
            record->binary_path = binary_path;
            runtime_functions_.push_back(record);
        }
    }
    std::call_once(source->initialization,
                   [&] { source->images = extract_embedded_ptx(tool_path, binary_path); });
    std::call_once(record->initialization, [&] {
        if (!source->images.has_value()) {
            return;
        }
        std::optional<PersistentPtx> selected;
        for (const auto& image : *source->images) {
            auto transformed = should_prepare_cooperative_
                                   ? transform_persistent_ptx(image, PtxExecutionKind::kPreemptive)
                                   : transform_transparent_ptx(image);
            if (!transformed.has_value()) {
                continue;
            }
            const auto found =
                std::find_if(transformed->kernels.begin(), transformed->kernels.end(),
                             [&](const auto& kernel) { return kernel.original_name == name; });
            if (found != transformed->kernels.end()) {
                if (selected.has_value()) {
                    // Multiple architecture images can have different code.
                    // Choosing one without a validated selector is unsafe.
                    return;
                }
                selected = std::move(transformed);
            }
        }
        if (!selected.has_value()) {
            return;
        }
        const auto found =
            std::find_if(selected->kernels.begin(), selected->kernels.end(),
                         [&](const auto& kernel) { return kernel.original_name == name; });
        // Complete allocating host metadata before acquiring a native library.
        // A throwing call_once initializer can retry; it must not overwrite an
        // already acquired library and lose its ownership on that retry.
        record->variant.parameter_layout = found->parameter_layout;
        record->variant.parameter_count = found->parameter_count;
        record->variant.execution_kind = found->execution_kind;
        if (driver_.library_load_data(&record->library, selected->image.c_str()) != CUDA_SUCCESS) {
            record->library = nullptr;
            return;
        }
        if (driver_.library_get_kernel(&record->variant.kernel, record->library,
                                       found->persistent_name.c_str()) == CUDA_SUCCESS &&
            record->variant.kernel != nullptr) {
            return;
        }
        record->variant.kernel = nullptr;
        if (driver_.library_unload(record->library) != CUDA_SUCCESS) {
            report_diagnostic("[glimmer] private Runtime PTX library cleanup failed\n");
        } else {
            record->library = nullptr;
        }
    });
    return record->variant.kernel == nullptr ? std::nullopt : std::optional{record->variant};
}

bool PersistentKernelRuntime::runtime_cooperative_source_is_prepared(
    CUcontext context, const std::string& binary_path) const {
    if (context == nullptr) {
        return false;
    }
    const std::scoped_lock lock(mutex_);
    return std::any_of(runtime_cooperative_sources_.begin(), runtime_cooperative_sources_.end(),
                       [&](const auto& entry) {
                           return entry->context == context && entry->binary_path == binary_path &&
                                  entry->is_prepared.load(std::memory_order_acquire);
                       });
}

std::optional<RuntimeKernelVariant> PersistentKernelRuntime::runtime_cooperative_kernel(
    CUcontext context, const std::string& name, RuntimeSourceLocation location,
    CUfunction original_function) {
    const auto& binary_path = location.binary_path;
    const auto& tool_path = location.tool_path;
    if (context == nullptr) {
        return std::nullopt;
    }
    std::shared_ptr<SourceRecord> source;
    std::shared_ptr<RuntimeCooperativeSourceRecord> record;
    {
        const std::scoped_lock lock(mutex_);
        for (const auto& candidate : runtime_cooperative_sources_) {
            if (candidate->context == context && candidate->binary_path == binary_path) {
                record = candidate;
                break;
            }
        }
        if (!record) {
            if (runtime_cooperative_sources_.size() >= kMaxCooperativeSources) {
                return std::nullopt;
            }
            record = std::make_shared<RuntimeCooperativeSourceRecord>();
            record->context = context;
            record->binary_path = binary_path;
            runtime_cooperative_sources_.push_back(record);
        }
        auto& cached_source = runtime_sources_[binary_path];
        if (!cached_source) {
            cached_source = std::make_shared<SourceRecord>();
        }
        source = cached_source;
    }
    std::call_once(record->initialization, [&] {
        // Every failure is terminal for this context/source. In particular,
        // throwing after acquisition must never retry a library load/unload.
        try {
            CUcontext current_context = nullptr;
            if (!driver_.has_function_load() ||
                driver_.context_get_current(&current_context) != CUDA_SUCCESS ||
                current_context != context) {
                record->is_prepared.store(true, std::memory_order_release);
                return;
            }
            CUmodule original_module{};
            if (original_function &&
                (!driver_.has_function_get_module() ||
                 driver_.function_get_module(&original_module, original_function) != CUDA_SUCCESS ||
                 !original_module)) {
                record->is_prepared.store(true, std::memory_order_release);
                return;
            }
            std::call_once(source->initialization,
                           [&] { source->images = extract_embedded_ptx(tool_path, binary_path); });
            if (source->images) {
                std::vector<PersistentPtx> images;
                std::unordered_set<std::string> seen;
                std::unordered_set<std::string> ambiguous;
                for (const auto& image : *source->images) {
                    auto transformed =
                        transform_persistent_ptx(image, PtxExecutionKind::kPreemptive);
                    if (transformed) {
                        for (const auto& kernel : transformed->kernels) {
                            if (!seen.insert(kernel.original_name).second) {
                                ambiguous.insert(kernel.original_name);
                            }
                        }
                        images.push_back(std::move(*transformed));
                    }
                }
                for (auto& image : images) {
                    std::erase_if(image.kernels, [&](const auto& kernel) {
                        return ambiguous.contains(kernel.original_name);
                    });
                    for (const auto& kernel : image.kernels) {
                        RuntimeKernelVariant variant;
                        variant.parameter_count = kernel.parameter_count;
                        variant.parameter_layout = kernel.parameter_layout;
                        variant.execution_kind = kernel.execution_kind;
                        record->kernels.emplace(kernel.original_name, std::move(variant));
                    }
                }
                record->libraries.resize(images.size());
                // All persistent host owners/table entries exist before CUDA.
                if (original_function) {
                    for (auto& [entry_name, variant] : record->kernels) {
                        CUfunction actual{};
                        if (driver_.module_get_function(&actual, original_module,
                                                        entry_name.c_str()) == CUDA_SUCCESS &&
                            actual && driver_.function_load(actual) == CUDA_SUCCESS) {
                            variant.original_module = original_module;
                            variant.original_function = actual;
                        }
                    }
                }
                for (std::size_t index = 0; index < images.size(); ++index) {
                    const auto& image = images[index];
                    if (image.kernels.empty() ||
                        (original_function &&
                         std::none_of(image.kernels.begin(), image.kernels.end(),
                                      [&](const auto& kernel) {
                                          return record->kernels.at(kernel.original_name)
                                                     .original_function != nullptr;
                                      }))) {
                        continue;
                    }
                    auto& owner = record->libraries[index];
                    const auto loaded =
                        driver_.library_load_data(&owner.library, image.image.c_str());
                    if (loaded != CUDA_SUCCESS || owner.library == nullptr) {
                        // An error with a published handle is uncertain ownership.
                        continue;
                    }
                    owner.can_unload = true;
                    bool has_ready_kernel = false;
                    for (const auto& kernel : image.kernels) {
                        auto& variant = record->kernels.at(kernel.original_name);
                        if (original_function && !variant.original_function) {
                            continue;
                        }
                        CUkernel native_kernel = nullptr;
                        CUfunction native_function = nullptr;
                        if (driver_.library_get_kernel(&native_kernel, owner.library,
                                                       kernel.persistent_name.c_str()) ==
                                CUDA_SUCCESS &&
                            native_kernel != nullptr &&
                            driver_.kernel_get_function(&native_function, native_kernel) ==
                                CUDA_SUCCESS &&
                            native_function != nullptr &&
                            driver_.function_load(native_function) == CUDA_SUCCESS) {
                            variant.kernel = native_kernel;
                            variant.function = native_function;
                            has_ready_kernel = true;
                        }
                    }
                    if (!has_ready_kernel) {
                        owner.can_unload = false;
                        if (driver_.library_unload(owner.library) == CUDA_SUCCESS) {
                            owner.library = nullptr;
                        } else {
                            report_diagnostic(
                                "[glimmer] cooperative Runtime library cleanup uncertain\n");
                        }
                    }
                }
            }
        } catch (...) {
            // No kernel is issued here. Retain native owners, and never publish
            // a partially prepared image as an available cooperative source.
            record->kernels.clear();
        }
        record->is_prepared.store(true, std::memory_order_release);
    });
    const auto found = record->kernels.find(name);
    if (found == record->kernels.end() || !found->second.function) {
        return std::nullopt;
    }
    if (original_function) {
        CUmodule actual_module{};
        if (found->second.original_function != original_function ||
            !found->second.original_module ||
            driver_.function_get_module(&actual_module, original_function) != CUDA_SUCCESS ||
            actual_module != found->second.original_module) {
            return std::nullopt;
        }
    }
    return found->second;
}

std::optional<CooperativeKernelVariant> PersistentKernelRuntime::cooperative_kernel(
    CUfunction function, std::array<unsigned int, 3> grid, std::array<unsigned int, 3> block,
    unsigned int shared_memory_bytes, CUstream stream, bool per_thread_default_stream) const {
    if (!should_prepare_cooperative_ || shared_memory_bytes != 0) {
        return std::nullopt;
    }
    std::optional<FunctionRecord> variant;
    {
        const std::scoped_lock lock(mutex_);
        const auto found = functions_.find(function);
        if (found != functions_.end() &&
            (!found->second.deferred || !found->second.deferred->is_binary) &&
            found->second.execution_kind == PtxExecutionKind::kPreemptive) {
            variant = found->second;
        }
    }
    if (!variant) {
        if (should_trace_) {
            report_diagnostic("[glimmer] cooperative kernel declined: no prepared variant\n");
        }
        return std::nullopt;
    }
    if (variant->deferred) {
        const auto& owner = variant->deferred;
        if (!owner->is_prepared.load(std::memory_order_acquire) ||
            variant->deferred_index >= owner->functions.size() ||
            !owner->functions[variant->deferred_index]) {
            return std::nullopt;
        }
        variant->persistent = owner->functions[variant->deferred_index];
    }
    const auto shape = validated_launch_shape(grid, block, variant->execution_kind);
    CUcontext context = nullptr;
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    int resident_blocks = 0;
    if (!shape || driver_.context_get_current(&context) != CUDA_SUCCESS ||
        context != variant->context ||
        driver_.stream_capture_status(stream, &capture_status, per_thread_default_stream) !=
            CUDA_SUCCESS ||
        capture_status != CU_STREAM_CAPTURE_STATUS_NONE ||
        driver_.occupancy(&resident_blocks, variant->persistent,
                          static_cast<int>(shape->block_threads), 0) != CUDA_SUCCESS ||
        resident_blocks <= 0) {
        if (should_trace_) {
            report_diagnostic(
                "[glimmer] cooperative kernel declined: native shape/context/capture/occupancy "
                "validation\n");
        }
        return std::nullopt;
    }
    if (should_trace_) {
        trace_persistent_launch(shape->logical_blocks,
                                std::min(worker_blocks_, shape->logical_blocks));
    }
    return CooperativeKernelVariant{context, variant->persistent,
                                    std::move(variant->parameter_layout), *shape,
                                    std::min(worker_blocks_, shape->logical_blocks)};
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): mirror the CUDA launch ABI.
CUresult PersistentKernelRuntime::launch(CUfunction function, unsigned int grid_dim_x,
                                         unsigned int grid_dim_y, unsigned int grid_dim_z,
                                         unsigned int block_dim_x, unsigned int block_dim_y,
                                         unsigned int block_dim_z, unsigned int shared_memory_bytes,
                                         CUstream stream, void** kernel_parameters, void** extra,
                                         bool per_thread_default_stream) {
    const auto result = try_launch(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                   block_dim_y, block_dim_z, shared_memory_bytes, stream,
                                   kernel_parameters, extra, per_thread_default_stream);
    if (result.has_value()) {
        return *result;
    }
    return per_thread_default_stream
               ? driver_.launch_kernel_ptsz(function, grid_dim_x, grid_dim_y, grid_dim_z,
                                            block_dim_x, block_dim_y, block_dim_z,
                                            shared_memory_bytes, stream, kernel_parameters, extra)
               : driver_.launch_kernel(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                       block_dim_y, block_dim_z, shared_memory_bytes, stream,
                                       kernel_parameters, extra);
}

std::optional<CUresult> PersistentKernelRuntime::try_launch(
    CUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_memory_bytes, CUstream stream, void** kernel_parameters, void** extra,
    bool per_thread_default_stream) {
    const auto invoke = [&](CUfunction selected, unsigned int blocks, void** arguments) {
        return per_thread_default_stream
                   ? driver_.launch_kernel_ptsz(selected, blocks, 1, 1, block_dim_x, block_dim_y,
                                                block_dim_z, shared_memory_bytes, stream, arguments,
                                                extra)
                   : driver_.launch_kernel(selected, blocks, 1, 1, block_dim_x, block_dim_y,
                                           block_dim_z, shared_memory_bytes, stream, arguments,
                                           extra);
    };
    std::optional<FunctionRecord> variant;
    if (worker_blocks_ > 0 && shared_memory_bytes == 0 && extra == nullptr) {
        std::scoped_lock lock(mutex_);
        const auto found = functions_.find(function);
        if (found != functions_.end() &&
            (!found->second.deferred || !found->second.deferred->is_binary) &&
            found->second.execution_kind != PtxExecutionKind::kPreemptive) {
            variant = found->second;
        }
    }
    std::vector<void*> arguments;
    std::optional<PersistentLaunchShape> shape;
    if (variant.has_value()) {
        try {
            CUcontext context = nullptr;
            int resident_blocks = 0;
            CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
            shape = validated_launch_shape({grid_dim_x, grid_dim_y, grid_dim_z},
                                           {block_dim_x, block_dim_y, block_dim_z},
                                           variant->execution_kind);
            if (!shape.has_value() ||
                (variant->parameter_count > 0 && kernel_parameters == nullptr) ||
                driver_.context_get_current(&context) != CUDA_SUCCESS ||
                context != variant->context ||
                driver_.stream_capture_status(stream, &capture_status, per_thread_default_stream) !=
                    CUDA_SUCCESS ||
                capture_status != CU_STREAM_CAPTURE_STATUS_NONE ||
                driver_.occupancy(&resident_blocks, variant->persistent,
                                  static_cast<int>(shape->block_threads), 0) != CUDA_SUCCESS ||
                resident_blocks <= 0) {
                variant.reset();
            } else {
                arguments.reserve(variant->parameter_count + 3);
                for (std::size_t index = 0; index < variant->parameter_count; ++index) {
                    arguments.push_back(kernel_parameters[index]);
                }
                arguments.push_back(&grid_dim_x);
                if (variant->execution_kind == PtxExecutionKind::kBlockStatic) {
                    arguments.push_back(&grid_dim_y);
                    arguments.push_back(&grid_dim_z);
                }
            }
        } catch (...) {
            variant.reset();
        }
    }
    if (!variant.has_value()) {
        return std::nullopt;
    }
    // Never retry a real launch: its error could belong to earlier async work.
    const auto result = invoke(variant->persistent, worker_blocks_, arguments.data());
    if (result == CUDA_SUCCESS && should_trace_) {
        trace_persistent_launch(shape->logical_blocks, worker_blocks_);
    }
    return result;
}

// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace glimmer::interceptor
