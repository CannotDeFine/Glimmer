#include "driver_frontend_state.h"

namespace {
cublasStatus_t blas_error(CUresult error) noexcept {
    switch (error) {
        case CUDA_SUCCESS:
            return CUBLAS_STATUS_SUCCESS;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return CUBLAS_STATUS_ALLOC_FAILED;
        case CUDA_ERROR_INVALID_VALUE:
            return CUBLAS_STATUS_INVALID_VALUE;
        case CUDA_ERROR_NOT_SUPPORTED:
            return CUBLAS_STATUS_NOT_SUPPORTED;
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return CUBLAS_STATUS_NOT_INITIALIZED;
        default:
            return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}
cublasStatus_t blas_status(const std::optional<ExecutionWireReply>& reply,
                           bool allows_replacement = false) noexcept {
    if (!reply) return CUBLAS_STATUS_INTERNAL_ERROR;
    if (reply->disposition == ExecutionWireDisposition::kComputeReplacement) {
        if (!allows_replacement || reply->original_status || !reply->id)
            return CUBLAS_STATUS_INTERNAL_ERROR;
        if (reply->status == CUDA_ERROR_LAUNCH_FAILED ||
            reply->status == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES ||
            reply->status == CUDA_ERROR_LAUNCH_TIMEOUT)
            return CUBLAS_STATUS_EXECUTION_FAILED;
        return blas_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition == ExecutionWireDisposition::kComplete && reply->original_status &&
        reply->status == *reply->original_status) {
        const auto value = *reply->original_status;
        return value >= 0 && value <= 31 ? static_cast<cublasStatus_t>(value)
                                         : CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return blas_error(static_cast<CUresult>(reply->status));
}
template <typename Function>
cublasStatus_t blas_boundary(Function&& function) noexcept {
    if (::getpid() != kProcess) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (is_entered) return CUBLAS_STATUS_NOT_SUPPORTED;
    Entered entered;
    try {
        auto& shared = state();
        const auto entered_at = shared.rpc_metrics
                                    ? shared.rpc_metrics->entry()
                                    : glimmer::interceptor::execution::RpcMetrics::Stamp{};
        Exclusion exclusion(shared.mutex);
        glimmer::interceptor::execution::RpcMetrics::Scope boundary_observation(
            shared.rpc_metrics.get(), exclusion, entered_at, true);
        if (!shared.config) return CUBLAS_STATUS_NOT_INITIALIZED;
        return function(shared, exclusion);
    } catch (const std::bad_alloc&) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}
template <typename Function>
cublasStatus_t blas_handle(cublasHandle_t handle, bool is_mutating, Function&& function,
                           bool is_command = false, Kind kind = Kind::kBlas,
                           bool allows_blas_alias = false) noexcept {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.find(handle, kind);
        if (!entry && allows_blas_alias) entry = shared.find(handle, Kind::kBlas);
        auto* context = shared.context();
        if (!entry || !context || entry->context != context || context->is_closing ||
            context->is_uncertain || entry->is_uncertain)
            return CUBLAS_STATUS_NOT_INITIALIZED;
        if (context->active_calls >= kMaxHandles - 1 ||
            ((is_mutating || is_command) && context->active_mutations == kMaxHandles) ||
            entry->active_calls == kMaxHandles)
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto epoch = context->epoch;
        // Read-only queries must drain after context sealing, not prevent
        // sealing indefinitely when another thread keeps polling metadata.
        std::optional<CounterPin> work;
        if (is_mutating || is_command) work.emplace(context->active_mutations, exclusion);
        CounterPin origin(context->active_calls, exclusion);
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && context->epoch == epoch;
        };
        while (entry->is_retiring) {
            if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        std::optional<FlagPin> mutation;
        if (is_mutating) {
            mutation.emplace(entry->is_retiring, exclusion);
            while (entry->active_calls) {
                if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
            }
        }
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
std::optional<ExecutionWireReply> query_blas(State& shared, Exclusion& exclusion, Entry& entry,
                                             ExecutionBlasAttribute attribute) {
    ExecutionPayloadWriter input;
    if (!input.write_u64(entry.remote) || !input.write_u32(static_cast<std::uint32_t>(attribute)))
        return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
    return shared.call(exclusion, *entry.context, ExecutionOpcode::kGetBlasAttribute,
                       std::move(input));
}
cublasStatus_t set_blas_value(cublasHandle_t handle, ExecutionBlasAttribute attribute,
                              std::uint64_t value) noexcept {
    return blas_handle(handle, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote) ||
            !input.write_u32(static_cast<std::uint32_t>(attribute)) || !input.write_u64(value))
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kSetBlasAttribute, std::move(input));
        const auto result = blas_status(reply);
        if (result == CUBLAS_STATUS_SUCCESS && attribute == ExecutionBlasAttribute::kPointerMode)
            entry.blas_pointer_mode = static_cast<unsigned int>(value);
        return result;
    });
}
template <typename Value>
cublasStatus_t get_blas_value(cublasHandle_t handle, ExecutionBlasAttribute attribute,
                              Value* output) noexcept {
    static thread_local off_t hint{-1};
    return blas_handle(handle, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!RpcMetrics::Scope::host_range(
                shared.rpc_metrics.get(), exclusion, RpcMetrics::HostRangeCaller::kBlasValue,
                [&](RpcMetrics::HostTrace* trace) noexcept {
                    return writable_host_range(output, sizeof(*output), trace,
                                               shared.should_seek_host_range ? &hint : nullptr);
                }))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto reply = query_blas(shared, exclusion, entry, attribute);
        const auto result = blas_status(reply);
        if (result != CUBLAS_STATUS_SUCCESS) return result;
        const auto maximum = (attribute == ExecutionBlasAttribute::kPointerMode ||
                              attribute == ExecutionBlasAttribute::kAtomicsMode)
                                 ? std::uint64_t{1}
                             : attribute == ExecutionBlasAttribute::kMathMode
                                 ? std::uint64_t{31}
                                 : static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        if (reply->address > maximum || (attribute == ExecutionBlasAttribute::kPointerMode &&
                                         reply->address != entry.blas_pointer_mode))
            return CUBLAS_STATUS_INTERNAL_ERROR;
        const auto value = static_cast<Value>(reply->address);
        return publish(output, value) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    });
}
}  // namespace
namespace {
CUresult open_lt_metadata(State& shared, Exclusion& exclusion) {
    if (!shared.lt_metadata_context) {
        auto* context = shared.allocate_entry(Kind::kLtMetadataContext, nullptr);
        if (!context) return CUDA_ERROR_OUT_OF_MEMORY;
        context->context = context;
        context->is_alive = true;
        shared.lt_metadata_context = context;
    }
    return shared.open_context(exclusion, *shared.lt_metadata_context, 0);
}
cublasStatus_t lt_status(const std::optional<ExecutionWireReply>& reply, Entry& context,
                         LtEntry* entry = nullptr) noexcept {
    if (!reply || reply->parameter_bytes > 1 || reply->parameter_bytes == 1) {
        context.is_uncertain = true;
        if (entry) entry->is_uncertain = true;
        if (!reply || reply->parameter_bytes > 1) return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    return blas_status(reply);
}
struct LtMetadataCreate {
    ExecutionLtObjectKind kind{};
    std::uint32_t type{CUDA_R_32F}, compute{CUBLAS_COMPUTE_32F}, scale{CUDA_R_32F};
    std::uint64_t rows{}, columns{};
    std::int64_t leading{};
};
template <typename Handle>
cublasStatus_t create_lt_metadata(Handle* output, const LtMetadataCreate& specification) noexcept {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(std::uintptr_t)))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto opened = open_lt_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return blas_error(opened);
        auto* context = shared.lt_metadata_context;
        if (!shared.has_lt_tokens || shared.lt_entries.size() == kMaxHandles ||
            context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CUBLAS_STATUS_ALLOC_FAILED;
        auto entry = std::make_shared<LtEntry>();
        entry->context = context;
        entry->kind = specification.kind;
        entry->epoch = context->epoch;
        entry->token = shared.next_lt_token;
        entry->scale_type = specification.scale;
        if (shared.next_lt_token == UINTPTR_MAX)
            shared.has_lt_tokens = false;
        else
            ++shared.next_lt_token;
        ExecutionPayloadWriter input;
        if (!input.write_u32(static_cast<std::uint32_t>(specification.kind)) ||
            !input.write_u32(specification.type) || !input.write_u32(specification.compute) ||
            !input.write_u32(specification.scale) || !input.write_u64(specification.rows) ||
            !input.write_u64(specification.columns) ||
            !input.write_u64(std::bit_cast<std::uint64_t>(specification.leading)))
            return CUBLAS_STATUS_ALLOC_FAILED;
        // Occupy bounded capacity before releasing exclusion. Unpublished or
        // uncertain native ownership is never replayed or silently reclaimed.
        shared.lt_entries.push_back(entry);
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateLtMetadata, std::move(input));
        const auto result = lt_status(reply, *context, entry.get());
        if (result != CUBLAS_STATUS_SUCCESS) {
            if (!context->is_uncertain && reply && !reply->id &&
                ((reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->status == *reply->original_status) ||
                 (!reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kRejected &&
                  reply->status != CUDA_SUCCESS)))
                std::erase(shared.lt_entries, entry);
            else
                context->is_uncertain = entry->is_uncertain = true;
            return result;
        }
        if (!reply->id) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        // Public opaque identity only: no pointer arithmetic or dereference.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto handle = reinterpret_cast<Handle>(static_cast<std::uintptr_t>(entry->token));
        if (!publish(output, handle)) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_SUCCESS;
    });
}
template <typename Handle, typename Function>
cublasStatus_t lt_metadata(Handle handle, ExecutionLtObjectKind kind, bool is_mutating,
                           Function&& function) noexcept {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        const auto token = reinterpret_cast<std::uintptr_t>(handle);
        const auto found = std::ranges::find_if(shared.lt_entries, [=](const auto& value) {
            return value->token == token && value->kind == kind && value->is_alive;
        });
        if (found == shared.lt_entries.end()) return CUBLAS_STATUS_NOT_INITIALIZED;
        // The strong record keeps counters/retirement pins valid even after
        // successful destruction removes the record from the public registry.
        const auto entry = *found;
        auto* context = entry->context;
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && !context->is_closing && context->epoch == entry->epoch;
        };
        if (!valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        if (entry->active_calls == kMaxHandles || context->active_calls >= kMaxHandles - 1 ||
            (is_mutating && context->active_mutations >= kMaxHandles - 1))
            return CUBLAS_STATUS_ALLOC_FAILED;
        CounterPin origin(context->active_calls, exclusion);
        std::optional<CounterPin> work;
        if (is_mutating) work.emplace(context->active_mutations, exclusion);
        while (entry->is_retiring)
            if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        std::optional<FlagPin> retiring;
        if (is_mutating) {
            retiring.emplace(entry->is_retiring, exclusion);
            while (entry->active_calls)
                if (!yield_frontend(exclusion) || !valid()) return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
template <typename Handle>
cublasStatus_t destroy_lt_metadata(Handle handle, ExecutionLtObjectKind kind) noexcept {
    return lt_metadata(handle, kind, true, [](State& shared, Exclusion& exclusion, LtEntry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kDestroyLtMetadata, std::move(input));
        const auto result = lt_status(reply, *entry.context, &entry);
        if (result == CUBLAS_STATUS_SUCCESS) {
            entry.is_alive = false;
            std::erase_if(shared.lt_entries,
                          [&](const auto& value) { return value.get() == &entry; });
        }
        return result;
    });
}
std::optional<std::size_t> lt_attribute_bytes(ExecutionLtObjectKind kind,
                                              std::uint32_t attribute) noexcept {
    if (kind == ExecutionLtObjectKind::kLayout && attribute <= CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET)
        return attribute == CUBLASLT_MATRIX_LAYOUT_TYPE ||
                       attribute == CUBLASLT_MATRIX_LAYOUT_ORDER ||
                       attribute == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
                   ? 4
                   : 8;
    if (kind == ExecutionLtObjectKind::kOperation) {
        if (attribute == CUBLASLT_MATMUL_DESC_FAST_ACCUM) return 1;
        if (attribute <= CUBLASLT_MATMUL_DESC_EPILOGUE ||
            attribute == CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET ||
            attribute == CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE)
            return 4;
        if (attribute == CUBLASLT_MATMUL_DESC_BIAS_POINTER ||
            attribute == CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE ||
            attribute == CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE)
            return 8;
    }
    if (kind == ExecutionLtObjectKind::kPreference) {
        if (attribute == CUBLASLT_MATMUL_PREF_SEARCH_MODE ||
            attribute == CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK ||
            (attribute >= CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES &&
             attribute <= CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES) ||
            attribute == CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT)
            return 4;
        if (attribute == CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES ||
            attribute == CUBLASLT_MATMUL_PREF_IMPL_MASK)
            return 8;
    }
    return std::nullopt;
}
template <typename Handle>
cublasStatus_t set_lt_attribute(Handle handle, ExecutionLtObjectKind kind, std::uint32_t attribute,
                                const void* buffer, std::size_t bytes) noexcept {
    return lt_metadata(
        handle, kind, true, [&](State& shared, Exclusion& exclusion, LtEntry& entry) {
            const auto width = lt_attribute_bytes(kind, attribute);
            if (!width) return CUBLAS_STATUS_NOT_SUPPORTED;
            if (bytes != *width) return CUBLAS_STATUS_INVALID_VALUE;
            std::vector<std::byte> data(bytes);
            if (!self_copy(data.data(), buffer, bytes, false)) return CUBLAS_STATUS_INVALID_VALUE;
            Entry* allocation{};
            std::uint64_t pointer{};
            std::optional<CounterPin> allocation_origin;
            const bool is_pointer = kind == ExecutionLtObjectKind::kOperation &&
                                    attribute == CUBLASLT_MATMUL_DESC_BIAS_POINTER;
            if (is_pointer) {
                if (bytes != sizeof(pointer)) return CUBLAS_STATUS_INVALID_VALUE;
                std::memcpy(&pointer, data.data(), sizeof(pointer));
                if (pointer) {
                    allocation = shared.range(pointer, 1);
                    if (!allocation || allocation->is_retiring || allocation->context->is_closing ||
                        allocation->context->is_uncertain)
                        return CUBLAS_STATUS_INVALID_VALUE;
                    if (allocation->context->active_calls == kMaxHandles)
                        return CUBLAS_STATUS_ALLOC_FAILED;
                    allocation_origin.emplace(allocation->context->active_calls, exclusion);
                }
                std::ranges::fill(data, std::byte{0});
            }
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !input.write_u32(attribute) ||
                !input.write_blob(data) || !input.write_u32(allocation ? 1 : 0))
                return CUBLAS_STATUS_ALLOC_FAILED;
            if (allocation && (!input.write_u64(allocation->context->connection->session()) ||
                               !input.write_u64(allocation->remote) ||
                               !input.write_u64(pointer - allocation->address)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kSetLtAttribute, std::move(input));
            const auto result = lt_status(reply, *entry.context, &entry);
            if (result == CUBLAS_STATUS_SUCCESS && kind == ExecutionLtObjectKind::kOperation &&
                bytes == 4) {
                if (attribute == CUBLASLT_MATMUL_DESC_SCALE_TYPE)
                    std::memcpy(&entry.scale_type, data.data(), 4);
                if (attribute == CUBLASLT_MATMUL_DESC_POINTER_MODE)
                    std::memcpy(&entry.pointer_mode, data.data(), 4);
            }
            return result;
        });
}
template <typename Handle>
cublasStatus_t get_lt_attribute(Handle handle, ExecutionLtObjectKind kind, std::uint32_t attribute,
                                void* buffer, std::size_t bytes, std::size_t* written) noexcept {
    return lt_metadata(
        handle, kind, false, [&](State& shared, Exclusion& exclusion, LtEntry& entry) {
            if (bytes > 4096 || (!bytes && !written) ||
                (bytes && !writable_host_range(buffer, bytes)) ||
                (written && !writable_host_range(written, sizeof(*written))))
                return CUBLAS_STATUS_INVALID_VALUE;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) || !input.write_u32(attribute) ||
                !input.write_u64(bytes))
                return CUBLAS_STATUS_ALLOC_FAILED;
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kGetLtAttribute, std::move(input));
            const auto result = lt_status(reply, *entry.context, &entry);
            if (result != CUBLAS_STATUS_SUCCESS) return result;
            if (!reply || reply->address > 4096 ||
                (bytes ? reply->data.size() != reply->address || reply->address > bytes
                       : !reply->data.empty())) {
                entry.context->is_uncertain = entry.is_uncertain = true;
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }
            const auto count = static_cast<std::size_t>(reply->address);
            if ((bytes && !self_copy(buffer, reply->data.data(), count, true)) ||
                (written && !publish(written, count)))
                return CUBLAS_STATUS_INVALID_VALUE;
            return CUBLAS_STATUS_SUCCESS;
        });
}
std::size_t lt_version(bool runtime) noexcept {
    std::size_t value{};
    const auto result = blas_boundary([&](State& shared, Exclusion& exclusion) {
        const auto opened = open_lt_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return blas_error(opened);
        ExecutionPayloadWriter input;
        if (!input.write_u32(runtime ? 1 : 0)) return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *shared.lt_metadata_context,
                                       ExecutionOpcode::kLtVersion, std::move(input));
        const auto status = lt_status(reply, *shared.lt_metadata_context);
        if (status == CUBLAS_STATUS_SUCCESS) {
            if (!reply || reply->original_status || reply->address > SIZE_MAX)
                return CUBLAS_STATUS_INTERNAL_ERROR;
            value = static_cast<std::size_t>(reply->address);
        }
        return status;
    });
    return result == CUBLAS_STATUS_SUCCESS ? value : 0;
}
struct LtDescriptorArguments {
    void* operation{};
    std::array<void*, 4> matrices{};
    void* preference{};
};
template <typename Function>
cublasStatus_t with_lt_descriptors(State& shared, Exclusion& exclusion, Entry& handle,
                                   const LtDescriptorArguments& arguments, Function&& function) {
    std::array<std::shared_ptr<LtEntry>, 6> entries;
    const std::array<void*, 6> handles{arguments.operation,   arguments.matrices[0],
                                       arguments.matrices[1], arguments.matrices[2],
                                       arguments.matrices[3], arguments.preference};
    auto* metadata = shared.lt_metadata_context;
    if (!metadata || !metadata->connection || metadata->is_uncertain || metadata->is_closing)
        return CUBLAS_STATUS_NOT_INITIALIZED;
    for (;;) {
        bool waiting{};
        for (std::size_t index = 0; index < entries.size(); ++index) {
            if ((index == 3 || index == 5) && !handles[index]) continue;
            const auto kind = index == 0   ? ExecutionLtObjectKind::kOperation
                              : index == 5 ? ExecutionLtObjectKind::kPreference
                                           : ExecutionLtObjectKind::kLayout;
            const auto token = reinterpret_cast<std::uintptr_t>(handles[index]);
            const auto found = std::ranges::find_if(shared.lt_entries, [&](const auto& entry) {
                return entry->token == token && entry->kind == kind && entry->is_alive;
            });
            if (found == shared.lt_entries.end()) return CUBLAS_STATUS_NOT_INITIALIZED;
            entries[index] = *found;
            const auto& entry = **found;
            if (entry.context != metadata || entry.epoch != metadata->epoch || entry.is_uncertain)
                return CUBLAS_STATUS_NOT_INITIALIZED;
            if (entry.active_calls > kMaxHandles - entries.size())
                return CUBLAS_STATUS_ALLOC_FAILED;
            waiting |= entry.is_retiring;
        }
        if (!waiting) break;
        if (!yield_frontend(exclusion) || handle.context->is_uncertain || metadata->is_uncertain ||
            metadata->is_closing)
            return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (metadata->active_calls >= kMaxHandles - 1) return CUBLAS_STATUS_ALLOC_FAILED;
    CounterPin origin(metadata->active_calls, exclusion);
    std::array<std::optional<CounterPin>, 6> pins;
    for (std::size_t index = 0; index < entries.size(); ++index)
        if (entries[index]) pins[index].emplace(entries[index]->active_calls, exclusion);
    ExecutionPayloadWriter input;
    if (!input.write_u64(handle.remote) || !input.write_u64(metadata->connection->session()))
        return CUBLAS_STATUS_ALLOC_FAILED;
    for (std::size_t index = 0; index < 5; ++index)
        if (!input.write_u64(entries[index] ? entries[index]->remote : 0))
            return CUBLAS_STATUS_ALLOC_FAILED;
    return function(entries, input);
}
cublasStatus_t query_lt_algorithms(cublasLtHandle_t handle, const LtDescriptorArguments& arguments,
                                   int requested, const cublasLtMatmulAlgo_t* algorithm,
                                   cublasLtMatmulHeuristicResult_t* output,
                                   int* returned) noexcept {
    return blas_handle(
        reinterpret_cast<cublasHandle_t>(handle), false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            const bool is_check = algorithm != nullptr;
            if (requested <= 0 || !output ||
                (!is_check && !writable_host_range(returned, sizeof(*returned))))
                return CUBLAS_STATUS_INVALID_VALUE;
            if (requested > 64) return CUBLAS_STATUS_NOT_SUPPORTED;
            if (!writable_host_range(output, sizeof(*output) * static_cast<std::size_t>(requested)))
                return CUBLAS_STATUS_INVALID_VALUE;
            cublasLtMatmulAlgo_t copied{};
            if (is_check && !self_copy(&copied, algorithm, sizeof(copied), false))
                return CUBLAS_STATUS_INVALID_VALUE;
            return with_lt_descriptors(
                shared, exclusion, entry, arguments,
                [&](const auto& descriptors, ExecutionPayloadWriter& input) {
                    if (is_check) {
                        if (!input.write_blob(std::as_bytes(std::span{&copied, 1})))
                            return CUBLAS_STATUS_ALLOC_FAILED;
                    } else if (!descriptors[5] || !input.write_u64(descriptors[5]->remote) ||
                               !input.write_u32(static_cast<std::uint32_t>(requested)))
                        return CUBLAS_STATUS_INVALID_VALUE;
                    const auto reply = shared.call(exclusion, *entry.context,
                                                   is_check ? ExecutionOpcode::kLtAlgorithmCheck
                                                            : ExecutionOpcode::kLtHeuristic,
                                                   std::move(input));
                    const auto status = lt_status(reply, *entry.context);
                    if (status != CUBLAS_STATUS_SUCCESS) return status;
                    const auto invalid = [&] {
                        entry.is_uncertain = entry.context->is_uncertain = true;
                        return CUBLAS_STATUS_INTERNAL_ERROR;
                    };
                    if (!reply || !reply->address) return invalid();
                    ExecutionPayloadReader result(reply->data);
                    std::uint32_t count{};
                    if (!result.read_u32(count) || count > static_cast<std::uint32_t>(requested) ||
                        (is_check && count != 1))
                        return invalid();
                    std::vector<cublasLtMatmulHeuristicResult_t> values(count);
                    for (auto& value : values) {
                        std::span<const std::byte> bytes;
                        std::uint64_t workspace{};
                        std::uint32_t state{}, waves{};
                        if (!result.read_blob(bytes) || bytes.size() != sizeof(value.algo) ||
                            !result.read_u64(workspace) || workspace > SIZE_MAX ||
                            !result.read_u32(state) || !result.read_u32(waves))
                            return invalid();
                        std::memcpy(&value.algo, bytes.data(), bytes.size());
                        value.workspaceSize = static_cast<std::size_t>(workspace);
                        value.state = static_cast<cublasStatus_t>(state);
                        value.wavesCount = std::bit_cast<float>(waves);
                        for (auto& reserved : value.reserved) {
                            std::uint32_t bits{};
                            if (!result.read_u32(bits)) return invalid();
                            reserved = std::bit_cast<std::int32_t>(bits);
                        }
                    }
                    if (!result.empty()) return invalid();
                    if (is_check) {
                        // AlgoCheck never updates result->algo, including when
                        // its input algorithm is a different nonaliasing object.
                        constexpr auto offset =
                            offsetof(cublasLtMatmulHeuristicResult_t, workspaceSize);
                        if (!self_copy(reinterpret_cast<std::byte*>(output) + offset,
                                       reinterpret_cast<const std::byte*>(values.data()) + offset,
                                       sizeof(*output) - offset, true))
                            return CUBLAS_STATUS_INVALID_VALUE;
                    } else if (!self_copy(output, values.data(), values.size() * sizeof(*output),
                                          true) ||
                               !publish(returned, static_cast<int>(count)))
                        return CUBLAS_STATUS_INVALID_VALUE;
                    return CUBLAS_STATUS_SUCCESS;
                });
        },
        false, Kind::kLt, true);
}
}  // namespace
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact installed SDK ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* output,
                                                                  cudaDataType type,
                                                                  std::uint64_t rows,
                                                                  std::uint64_t columns,
                                                                  std::int64_t leading) {
    return create_lt_metadata(output,
                              {ExecutionLtObjectKind::kLayout, static_cast<std::uint32_t>(type),
                               CUBLAS_COMPUTE_32F, CUDA_R_32F, rows, columns, leading});
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* output,
                                                                cublasComputeType_t compute,
                                                                cudaDataType scale) {
    return create_lt_metadata(
        output, {ExecutionLtObjectKind::kOperation, CUDA_R_32F, static_cast<std::uint32_t>(compute),
                 static_cast<std::uint32_t>(scale), 0, 0, 0});
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* output) {
    return create_lt_metadata(output, {ExecutionLtObjectKind::kPreference});
}
#define LT_ATTRIBUTES(PREFIX, HANDLE, ATTRIBUTE, KIND)                                          \
    extern "C" cublasStatus_t CUBLASWINAPI PREFIX##Destroy(HANDLE handle) {                     \
        return destroy_lt_metadata(handle, ExecutionLtObjectKind::KIND);                        \
    }                                                                                           \
    extern "C" cublasStatus_t CUBLASWINAPI PREFIX##SetAttribute(                                \
        HANDLE handle, ATTRIBUTE attribute, const void* buffer, std::size_t bytes) {            \
        return set_lt_attribute(handle, ExecutionLtObjectKind::KIND,                            \
                                static_cast<std::uint32_t>(attribute), buffer, bytes);          \
    }                                                                                           \
    extern "C" cublasStatus_t CUBLASWINAPI PREFIX##GetAttribute(                                \
        HANDLE handle, ATTRIBUTE attribute, void* buffer, std::size_t bytes,                    \
        std::size_t* written) {                                                                 \
        return get_lt_attribute(handle, ExecutionLtObjectKind::KIND,                            \
                                static_cast<std::uint32_t>(attribute), buffer, bytes, written); \
    }
LT_ATTRIBUTES(cublasLtMatrixLayout, cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t,
              kLayout)
LT_ATTRIBUTES(cublasLtMatmulDesc, cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, kOperation)
LT_ATTRIBUTES(cublasLtMatmulPreference, cublasLtMatmulPreference_t,
              cublasLtMatmulPreferenceAttributes_t, kPreference)
#undef LT_ATTRIBUTES
extern "C" std::size_t CUBLASWINAPI cublasLtGetVersion() {
    return lt_version(false);
}
extern "C" std::size_t CUBLASWINAPI cublasLtGetCudartVersion() {
    return lt_version(true);
}
// These entry points must not reach the caller's SDK with owned opaque
// descriptors. GPU execution and caller-allocated descriptor storage need
// their own authenticated contracts, rather than a native fallback.
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed Lt ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* output) {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(cublasLtHandle_t)))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return blas_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CUBLAS_STATUS_NOT_INITIALIZED;
        auto* entry = shared.allocate_entry(Kind::kLt, context);
        if (!entry) return CUBLAS_STATUS_ALLOC_FAILED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateLt, ExecutionPayloadWriter{});
        const auto result = lt_status(reply, *context);
        if (result != CUBLAS_STATUS_SUCCESS || context->is_uncertain) {
            if (!context->is_uncertain && reply && !reply->id &&
                ((reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->status == *reply->original_status) ||
                 (!reply->original_status &&
                  reply->disposition == ExecutionWireDisposition::kRejected && reply->status)))
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            else
                context->is_uncertain = entry->is_uncertain = true;
            return result == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_INTERNAL_ERROR : result;
        }
        if (!reply->id) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        if (!publish(output, reinterpret_cast<cublasLtHandle_t>(entry))) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_SUCCESS;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t handle) {
    return blas_handle(
        reinterpret_cast<cublasHandle_t>(handle), true,
        [](State& shared, Exclusion& exclusion, Entry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote)) return CUBLAS_STATUS_ALLOC_FAILED;
            const auto reply = shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyLt,
                                           std::move(input));
            const auto result = lt_status(reply, *entry.context);
            if (result == CUBLAS_STATUS_SUCCESS) entry.is_alive = false;
            if (entry.context->is_uncertain) entry.is_uncertain = true;
            return result;
        },
        false, Kind::kLt);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    cublasLtMatmulPreference_t preference, int requested, cublasLtMatmulHeuristicResult_t* output,
    int* returned) {
    return query_lt_algorithms(handle, {operation, {a, b, c, d}, preference}, requested, nullptr,
                               output, returned);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
    const cublasLtMatmulAlgo_t* algorithm, cublasLtMatmulHeuristicResult_t* output) {
    if (!algorithm) return CUBLAS_STATUS_INVALID_VALUE;
    return query_lt_algorithms(handle, {operation, {a, b, c, d}, nullptr}, 1, algorithm, output,
                               nullptr);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCapGetAttribute(
    const cublasLtMatmulAlgo_t* algorithm, cublasLtMatmulAlgoCapAttributes_t attribute,
    void* buffer, std::size_t bytes, std::size_t* written) {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (bytes > 4096) return CUBLAS_STATUS_NOT_SUPPORTED;
        cublasLtMatmulAlgo_t copied{};
        std::size_t initial_written{};
        if (!self_copy(&copied, algorithm, sizeof(copied), false) ||
            (bytes && buffer && !writable_host_range(buffer, bytes)) ||
            (written && (!writable(written) ||
                         !self_copy(&initial_written, written, sizeof(initial_written), false))))
            return CUBLAS_STATUS_INVALID_VALUE;
        if (bytes && buffer && written) {
            const auto left = reinterpret_cast<std::uintptr_t>(buffer);
            const auto right = reinterpret_cast<std::uintptr_t>(written);
            if (left <= right ? right - left < bytes : left - right < sizeof(*written))
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
        std::vector<std::byte> initial(buffer ? bytes : 0);
        if (!initial.empty() && !self_copy(initial.data(), buffer, bytes, false))
            return CUBLAS_STATUS_INVALID_VALUE;
        if (shared.lt_metadata_context &&
            (shared.lt_metadata_context->is_uncertain || shared.lt_metadata_context->is_closing))
            return CUBLAS_STATUS_NOT_INITIALIZED;
        const auto opened = open_lt_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return blas_error(opened);
        auto& context = *shared.lt_metadata_context;
        if (context.is_uncertain || context.is_closing) return CUBLAS_STATUS_NOT_INITIALIZED;
        if (context.active_calls >= kMaxHandles - 1) return CUBLAS_STATUS_ALLOC_FAILED;
        CounterPin use(context.active_calls, exclusion);
        const auto presence = (buffer ? 1U : 0U) | (written ? 2U : 0U);
        ExecutionPayloadWriter input;
        if (!input.write_blob(std::as_bytes(std::span{&copied, 1})) ||
            !input.write_u32(static_cast<std::uint32_t>(attribute)) || !input.write_u64(bytes) ||
            !input.write_u32(presence) || !input.write_blob(initial) ||
            !input.write_u64(initial_written))
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, context, ExecutionOpcode::kLtAlgorithmCapability,
                                       std::move(input));
        // Validate the status domain before conversion to an SDK enum and
        // before publishing output, including native error-side effects.
        if (reply && ((reply->original_status &&
                       (*reply->original_status < 0 || *reply->original_status > 31)) ||
                      (!reply->original_status && (reply->status < 0 || reply->status > 999)))) {
            context.is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        const auto result = lt_status(reply, context);
        if (!reply || context.is_uncertain)
            return result == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_INTERNAL_ERROR : result;
        const auto invalid = [&] {
            context.is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        };
        if (reply->disposition != ExecutionWireDisposition::kComplete) {
            if (reply->id || reply->address || !reply->data.empty() || reply->original_status ||
                result == CUBLAS_STATUS_SUCCESS)
                return invalid();
            return result;
        }
        if (!reply->original_status || reply->status != *reply->original_status || reply->id ||
            !reply->arguments.empty() || reply->address > SIZE_MAX ||
            reply->data.size() != initial.size() ||
            (!written && reply->address != initial_written) ||
            (result == CUBLAS_STATUS_SUCCESS && written && bytes && reply->address > bytes))
            return invalid();
        const auto count = static_cast<std::size_t>(reply->address);
        if ((!initial.empty() && !self_copy(buffer, reply->data.data(), initial.size(), true)) ||
            (written && !publish(written, count)))
            return CUBLAS_STATUS_INVALID_VALUE;
        return result;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasLtMatrixTransform(cublasLtHandle_t, cublasLtMatrixTransformDesc_t, const void*, const void*,
                        cublasLtMatrixLayout_t, const void*, const void*, cublasLtMatrixLayout_t,
                        void*, cublasLtMatrixLayout_t, cudaStream_t) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutInit_internal(
    cublasLtMatrixLayout_t, std::size_t, cudaDataType, std::uint64_t, std::uint64_t, std::int64_t) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t,
                                                                       std::size_t,
                                                                       cublasComputeType_t,
                                                                       cudaDataType) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t, std::size_t) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
extern "C" cublasStatus_t CUBLASWINAPI cublasLtMatmul(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, const void* alpha, const void* a,
    cublasLtMatrixLayout_t a_layout, const void* b, cublasLtMatrixLayout_t b_layout,
    const void* beta, const void* c, cublasLtMatrixLayout_t c_layout, void* d,
    cublasLtMatrixLayout_t d_layout, const cublasLtMatmulAlgo_t* algorithm, void* workspace,
    std::size_t workspace_bytes, cudaStream_t stream) {
    return blas_handle(
        reinterpret_cast<cublasHandle_t>(handle), false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            if (entry.context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            if (stream == cudaStreamPerThread) return CUBLAS_STATUS_NOT_SUPPORTED;
            auto* binding =
                stream && stream != cudaStreamLegacy ? shared.find(stream, Kind::kStream) : nullptr;
            if (stream && stream != cudaStreamLegacy &&
                (!binding || binding->context != entry.context || binding->is_retiring))
                return CUBLAS_STATUS_INVALID_VALUE;
            return with_lt_descriptors(
                shared, exclusion, entry,
                {operation, {a_layout, b_layout, c_layout, d_layout}, nullptr},
                [&](const auto& descriptors, ExecutionPayloadWriter& input) {
                    if (!input.write_u32(algorithm ? 1 : 0)) return CUBLAS_STATUS_ALLOC_FAILED;
                    if (algorithm) {
                        cublasLtMatmulAlgo_t copy{};
                        if (!self_copy(&copy, algorithm, sizeof(copy), false))
                            return CUBLAS_STATUS_INVALID_VALUE;
                        if (!input.write_blob(std::as_bytes(std::span{&copy, 1})))
                            return CUBLAS_STATUS_ALLOC_FAILED;
                    }
                    if (!input.write_u64(binding ? binding->remote : 0) ||
                        !input.write_u64(workspace_bytes))
                        return CUBLAS_STATUS_ALLOC_FAILED;
                    const auto range = [&](const void* pointer, std::size_t bytes,
                                           bool optional = false) {
                        if (!pointer && optional && !bytes)
                            return input.write_u64(0) && input.write_u64(0);
                        const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                        auto* allocation = shared.range(address, bytes);
                        return allocation && !allocation->is_retiring &&
                               allocation->context == entry.context &&
                               input.write_u64(allocation->remote) &&
                               input.write_u64(address - allocation->address);
                    };
                    if (!range(workspace, workspace_bytes, true))
                        return CUBLAS_STATUS_INVALID_VALUE;
                    for (const auto* matrix : {a, b, c, static_cast<const void*>(d)})
                        if (!range(matrix, 1)) return CUBLAS_STATUS_INVALID_VALUE;
                    const auto type = static_cast<cudaDataType>(descriptors[0]->scale_type);
                    std::size_t width{};
                    switch (type) {
                        case CUDA_R_16F:
                            width = 2;
                            break;
                        case CUDA_R_32F:
                        case CUDA_R_32I:
                            width = 4;
                            break;
                        case CUDA_R_64F:
                        case CUDA_C_32F:
                            width = 8;
                            break;
                        case CUDA_C_64F:
                            width = 16;
                            break;
                        default:
                            return CUBLAS_STATUS_NOT_SUPPORTED;
                    }
                    const auto mode = descriptors[0]->pointer_mode;
                    if (mode > 1) return CUBLAS_STATUS_NOT_SUPPORTED;
                    if (!input.write_u32(mode) ||
                        !input.write_u32(static_cast<std::uint32_t>(width)))
                        return CUBLAS_STATUS_ALLOC_FAILED;
                    if (mode) {
                        if (!range(alpha, width) || !range(beta, width))
                            return CUBLAS_STATUS_INVALID_VALUE;
                    } else {
                        std::array<std::byte, 16> av{}, bv{};
                        if (!self_copy(av.data(), alpha, width, false) ||
                            !self_copy(bv.data(), beta, width, false))
                            return CUBLAS_STATUS_INVALID_VALUE;
                        if (!input.write_blob(std::span{av}.first(width)) ||
                            !input.write_blob(std::span{bv}.first(width)))
                            return CUBLAS_STATUS_ALLOC_FAILED;
                    }
                    ++entry.context->submissions;
                    return blas_status(shared.call(exclusion, *entry.context,
                                                   ExecutionOpcode::kLtMatmul, std::move(input)),
                                       true);
                });
        },
        true, Kind::kLt, true);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
namespace {
curandStatus_t random_error(CUresult error) noexcept {
    switch (error) {
        case CUDA_ERROR_OUT_OF_MEMORY:
            return CURAND_STATUS_ALLOCATION_FAILED;
        case CUDA_ERROR_INVALID_VALUE:
            return CURAND_STATUS_OUT_OF_RANGE;
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return CURAND_STATUS_NOT_INITIALIZED;
        case CUDA_ERROR_NOT_SUPPORTED:
            return CURAND_STATUS_TYPE_ERROR;
        default:
            return CURAND_STATUS_INTERNAL_ERROR;
    }
}
curandStatus_t random_status(const std::optional<ExecutionWireReply>& reply) noexcept {
    if (!reply) return CURAND_STATUS_INTERNAL_ERROR;
    if (reply->disposition == ExecutionWireDisposition::kComplete && reply->original_status &&
        reply->status == *reply->original_status) {
        const auto value = *reply->original_status;
        if (value == 0 || (value >= 100 && value <= 106) || (value >= 201 && value <= 204) ||
            value == 999)
            return static_cast<curandStatus_t>(value);
        return CURAND_STATUS_INTERNAL_ERROR;
    }
    return random_error(static_cast<CUresult>(reply->status));
}
template <typename Function>
curandStatus_t random_boundary(Function&& function) noexcept {
    if (::getpid() != kProcess) return CURAND_STATUS_NOT_INITIALIZED;
    if (is_entered) return CURAND_STATUS_TYPE_ERROR;
    Entered entered;
    try {
        auto& shared = state();
        const auto entered_at = shared.rpc_metrics
                                    ? shared.rpc_metrics->entry()
                                    : glimmer::interceptor::execution::RpcMetrics::Stamp{};
        Exclusion exclusion(shared.mutex);
        glimmer::interceptor::execution::RpcMetrics::Scope boundary_observation(
            shared.rpc_metrics.get(), exclusion, entered_at, true);
        if (!shared.config) return CURAND_STATUS_NOT_INITIALIZED;
        return function(shared, exclusion);
    } catch (const std::bad_alloc&) {
        return CURAND_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return CURAND_STATUS_INTERNAL_ERROR;
    }
}
template <typename Function>
curandStatus_t random_handle(curandGenerator_t handle, Function&& function) noexcept {
    return random_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.find(handle, Kind::kRandom);
        auto* context = shared.context();
        if (!entry || !context || entry->context != context || context->is_closing ||
            context->is_uncertain || entry->is_uncertain)
            return CURAND_STATUS_NOT_INITIALIZED;
        if (context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1 || entry->active_calls == kMaxHandles)
            return CURAND_STATUS_ALLOCATION_FAILED;
        const auto epoch = context->epoch;
        CounterPin work(context->active_mutations, exclusion);
        CounterPin origin(context->active_calls, exclusion);
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && context->epoch == epoch;
        };
        while (entry->is_retiring)
            if (!yield_frontend(exclusion) || !valid()) return CURAND_STATUS_NOT_INITIALIZED;
        FlagPin mutation(entry->is_retiring, exclusion);
        while (entry->active_calls)
            if (!yield_frontend(exclusion) || !valid()) return CURAND_STATUS_NOT_INITIALIZED;
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
}  // namespace
extern "C" curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t* output,
                                                          curandRngType_t type) {
    return random_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(*output))) return CURAND_STATUS_OUT_OF_RANGE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return random_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CURAND_STATUS_NOT_INITIALIZED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        ExecutionPayloadWriter input;
        std::uint32_t type_bits{};
        static_assert(sizeof(type) == sizeof(type_bits));
        std::memcpy(&type_bits, &type, sizeof(type_bits));
        if (!input.write_u32(type_bits)) return CURAND_STATUS_ALLOCATION_FAILED;
        auto* entry = shared.allocate_entry(Kind::kRandom, context);
        if (!entry) return CURAND_STATUS_ALLOCATION_FAILED;
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateRand, std::move(input));
        const auto result = random_status(reply);
        if (result != CURAND_STATUS_SUCCESS) {
            if (reply && !reply->id &&
                ((reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->original_status && reply->status == *reply->original_status) ||
                 (reply->disposition == ExecutionWireDisposition::kRejected &&
                  !reply->original_status &&
                  (reply->status == CUDA_ERROR_INVALID_VALUE ||
                   reply->status == CUDA_ERROR_OUT_OF_MEMORY ||
                   reply->status == CUDA_ERROR_NOT_SUPPORTED))))
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            else
                context->is_uncertain = entry->is_uncertain = true;
            return result;
        }
        if (!reply->id) {
            context->is_uncertain = entry->is_uncertain = true;
            return CURAND_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        if (!publish(output, reinterpret_cast<curandGenerator_t>(entry))) {
            context->is_uncertain = entry->is_uncertain = true;
            return CURAND_STATUS_INTERNAL_ERROR;
        }
        return CURAND_STATUS_SUCCESS;
    });
}
extern "C" curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(curandGenerator_t handle,
                                                                       unsigned long long seed) {
    return random_handle(handle, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote) || !input.write_u64(seed))
            return CURAND_STATUS_ALLOCATION_FAILED;
        return random_status(shared.call(exclusion, *entry.context, ExecutionOpcode::kSetRandSeed,
                                         std::move(input)));
    });
}
extern "C" curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t handle) {
    return random_handle(handle, [](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return CURAND_STATUS_ALLOCATION_FAILED;
        const auto reply =
            shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyRand, std::move(input));
        const auto result = random_status(reply);
        if (result == CURAND_STATUS_SUCCESS)
            entry.is_alive = false;
        else if (!reply || reply->original_status)
            entry.is_uncertain = true;
        return result;
    });
}

// No client-native fallback may consume an opaque server generator. These are
// explicit policy rejections, not reports of an original SDK invocation.
extern "C" {
curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t*, curandRngType_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetGeneratorOffset(curandGenerator_t, unsigned long long) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t, curandOrdering_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetQuasiRandomGeneratorDimensions(curandGenerator_t, unsigned int) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandSetStream(curandGenerator_t, cudaStream_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateSeeds(curandGenerator_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerate(curandGenerator_t, unsigned int*, std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateLongLong(curandGenerator_t, unsigned long long*,
                                                std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t, float*, std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateUniformDouble(curandGenerator_t, double*, std::size_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateNormal(curandGenerator_t, float*, std::size_t, float,
                                              float) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateNormalDouble(curandGenerator_t, double*, std::size_t, double,
                                                    double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateLogNormal(curandGenerator_t, float*, std::size_t, float,
                                                 float) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateLogNormalDouble(curandGenerator_t, double*, std::size_t,
                                                       double, double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGeneratePoisson(curandGenerator_t, unsigned int*, std::size_t,
                                               double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGeneratePoissonMethod(curandGenerator_t, unsigned int*, std::size_t,
                                                     double, curandMethod_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateBinomial(curandGenerator_t, unsigned int*, std::size_t,
                                                unsigned int, double) {
    return CURAND_STATUS_TYPE_ERROR;
}
curandStatus_t CURANDAPI curandGenerateBinomialMethod(curandGenerator_t, unsigned int*, std::size_t,
                                                      unsigned int, double, curandMethod_t) {
    return CURAND_STATUS_TYPE_ERROR;
}
}

extern "C" cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* output) {
    return blas_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(cublasHandle_t)))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return blas_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations == kMaxHandles)
            return CUBLAS_STATUS_NOT_INITIALIZED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        auto* entry = shared.allocate_entry(Kind::kBlas, context);
        if (!entry) return CUBLAS_STATUS_ALLOC_FAILED;
        ExecutionPayloadWriter input;
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateBlas, std::move(input));
        const auto result = blas_status(reply);
        if (result != CUBLAS_STATUS_SUCCESS) {
            if (reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                reply->original_status && reply->status == *reply->original_status && !reply->id) {
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            } else {
                context->is_uncertain = entry->is_uncertain = true;
            }
            return result;
        }
        if (!reply->id) {
            context->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        entry->remote = reply->id;
        entry->is_alive = true;
        if (!publish(output, reinterpret_cast<cublasHandle_t>(entry))) {
            context->is_uncertain = entry->is_uncertain = true;
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
        return CUBLAS_STATUS_SUCCESS;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle) {
    return blas_handle(handle, true, [](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply =
            shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyBlas, std::move(input));
        const auto result = blas_status(reply);
        if (result == CUBLAS_STATUS_SUCCESS)
            entry.is_alive = false;
        else if (!reply || reply->original_status)
            entry.is_uncertain = true;
        return result;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                                             std::size_t bytes) {
    return blas_handle(handle, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        Entry* allocation{};
        std::uint64_t offset{};
        std::optional<CounterPin> workspace_use;
        if (workspace) {
            const auto address = reinterpret_cast<CUdeviceptr>(workspace);
            allocation = shared.range(address, bytes ? bytes : 1);
            if (!allocation || allocation->context != entry.context ||
                shared.range_is_retiring(*allocation))
                return CUBLAS_STATUS_INVALID_VALUE;
            if (allocation->active_calls >= kMaxHandles) return CUBLAS_STATUS_ALLOC_FAILED;
            workspace_use.emplace(allocation->active_calls, exclusion);
            offset = address - allocation->address;
        } else if (bytes) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote) ||
            !input.write_u64(allocation ? allocation->remote : 0) || !input.write_u64(offset) ||
            !input.write_u64(bytes))
            return CUBLAS_STATUS_ALLOC_FAILED;
        return blas_status(shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kSetBlasWorkspace, std::move(input)));
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle,
                                                            cublasAtomicsMode_t mode) {
    std::uint32_t bits{};
    static_assert(sizeof(mode) == sizeof(bits));
    std::memcpy(&bits, &mode, sizeof(bits));
    return set_blas_value(handle, ExecutionBlasAttribute::kAtomicsMode, bits);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle,
                                                            cublasAtomicsMode_t* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kAtomicsMode, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetSmCountTarget(cublasHandle_t handle, int count) {
    return set_blas_value(handle, ExecutionBlasAttribute::kSmCountTarget,
                          static_cast<std::uint64_t>(count));
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetSmCountTarget(cublasHandle_t handle, int* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kSmCountTarget, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle,
                                                          cudaStream_t stream) {
    return blas_handle(handle, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (stream == cudaStreamLegacy || stream == cudaStreamPerThread)
            return CUBLAS_STATUS_NOT_SUPPORTED;
        auto* binding = stream ? shared.find(stream, Kind::kStream) : nullptr;
        if (stream && (!binding || binding->context != entry.context || binding->is_retiring))
            return CUBLAS_STATUS_INVALID_VALUE;
        ExecutionPayloadWriter input;
        const auto remote = binding ? binding->remote : 0;
        if (!input.write_u64(entry.remote) ||
            !input.write_u32(static_cast<std::uint32_t>(ExecutionBlasAttribute::kStream)) ||
            !input.write_u64(remote))
            return CUBLAS_STATUS_ALLOC_FAILED;
        const auto reply = shared.call(exclusion, *entry.context,
                                       ExecutionOpcode::kSetBlasAttribute, std::move(input));
        const auto result = blas_status(reply);
        if (result == CUBLAS_STATUS_SUCCESS) entry.blas_stream = remote;
        return result;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle,
                                                          cudaStream_t* output) {
    return blas_handle(handle, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!RpcMetrics::Scope::host_range(
                shared.rpc_metrics.get(), exclusion, RpcMetrics::HostRangeCaller::kBlasStream,
                [&](RpcMetrics::HostTrace* trace) noexcept {
                    return writable_host_range(output, sizeof(cudaStream_t), trace);
                }))
            return CUBLAS_STATUS_INVALID_VALUE;
        const auto reply = query_blas(shared, exclusion, entry, ExecutionBlasAttribute::kStream);
        const auto result = blas_status(reply);
        if (result != CUBLAS_STATUS_SUCCESS) return result;
        if (reply->address != entry.blas_stream) return CUBLAS_STATUS_INTERNAL_ERROR;
        cudaStream_t stream{};
        if (reply->address) {
            const auto found = std::ranges::find_if(shared.entries, [&](const auto& value) {
                return value->kind == Kind::kStream && value->context == entry.context &&
                       value->remote == reply->address && value->is_alive;
            });
            if (found == shared.entries.end()) return CUBLAS_STATUS_INTERNAL_ERROR;
            stream = reinterpret_cast<cudaStream_t>(found->get());
        }
        return publish(output, stream) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    });
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle,
                                                               cublasPointerMode_t mode) {
    return set_blas_value(handle, ExecutionBlasAttribute::kPointerMode,
                          static_cast<std::uint64_t>(mode));
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle,
                                                               cublasPointerMode_t* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kPointerMode, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    return set_blas_value(handle, ExecutionBlasAttribute::kMathMode,
                          static_cast<std::uint64_t>(mode));
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t handle,
                                                         cublasMath_t* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kMathMode, output);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t handle, int* output) {
    return get_blas_value(handle, ExecutionBlasAttribute::kVersion, output);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native cuBLAS ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t ta,
                                                      cublasOperation_t tb, int rows, int columns,
                                                      int depth, const float* alpha, const float* a,
                                                      int leading_a, const float* b, int leading_b,
                                                      const float* beta, float* c, int leading_c) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            auto* context = entry.context;
            if (context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(ta)) ||
                !input.write_u32(static_cast<std::uint32_t>(tb)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value : {rows, columns, depth, leading_a, leading_b, leading_c})
                if (!input.write_u32(std::bit_cast<std::uint32_t>(value)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            const auto range = [&](const void* pointer, std::size_t bytes) {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                auto* allocation = shared.range(address, bytes);
                return allocation && !allocation->is_retiring &&
                       input.write_u64(allocation->remote) &&
                       input.write_u64(address - allocation->address);
            };
            // Positive-size matrix starts must not bind to the preceding
            // allocation's one-past-end address when allocations are adjacent.
            if (!range(a, 1) || !range(b, 1) || !range(c, 1)) return CUBLAS_STATUS_INVALID_VALUE;
            const bool device = entry.blas_pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
            if (!input.write_u32(device ? 1 : 0)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (device) {
                if (!range(alpha, sizeof(float)) || !range(beta, sizeof(float)))
                    return CUBLAS_STATUS_INVALID_VALUE;
            } else {
                std::uint32_t av{}, bv{};
                if (!self_copy(&av, alpha, sizeof(av), false) ||
                    !self_copy(&bv, beta, sizeof(bv), false))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!input.write_u32(av) || !input.write_u32(bv)) return CUBLAS_STATUS_ALLOC_FAILED;
            }
            ++context->submissions;
            return blas_status(
                shared.call(exclusion, *context, ExecutionOpcode::kSgemm, std::move(input)), true);
        },
        true);
}

namespace {
std::size_t blas_ex_scalar_bytes(cublasComputeType_t compute, cudaDataType output) noexcept {
    std::size_t width{};
    switch (compute) {
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            width = 2;
            break;
        case CUBLAS_COMPUTE_32F:
        case CUBLAS_COMPUTE_32F_PEDANTIC:
        case CUBLAS_COMPUTE_32F_FAST_16F:
        case CUBLAS_COMPUTE_32F_FAST_16BF:
        case CUBLAS_COMPUTE_32F_FAST_TF32:
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            width = 4;
            break;
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            width = 8;
            break;
        default:
            return 0;
    }
    const bool is_complex = output == CUDA_C_16F || output == CUDA_C_16BF || output == CUDA_C_32F ||
                            output == CUDA_C_64F || output == CUDA_C_8I || output == CUDA_C_8U ||
                            output == CUDA_C_16I || output == CUDA_C_16U || output == CUDA_C_32I ||
                            output == CUDA_C_32U || output == CUDA_C_64I || output == CUDA_C_64U;
    return width * (is_complex ? 2U : 1U);
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): original installed C ABI.
extern "C" cublasStatus_t CUBLASWINAPI cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows, int columns,
    int depth, const void* alpha, const void* a, cudaDataType a_type, int leading_a, const void* b,
    cudaDataType b_type, int leading_b, const void* beta, void* c, cudaDataType c_type,
    int leading_c, cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            const auto scalar_bytes = blas_ex_scalar_bytes(compute, c_type);
            if (!scalar_bytes) return CUBLAS_STATUS_NOT_SUPPORTED;
            for (const auto type : {a_type, b_type, c_type}) {
                const auto bits = std::bit_cast<std::uint32_t>(type);
                if (bits > 29) return CUBLAS_STATUS_INVALID_VALUE;
                if (bits >= 16 && bits <= 19) return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            const auto algorithm_bits = std::bit_cast<std::int32_t>(algorithm);
            if (algorithm_bits < -1 || algorithm_bits > 127) return CUBLAS_STATUS_INVALID_VALUE;
            auto* context = entry.context;
            if (context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            ExecutionPayloadWriter input;
            for (const auto value :
                 {static_cast<std::uint32_t>(a_type), static_cast<std::uint32_t>(b_type),
                  static_cast<std::uint32_t>(c_type), static_cast<std::uint32_t>(compute),
                  std::bit_cast<std::uint32_t>(algorithm_bits)})
                if (!input.write_u32(value)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(ta)) ||
                !input.write_u32(static_cast<std::uint32_t>(tb)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value : {rows, columns, depth, leading_a, leading_b, leading_c})
                if (!input.write_u32(std::bit_cast<std::uint32_t>(value)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            const auto range = [&](const void* pointer, std::size_t bytes) {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                auto* allocation = shared.range(address, bytes);
                return allocation && !allocation->is_retiring &&
                       input.write_u64(allocation->remote) &&
                       input.write_u64(address - allocation->address);
            };
            if (!range(a, 1) || !range(b, 1) || !range(c, 1)) return CUBLAS_STATUS_INVALID_VALUE;
            const bool device = entry.blas_pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
            if (!input.write_u32(device ? 1U : 0U)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (device) {
                if (!range(alpha, scalar_bytes) || !range(beta, scalar_bytes))
                    return CUBLAS_STATUS_INVALID_VALUE;
            } else {
                std::array<std::byte, 16> av{}, bv{};
                if (!self_copy(av.data(), alpha, scalar_bytes, false) ||
                    !self_copy(bv.data(), beta, scalar_bytes, false))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!input.write_blob(std::span{av}.first(scalar_bytes)) ||
                    !input.write_blob(std::span{bv}.first(scalar_bytes)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            }
            ++context->submissions;
            return blas_status(
                shared.call(exclusion, *context, ExecutionOpcode::kGemmEx, std::move(input)), true);
        },
        true);
}

namespace {
// NOLINTBEGIN(bugprone-easily-swappable-parameters): shared exact SDK argument order.
cublasStatus_t submit_strided_blas(cublasHandle_t handle, cublasOperation_t ta,
                                   cublasOperation_t tb, int rows, int columns, int depth,
                                   const void* alpha, const void* a, cudaDataType a_type,
                                   int leading_a, long long stride_a, const void* b,
                                   cudaDataType b_type, int leading_b, long long stride_b,
                                   const void* beta, void* c, cudaDataType c_type, int leading_c,
                                   long long stride_c, int batches, cublasComputeType_t compute,
                                   cublasGemmAlgo_t algorithm, ExecutionOpcode operation) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            const auto scalar_bytes = blas_ex_scalar_bytes(compute, c_type);
            if (!scalar_bytes) return CUBLAS_STATUS_NOT_SUPPORTED;
            if (batches <= 0) return CUBLAS_STATUS_INVALID_VALUE;
            // Inspect raw external enum bits before evaluating either enum.
            std::uint32_t transpose_a_bits{}, transpose_b_bits{};
            static_assert(sizeof(ta) == sizeof(transpose_a_bits) &&
                          sizeof(tb) == sizeof(transpose_b_bits));
            std::memcpy(&transpose_a_bits, &ta, sizeof(transpose_a_bits));
            std::memcpy(&transpose_b_bits, &tb, sizeof(transpose_b_bits));
            if (transpose_a_bits > 2 || transpose_b_bits > 2) return CUBLAS_STATUS_INVALID_VALUE;
            // Installed cuBLAS faults asynchronously on a valid reverse hull.
            // Do not expose that known-unsafe native path in the shared owner.
            if (batches > 1 && (stride_a < 0 || stride_b < 0 || stride_c < 0))
                return CUBLAS_STATUS_NOT_SUPPORTED;
            for (const auto type : {a_type, b_type, c_type}) {
                const auto bits = std::bit_cast<std::uint32_t>(type);
                if (bits > 29) return CUBLAS_STATUS_INVALID_VALUE;
                if (bits >= 16 && bits <= 19) return CUBLAS_STATUS_NOT_SUPPORTED;
            }
            const auto algorithm_bits = std::bit_cast<std::int32_t>(algorithm);
            if (algorithm_bits < -1 || algorithm_bits > 127) return CUBLAS_STATUS_INVALID_VALUE;
            auto* context = entry.context;
            if (context->submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            ExecutionPayloadWriter input;
            for (const auto stride : {stride_a, stride_b, stride_c})
                if (!input.write_u64(std::bit_cast<std::uint64_t>(stride)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            if (!input.write_u32(static_cast<std::uint32_t>(batches)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value :
                 {static_cast<std::uint32_t>(a_type), static_cast<std::uint32_t>(b_type),
                  static_cast<std::uint32_t>(c_type), static_cast<std::uint32_t>(compute),
                  std::bit_cast<std::uint32_t>(algorithm_bits)})
                if (!input.write_u32(value)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (!input.write_u64(entry.remote) ||
                !input.write_u32(static_cast<std::uint32_t>(ta)) ||
                !input.write_u32(static_cast<std::uint32_t>(tb)))
                return CUBLAS_STATUS_ALLOC_FAILED;
            for (const auto value : {rows, columns, depth, leading_a, leading_b, leading_c})
                if (!input.write_u32(std::bit_cast<std::uint32_t>(value)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            const auto range = [&](const void* pointer, std::size_t bytes) {
                const auto address = reinterpret_cast<CUdeviceptr>(pointer);
                auto* allocation = shared.range(address, bytes);
                return allocation && !allocation->is_retiring &&
                       input.write_u64(allocation->remote) &&
                       input.write_u64(address - allocation->address);
            };
            if (!range(a, 1) || !range(b, 1) || !range(c, 1)) return CUBLAS_STATUS_INVALID_VALUE;
            const bool device = entry.blas_pointer_mode == CUBLAS_POINTER_MODE_DEVICE;
            if (!input.write_u32(device ? 1U : 0U)) return CUBLAS_STATUS_ALLOC_FAILED;
            if (device) {
                if (!range(alpha, scalar_bytes) || !range(beta, scalar_bytes))
                    return CUBLAS_STATUS_INVALID_VALUE;
            } else {
                std::array<std::byte, 16> av{}, bv{};
                if (!self_copy(av.data(), alpha, scalar_bytes, false) ||
                    !self_copy(bv.data(), beta, scalar_bytes, false))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!input.write_blob(std::span{av}.first(scalar_bytes)) ||
                    !input.write_blob(std::span{bv}.first(scalar_bytes)))
                    return CUBLAS_STATUS_ALLOC_FAILED;
            }
            ++context->submissions;
            return blas_status(shared.call(exclusion, *context, operation, std::move(input)), true);
        },
        true);
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed public SDK ABIs.
extern "C" cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows, int columns,
    int depth, const void* alpha, const void* a, cudaDataType a_type, int leading_a,
    long long stride_a, const void* b, cudaDataType b_type, int leading_b, long long stride_b,
    const void* beta, void* c, cudaDataType c_type, int leading_c, long long stride_c, int batches,
    cublasComputeType_t compute, cublasGemmAlgo_t algorithm) {
    return submit_strided_blas(handle, ta, tb, rows, columns, depth, alpha, a, a_type, leading_a,
                               stride_a, b, b_type, leading_b, stride_b, beta, c, c_type, leading_c,
                               stride_c, batches, compute, algorithm,
                               ExecutionOpcode::kStridedGemmEx);
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows, int columns,
    int depth, const float* alpha, const float* a, int leading_a, long long stride_a,
    const float* b, int leading_b, long long stride_b, const float* beta, float* c, int leading_c,
    long long stride_c, int batches) {
    return submit_strided_blas(handle, ta, tb, rows, columns, depth, alpha, a, CUDA_R_32F,
                               leading_a, stride_a, b, CUDA_R_32F, leading_b, stride_b, beta, c,
                               CUDA_R_32F, leading_c, stride_c, batches, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT, ExecutionOpcode::kStridedSgemm);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

#include "geam_frontend.inc"
#include "blas_extension_frontend.inc"
