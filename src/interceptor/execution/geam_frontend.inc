// S/D GEAM executes once on the original library owner, never on a caller token.
namespace {
cublasStatus_t geam_status(const std::optional<ExecutionWireReply>& reply, Entry& handle,
                           std::uint64_t epoch) noexcept {
    auto& context = *handle.context;
    const auto fault = [&] { context.is_uncertain = handle.is_uncertain = true; };
    const auto invalid = [&] {
        fault();
        return CUBLAS_STATUS_INTERNAL_ERROR;
    };
    if (!reply || reply->address || !reply->arguments.empty() || !reply->data.empty() ||
        reply->parameter_bytes > 1)
        return invalid();
    const auto original = reply->original_status;
    if (original && (*original < 0 || *original > 31)) return invalid();
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        constexpr auto driver_max =
            std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
        if (reply->status <= 0 || static_cast<std::uint32_t>(reply->status) > driver_max ||
            original)
            return invalid();
        if (reply->parameter_bytes) fault();
        return blas_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || !reply->id || !original ||
        reply->status != *original)
        return invalid();
    if (reply->parameter_bytes) {
        // The observed SDK host return survives a later physical-proof fault.
        // It grants neither future ownership nor successful physical completion.
        fault();
        return static_cast<cublasStatus_t>(*original);
    }
    if (*original != CUBLAS_STATUS_SUCCESS) return static_cast<cublasStatus_t>(*original);
    if (!context.is_alive || context.is_uncertain || context.epoch != epoch || !handle.is_alive ||
        handle.is_uncertain)
        return invalid();
    return CUBLAS_STATUS_SUCCESS;
}

bool geam_reads_host_scalars(const ExecutionGeam& value) noexcept {
    // Audited native scalar admission, not a replacement for original errors.
    // Unknown/invalid operations, invalid LD/shape and zero extents carry
    // presence only; the version-qualified backend supplies safe placeholders.
    return value.transpose_a <= 2 && value.transpose_b <= 2 && value.rows > 0 &&
           value.columns > 0 &&
           value.leading_a >= (value.transpose_a == 0 ? value.rows : value.columns) &&
           value.leading_b >= (value.transpose_b == 0 ? value.rows : value.columns) &&
           value.leading_c >= value.rows;
}

cublasStatus_t submit_geam(cublasHandle_t handle, ExecutionGeam value,
                           const std::array<const void*, 5>& pointers) noexcept {
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            auto& context = *entry.context;
            if (context.submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            const auto epoch = context.epoch;
            value.handle = entry.remote;
            value.pointer_mode = entry.blas_pointer_mode;
            if (value.pointer_mode > 1) return CUBLAS_STATUS_INTERNAL_ERROR;
            constexpr std::array<std::uint32_t, 5> presence{4, 8, 16, 1, 2};
            std::array<Entry*, 10> pinned{};
            std::array<std::optional<CounterPin>, 10> pins;
            std::size_t pin_count{};
            const auto pin = [&](Entry& resource) {
                if (std::ranges::find(pinned, &resource) != pinned.end()) return true;
                if (resource.active_calls >= kMaxHandles - 1 || pin_count == pins.size())
                    return false;
                pinned[pin_count] = &resource;
                pins[pin_count++].emplace(resource.active_calls, exclusion);
                return true;
            };
            for (std::size_t index = 0; index != pointers.size(); ++index) {
                if (!pointers[index]) continue;
                value.presence |= presence[index];
                if (index >= 3 && value.pointer_mode == 0) continue;
                const auto address = reinterpret_cast<CUdeviceptr>(pointers[index]);
                // Prefer positive containment: a neighboring live base must
                // not be mistaken for the preceding allocation's endpoint.
                auto* allocation = shared.range(address, 1);
                if (!allocation) allocation = shared.range(address, 0);
                if (!allocation) {
                    const auto known = std::ranges::any_of(shared.entries, [&](const auto& old) {
                        return (old->kind == Kind::kAllocation ||
                                old->kind == Kind::kModuleGlobal) &&
                               old->address && address >= old->address &&
                               address - old->address <= old->bytes;
                    });
                    if (known) return CUBLAS_STATUS_INVALID_VALUE;
                    continue;  // Unresolved presence conveys no native pointer or authority.
                }
                if (!allocation->is_alive || allocation->is_uncertain ||
                    allocation->context != &context || shared.range_is_retiring(*allocation))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!pin(*allocation)) return CUBLAS_STATUS_ALLOC_FAILED;
                if (allocation->kind == Kind::kModuleGlobal) {
                    const auto parent =
                        std::ranges::find_if(shared.entries, [&](const auto& module) {
                            return module->kind == Kind::kModule && module->context == &context &&
                                   module->remote == allocation->parent && module->is_alive &&
                                   !module->is_retiring && !module->is_uncertain;
                        });
                    if (parent == shared.entries.end()) return CUBLAS_STATUS_INVALID_VALUE;
                    if (!pin(**parent)) return CUBLAS_STATUS_ALLOC_FAILED;
                }
                value.buffers[index] = {allocation->remote, address - allocation->address};
            }
            if (value.pointer_mode == 0 && geam_reads_host_scalars(value)) {
                const auto width = value.type == ExecutionGeamType::kFloat ? 4U : 8U;
                if ((pointers[3] && !self_copy(value.alpha.data(), pointers[3], width, false)) ||
                    (pointers[4] && !self_copy(value.beta.data(), pointers[4], width, false)))
                    return CUBLAS_STATUS_INVALID_VALUE;
            }
            ExecutionPayloadWriter payload;
            if (!write_execution_geam(payload, value)) return CUBLAS_STATUS_ALLOC_FAILED;
            ++context.submissions;
            return geam_status(
                shared.call(exclusion, context, ExecutionOpcode::kGeam, std::move(payload)), entry,
                epoch);
        },
        true);
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed cuBLAS C ABIs.
extern "C" cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle,
                                                   cublasOperation_t transpose_a,
                                                   cublasOperation_t transpose_b, int rows,
                                                   int columns, const float* alpha, const float* a,
                                                   int leading_a, const float* beta, const float* b,
                                                   int leading_b, float* c, int leading_c) {
    return submit_geam(handle,
                       {.type = ExecutionGeamType::kFloat,
                        .transpose_a = std::bit_cast<std::uint32_t>(transpose_a),
                        .transpose_b = std::bit_cast<std::uint32_t>(transpose_b),
                        .rows = rows,
                        .columns = columns,
                        .leading_a = leading_a,
                        .leading_b = leading_b,
                        .leading_c = leading_c},
                       {a, b, c, alpha, beta});
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasDgeam(cublasHandle_t handle, cublasOperation_t transpose_a, cublasOperation_t transpose_b,
            int rows, int columns, const double* alpha, const double* a, int leading_a,
            const double* beta, const double* b, int leading_b, double* c, int leading_c) {
    return submit_geam(handle,
                       {.type = ExecutionGeamType::kDouble,
                        .transpose_a = std::bit_cast<std::uint32_t>(transpose_a),
                        .transpose_b = std::bit_cast<std::uint32_t>(transpose_b),
                        .rows = rows,
                        .columns = columns,
                        .leading_a = leading_a,
                        .leading_b = leading_b,
                        .leading_c = leading_c},
                       {a, b, c, alpha, beta});
}
// NOLINTEND(bugprone-easily-swappable-parameters)
