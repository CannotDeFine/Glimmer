// These exact SDK entries are original-only, including when GEMM replacement is enabled.
namespace {
template <typename Request>
bool blas_extension_is_metadata_only(const Request& value) noexcept {
    // Candidate API-specific access rules. The backend independently checks
    // metadata and binds every relaxed path to its qualified native version.
    if constexpr (std::is_same_v<Request, ExecutionSgemv>) {
        return value.rows <= 0 || value.columns <= 0;
    } else {
        return value.rows <= 0 || value.columns <= 0 || value.depth < 0 || value.transpose_a > 2 ||
               value.transpose_b > 2 ||
               value.leading_a < std::max(1, value.transpose_a == 0 ? value.rows : value.depth) ||
               value.leading_b <
                   std::max(1, value.transpose_b == 0 ? value.depth : value.columns) ||
               value.leading_c < std::max(1, value.rows);
    }
}

template <typename Request>
bool blas_extension_has_zero_depth(const Request& value) noexcept {
    if constexpr (std::is_same_v<Request, ExecutionSgemmEx>) return value.depth == 0;
    return false;
}

template <typename Request, typename Encode>
cublasStatus_t submit_blas_extension(cublasHandle_t handle, Request value,
                                     const std::array<const void*, 5>& pointers,
                                     ExecutionOpcode operation, Encode encode) noexcept {
    return blas_handle(
        handle, false,
        [&](State& shared, Exclusion& exclusion, Entry& entry) {
            auto& context = *entry.context;
            if (context.submissions == UINT64_MAX) return CUBLAS_STATUS_ALLOC_FAILED;
            const auto epoch = context.epoch;
            value.handle = entry.remote;
            value.pointer_mode = entry.blas_pointer_mode;
            if (value.pointer_mode > 1) return CUBLAS_STATUS_INTERNAL_ERROR;
            const bool is_metadata_only = blas_extension_is_metadata_only(value);
            const bool has_zero_depth = blas_extension_has_zero_depth(value);
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
                const bool can_omit = is_metadata_only || (has_zero_depth && index < 2);
                if (!pointers[index]) {
                    if (!can_omit) return CUBLAS_STATUS_NOT_SUPPORTED;
                    continue;
                }
                value.presence |= presence[index];
                if (index >= 3 && value.pointer_mode == 0) continue;
                const auto address = reinterpret_cast<CUdeviceptr>(pointers[index]);
                // Full strided/matrix extents are checked by the backend against
                // its independent allocation ledger. Never shift a negative-inc base.
                auto* allocation = shared.range(address, 1);
                if (!allocation && can_omit) allocation = shared.range(address, 0);
                if (!allocation) {
                    // No-read evidence does not authorize changing a non-NULL
                    // argument into NULL: native pointer checks can affect status.
                    return CUBLAS_STATUS_INVALID_VALUE;
                }
                if (!allocation->is_alive || allocation->is_uncertain ||
                    allocation->context != &context || shared.range_is_retiring(*allocation))
                    return CUBLAS_STATUS_INVALID_VALUE;
                if (!can_omit && index >= 3 &&
                    (address - allocation->address > allocation->bytes ||
                     sizeof(float) > allocation->bytes - (address - allocation->address)))
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
            // Mixed NULL/non-NULL operand classes need independent access
            // qualification; do not generalize the grouped native controls.
            if (is_metadata_only && value.presence != 0 && value.presence != 3 &&
                value.presence != 28 && value.presence != 31)
                return CUBLAS_STATUS_NOT_SUPPORTED;
            if (!is_metadata_only && has_zero_depth && value.presence != 19 && value.presence != 31)
                return CUBLAS_STATUS_NOT_SUPPORTED;
            // Both API signatures use FP32 scalars, including SgemmEx with FP16 matrices.
            if (value.pointer_mode == 0 && !is_metadata_only &&
                (!self_copy(value.alpha.data(), pointers[3], sizeof(float), false) ||
                 !self_copy(value.beta.data(), pointers[4], sizeof(float), false)))
                return CUBLAS_STATUS_INVALID_VALUE;
            ExecutionPayloadWriter payload;
            if (!encode(payload, value)) return CUBLAS_STATUS_ALLOC_FAILED;
            ++context.submissions;
            // GEAM and these entries share the same strict original-only receipt:
            // neither ASY1 admission nor replacement status grants an SDK return.
            return geam_status(shared.call(exclusion, context, operation, std::move(payload)),
                               entry, epoch);
        },
        true);
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed cuBLAS C ABIs.
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle,
                                                      cublasOperation_t transpose, int rows,
                                                      int columns, const float* alpha,
                                                      const float* a, int leading_a, const float* x,
                                                      int increment_x, const float* beta, float* y,
                                                      int increment_y) {
    return submit_blas_extension(
        handle,
        ExecutionSgemv{.transpose = std::bit_cast<std::uint32_t>(transpose),
                       .rows = rows,
                       .columns = columns,
                       .leading_a = leading_a,
                       .increment_x = increment_x,
                       .increment_y = increment_y},
        {a, x, y, alpha, beta}, ExecutionOpcode::kSgemv, write_execution_sgemv);
}
extern "C" cublasStatus_t CUBLASWINAPI
cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transpose_a, cublasOperation_t transpose_b,
              int rows, int columns, int depth, const float* alpha, const void* a,
              cudaDataType type_a, int leading_a, const void* b, cudaDataType type_b, int leading_b,
              const float* beta, void* c, cudaDataType type_c, int leading_c) {
    return submit_blas_extension(
        handle,
        ExecutionSgemmEx{.transpose_a = std::bit_cast<std::uint32_t>(transpose_a),
                         .transpose_b = std::bit_cast<std::uint32_t>(transpose_b),
                         .type_a = std::bit_cast<std::uint32_t>(type_a),
                         .type_b = std::bit_cast<std::uint32_t>(type_b),
                         .type_c = std::bit_cast<std::uint32_t>(type_c),
                         .rows = rows,
                         .columns = columns,
                         .depth = depth,
                         .leading_a = leading_a,
                         .leading_b = leading_b,
                         .leading_c = leading_c},
        {a, b, c, alpha, beta}, ExecutionOpcode::kSgemmEx, write_execution_sgemm_ex);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
