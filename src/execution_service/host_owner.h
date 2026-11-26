#pragma once

#include "glimmer/control/execution_socket.h"
#include "internal/cuda_execution_session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace glimmer::app {
enum class HostOwnerOperation { kInitialize, kImport, kRelease, kRuntimeMetadata };
struct HostOwnerCommand {
    HostOwnerOperation operation{HostOwnerOperation::kInitialize};
    std::uint64_t sequence{}, allocation{};
    std::size_t bytes{}, mapping_bytes{};
    unsigned int flags{};
    backend::cuda::CudaRuntimeMetadataQuery metadata{};
    bool is_pageable{};
};
struct HostOwnerReply {
    backend::cuda::CudaSessionResult result;
    control::ExecutionBootstrapOwnership ownership{control::ExecutionBootstrapOwnership::kUnproven};
    std::optional<backend::cuda::CudaRuntimeMetadataResult> metadata{};
};
struct HostOwnerSnapshot {
    backend::cuda::CudaSessionSnapshot session;
    bool has_reservation{}, is_closed{};
};
// CPU methods/lifecycle belong to the constructing process/thread. CUDA session
// creation, calls and cleanup belong exclusively to a separate stable worker.
// At most one granted command/result is retained. Never waits for native work.
class HostOwner final {
   public:
    HostOwner(backend::ExecutionSessionIdentity identity, std::uint64_t context,
              backend::cuda::CudaSessionDispatch dispatch, backend::cuda::CudaSessionLimits limits,
              int device, std::shared_ptr<control::ExecutionSocketWake> wake = {});
    ~HostOwner();
    HostOwner(const HostOwner&) = delete;
    HostOwner& operator=(const HostOwner&) = delete;
    [[nodiscard]] bool submit(const HostOwnerCommand& command, int borrowed_descriptor = -1);
    [[nodiscard]] std::optional<HostOwnerReply> take_reply(std::uint64_t sequence);
    [[nodiscard]] HostOwnerSnapshot snapshot() const;
    // CPU-owner observation only: null means invalid/retiring/unproven,
    // false means a physical borrower remains. Does not borrow or run the SDK.
    [[nodiscard]] std::optional<bool> allocation_idle(std::uint64_t allocation) const;
    [[nodiscard]] std::optional<backend::cuda::CudaHostMemoryView> borrow(
        backend::cuda::CudaAllocationRange range, std::size_t bytes);
    [[nodiscard]] std::vector<std::shared_ptr<const void>> lifetimes();
    void close();
    void permit_shutdown();
    // Exact shutdown attempts retain their actual result until consumed. A
    // Deferred result needs a new external grant; idle reservation is no proof.
    [[nodiscard]] bool permit_shutdown(std::uint64_t sequence);
    [[nodiscard]] std::optional<backend::cuda::CudaSessionResult> take_shutdown_reply(
        std::uint64_t sequence);

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::app
