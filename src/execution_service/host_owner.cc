#include "host_owner.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <fcntl.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::app {
namespace {
using namespace backend::cuda;
thread_local std::shared_ptr<const void> control_owner;
CudaSessionResult rejected(CUresult status) {
    CudaSessionResult result;
    result.status = status;
    return result;
}
}  // namespace
struct HostOwner::Implementation {
    struct Work {
        HostOwnerCommand command;
        control::ExecutionOwnedDescriptor descriptor;
    };
    struct Completed {
        HostOwnerCommand command;
        HostOwnerReply reply;
        CudaHostMemoryView view;
    };
    struct CompletedShutdown {
        std::uint64_t sequence{};
        CudaSessionResult result;
    };
    struct State {
        std::mutex mutex;
        std::condition_variable wake;
        std::optional<Work> work;
        std::optional<Completed> completed;
        std::optional<CompletedShutdown> completed_shutdown;
        std::uint64_t shutdown_sequence{};
        HostOwnerSnapshot snapshot;
        bool is_closing{}, is_executing{}, may_shutdown{};
        std::shared_ptr<control::ExecutionSocketWake> completion_wake;
    };
    struct Mapping {
        std::uint64_t id{};
        std::size_t bytes{};
        void* pointer{};
        CUdeviceptr address{};
        std::weak_ptr<const void> lifetime;
        bool is_retiring{};
        bool is_pageable{};
    };
    const pid_t process{::getpid()};
    std::weak_ptr<const void> owner;
    std::shared_ptr<State> state{std::make_shared<State>()};
    std::vector<Mapping> mappings;
    std::thread worker;

    Implementation(backend::ExecutionSessionIdentity identity, std::uint64_t context,
                   CudaSessionDispatch dispatch, CudaSessionLimits limits, int device,
                   std::shared_ptr<control::ExecutionSocketWake> wake) {
        if (!control_owner) control_owner = std::make_shared<int>(0);
        owner = control_owner;
        state->completion_wake = std::move(wake);
        mappings.reserve(limits.max_objects);
        worker = std::thread([shared = state, identity, context, dispatch = std::move(dispatch),
                              limits, device] {
            std::unique_ptr<CudaExecutionSession> session;
            for (;;) {
                std::optional<Work> work;
                bool should_shutdown{};
                std::uint64_t shutdown_sequence{};
                {
                    std::unique_lock lock(shared->mutex);
                    shared->wake.wait(lock, [&] { return shared->work || shared->may_shutdown; });
                    should_shutdown = shared->may_shutdown;
                    shutdown_sequence = shared->shutdown_sequence;
                    shared->may_shutdown = false;
                    shared->shutdown_sequence = 0;
                    if (!should_shutdown) work = std::move(shared->work);
                    shared->work.reset();
                    shared->is_executing = true;
                }
                Completed completed;
                CudaSessionResult shutdown_result;
                CudaSessionSnapshot snapshot;
                bool is_closed{};
                bool is_unproven{};
                try {
                    if (should_shutdown) {
                        if (session)
                            shutdown_result = session->shutdown();
                        else {
                            shutdown_result.status = CUDA_SUCCESS;
                            shutdown_result.disposition = CudaSessionDisposition::kComplete;
                        }
                        is_closed =
                            shutdown_result.status == CUDA_SUCCESS &&
                            shutdown_result.disposition == CudaSessionDisposition::kComplete;
                    } else {
                        completed.command = work->command;
                        auto& reply = completed.reply;
                        const auto& command = work->command;
                        if (command.operation == HostOwnerOperation::kInitialize) {
                            if (session)
                                reply.result = rejected(CUDA_ERROR_INVALID_CONTEXT);
                            else {
                                session = std::make_unique<CudaExecutionSession>(identity, context,
                                                                                 dispatch, limits);
                                reply.result = session->initialize(device, {command.flags});
                                if (reply.result.status != CUDA_SUCCESS) (void)session->shutdown();
                            }
                            const auto observed =
                                session ? session->snapshot() : CudaSessionSnapshot{};
                            const bool retained =
                                observed.has_context || observed.host_memory_bytes;
                            reply.ownership = retained
                                                  ? control::ExecutionBootstrapOwnership::kRetained
                                              : observed.is_quarantined
                                                  ? control::ExecutionBootstrapOwnership::kUnproven
                                                  : control::ExecutionBootstrapOwnership::kNone;
                        } else if (!session)
                            reply.result = rejected(CUDA_ERROR_INVALID_CONTEXT);
                        else if (command.operation == HostOwnerOperation::kRuntimeMetadata) {
                            reply.metadata = session->runtime_metadata(command.metadata);
                            reply.result.status = reply.metadata->status;
                            reply.result.disposition = reply.metadata->disposition;
                        } else if (command.operation == HostOwnerOperation::kImport) {
                            reply.result = session->import_host_memory(
                                work->descriptor.get(), command.bytes, command.mapping_bytes,
                                command.flags, command.is_pageable);
                            if (reply.result.status == CUDA_SUCCESS) {
                                auto view =
                                    session->host_memory({reply.result.id, 0}, command.bytes);
                                if (view)
                                    completed.view = std::move(*view);
                                else {
                                    is_unproven = true;
                                    reply.result.status = CUDA_ERROR_UNKNOWN;
                                    reply.result.disposition = CudaSessionDisposition::kRejected;
                                    session->close();
                                }
                            }
                        } else
                            reply.result = session->release_host_memory(command.allocation);
                    }
                    if (session) snapshot = session->snapshot();
                } catch (...) {
                    completed.reply.result = rejected(CUDA_ERROR_OUT_OF_MEMORY);
                    if (should_shutdown) {
                        shutdown_result = rejected(CUDA_ERROR_UNKNOWN);
                        is_unproven = true;
                    }
                    if (session) snapshot = session->snapshot();
                    if (work && work->command.operation == HostOwnerOperation::kInitialize)
                        completed.reply.ownership =
                            session ? control::ExecutionBootstrapOwnership::kUnproven
                                    : control::ExecutionBootstrapOwnership::kNone;
                }
                snapshot.is_quarantined = snapshot.is_quarantined || is_unproven;
                {
                    std::lock_guard lock(shared->mutex);
                    shared->snapshot = {snapshot, false, is_closed};
                    shared->is_executing = false;
                    if (!should_shutdown && !shared->is_closing)
                        shared->completed = std::move(completed);
                    if (should_shutdown && shutdown_sequence)
                        shared->completed_shutdown =
                            CompletedShutdown{shutdown_sequence, std::move(shutdown_result)};
                }
                // Publish first, then wake the socket consumer outside the
                // mailbox mutex. The hint never substitutes for an exact result.
                if (shared->completion_wake && !shared->completion_wake->notify()) {
                    try {
                        spdlog::warn("Execution host publication wake failed; finite poll remains");
                    } catch (...) {
                    }
                }
                if (is_closed) return;
            }
        });
    }
    bool owns() const noexcept {
        return ::getpid() == process && control_owner && owner.lock() == control_owner;
    }
};
HostOwner::HostOwner(backend::ExecutionSessionIdentity identity, std::uint64_t context,
                     CudaSessionDispatch dispatch, CudaSessionLimits limits, int device,
                     std::shared_ptr<control::ExecutionSocketWake> wake)
    : implementation_(std::make_unique<Implementation>(identity, context, std::move(dispatch),
                                                       limits, device, std::move(wake))) {}
HostOwner::~HostOwner() {
    if (!implementation_->owns()) {
        (void)implementation_.release();
        return;
    }
    close();
    const auto observed = snapshot();
    if (observed.is_closed)
        implementation_->worker.join();
    else {
        // The worker owns its shared state/session until real native cleanup.
        // Never join an SDK call that may need GPU-owner progress to return.
        implementation_->worker.detach();
        try {
            spdlog::critical("Retaining unsettled native host owner");
        } catch (...) {
        }
    }
}
bool HostOwner::submit(const HostOwnerCommand& command, int borrowed_descriptor) {
    auto& state = *implementation_;
    if (!state.owns() || !command.sequence) return false;
    std::lock_guard lock(state.state->mutex);
    if (state.state->is_closing || state.state->is_executing || state.state->work ||
        state.state->completed)
        return false;
    control::ExecutionOwnedDescriptor descriptor;
    if (command.operation == HostOwnerOperation::kImport) {
        descriptor =
            control::ExecutionOwnedDescriptor(::fcntl(borrowed_descriptor, F_DUPFD_CLOEXEC, 0));
        if (descriptor.get() < 0)
            throw std::runtime_error("host import descriptor duplication failed");
    }
    if (command.operation == HostOwnerOperation::kRelease) {
        const auto found =
            std::ranges::find(state.mappings, command.allocation, &Implementation::Mapping::id);
        if (found != state.mappings.end()) found->is_retiring = true;
    }
    state.state->work = Implementation::Work{command, std::move(descriptor)};
    state.state->snapshot.has_reservation = true;
    state.state->wake.notify_one();
    return true;
}
std::optional<HostOwnerReply> HostOwner::take_reply(std::uint64_t sequence) {
    auto& state = *implementation_;
    if (!state.owns()) return std::nullopt;
    std::lock_guard lock(state.state->mutex);
    if (!state.state->completed || state.state->completed->command.sequence != sequence)
        return std::nullopt;
    auto completed = std::move(*state.state->completed);
    state.state->completed.reset();
    const auto& result = completed.reply.result;
    if (completed.command.operation == HostOwnerOperation::kImport &&
        result.status == CUDA_SUCCESS) {
        if (!completed.view.lifetime)
            throw std::logic_error("successful host import has no native lease");
        state.mappings.push_back({result.id, completed.command.bytes, completed.view.pointer,
                                  completed.view.address, completed.view.lifetime, false,
                                  completed.view.is_pageable});
    } else if (completed.command.operation == HostOwnerOperation::kRelease &&
               result.status == CUDA_SUCCESS)
        std::erase_if(state.mappings, [&](const auto& mapping) {
            return mapping.id == completed.command.allocation;
        });
    return completed.reply;
}
HostOwnerSnapshot HostOwner::snapshot() const {
    const auto& state = *implementation_;
    if (!state.owns()) {
        HostOwnerSnapshot rejected;
        rejected.session.is_quarantined = true;
        return rejected;
    }
    std::lock_guard lock(state.state->mutex);
    return state.state->snapshot;
}
std::optional<bool> HostOwner::allocation_idle(std::uint64_t allocation) const {
    const auto& state = *implementation_;
    if (!state.owns() || !allocation) return std::nullopt;
    std::lock_guard lock(state.state->mutex);
    if (state.state->is_closing || state.state->snapshot.is_closed ||
        state.state->snapshot.session.is_quarantined)
        return std::nullopt;
    const auto found = std::ranges::find(state.mappings, allocation, &Implementation::Mapping::id);
    if (found == state.mappings.end() || found->is_retiring) return std::nullopt;
    // The backend retains exactly one baseline owner. Every accepted physical
    // use retains a strong lease until actual retirement, not SDK return. Only
    // this serialized control thread can borrow or start release of a mapping.
    const auto owners = found->lifetime.use_count();
    if (!owners) return std::nullopt;
    return owners == 1;
}
std::optional<CudaHostMemoryView> HostOwner::borrow(backend::cuda::CudaAllocationRange range,
                                                    std::size_t bytes) {
    auto& state = *implementation_;
    if (!state.owns() || snapshot().session.is_quarantined) return std::nullopt;
    const auto found =
        std::ranges::find(state.mappings, range.allocation, &Implementation::Mapping::id);
    if (found == state.mappings.end() || found->is_retiring || range.offset > found->bytes ||
        bytes > found->bytes - range.offset)
        return std::nullopt;
    auto lifetime = found->lifetime.lock();
    if (!lifetime) return std::nullopt;
    return CudaHostMemoryView{static_cast<std::byte*>(found->pointer) + range.offset,
                              found->is_pageable ? 0 : found->address + range.offset, bytes,
                              std::move(lifetime), found->is_pageable};
}
std::vector<std::shared_ptr<const void>> HostOwner::lifetimes() {
    auto& state = *implementation_;
    if (!state.owns() || snapshot().session.is_quarantined)
        throw std::logic_error("unavailable native host owner");
    std::vector<std::shared_ptr<const void>> result;
    result.reserve(state.mappings.size());
    for (const auto& mapping : state.mappings)
        if (!mapping.is_retiring) {
            auto lease = mapping.lifetime.lock();
            if (!lease) throw std::logic_error("live native host mapping lost ownership");
            result.push_back(std::move(lease));
        }
    return result;
}
void HostOwner::close() {
    auto& state = *implementation_;
    if (!state.owns()) return;
    for (auto& mapping : state.mappings) mapping.is_retiring = true;
    std::lock_guard lock(state.state->mutex);
    state.state->is_closing = true;
    state.state->work.reset();
    state.state->completed.reset();
    if (!state.state->is_executing && !state.state->may_shutdown)
        state.state->snapshot.has_reservation = false;
}
void HostOwner::permit_shutdown() {
    auto& state = *implementation_;
    if (!state.owns()) return;
    std::lock_guard lock(state.state->mutex);
    if (!state.state->is_closing || state.state->snapshot.is_closed || state.state->is_executing ||
        state.state->may_shutdown || state.state->completed_shutdown)
        return;
    state.state->may_shutdown = true;
    state.state->snapshot.has_reservation = true;
    state.state->wake.notify_one();
}
bool HostOwner::permit_shutdown(std::uint64_t sequence) {
    auto& state = *implementation_;
    if (!state.owns() || !sequence) return false;
    std::lock_guard lock(state.state->mutex);
    if (!state.state->is_closing || state.state->snapshot.is_closed || state.state->is_executing ||
        state.state->may_shutdown || state.state->work || state.state->completed ||
        state.state->completed_shutdown || state.state->snapshot.session.is_quarantined)
        return false;
    state.state->shutdown_sequence = sequence;
    state.state->may_shutdown = true;
    state.state->snapshot.has_reservation = true;
    state.state->wake.notify_one();
    return true;
}
std::optional<CudaSessionResult> HostOwner::take_shutdown_reply(std::uint64_t sequence) {
    auto& state = *implementation_;
    if (!state.owns() || !sequence) return std::nullopt;
    std::lock_guard lock(state.state->mutex);
    if (!state.state->completed_shutdown || state.state->completed_shutdown->sequence != sequence)
        return std::nullopt;
    auto result = std::move(state.state->completed_shutdown->result);
    state.state->completed_shutdown.reset();
    return result;
}
}  // namespace glimmer::app
