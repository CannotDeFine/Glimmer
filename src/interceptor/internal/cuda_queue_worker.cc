#include "cuda_queue_worker.h"

#include "diagnostics.h"
#include "owner_control_pump.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace glimmer::interceptor {

struct CudaQueueWorker::Implementation {
    struct NativeScope {
        const CudaQueueWorkerDispatch& dispatch;
        bool previous{};
        explicit NativeScope(const CudaQueueWorkerDispatch& value) : dispatch(value) {
            if (dispatch.enter_native_scope) {
                previous = dispatch.enter_native_scope();
            }
        }
        ~NativeScope() {
            if (dispatch.leave_native_scope) {
                dispatch.leave_native_scope(previous);
            }
        }
        NativeScope(const NativeScope&) = delete;
        NativeScope& operator=(const NativeScope&) = delete;
    };
    struct PendingCommand {
        Command command;
        CUresult result{CUDA_ERROR_NOT_READY};
        bool is_done{};
        bool can_run_paused{};
        CudaOperationToken native_reservation;
    };

    CUcontext context;
    CudaQueueWorkerDispatch dispatch;
    std::shared_ptr<CudaQueueCoordinator> coordinator;
    CudaQueueWorkerOptions options;
    OwnerControlPump control;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<PendingCommand>> commands;
    std::vector<std::shared_ptr<PendingCommand>> native_waits;
    std::thread thread;
    std::unique_ptr<CudaOperationQueue> owned_queue;
    std::thread::id owner;
    CUresult result{CUDA_ERROR_NOT_READY};
    bool is_initialized{};
    bool is_closing{};
    bool is_finished{};
    bool has_checkpoint_request{};
    bool has_checkpoint{};
    bool is_checkpoint_joined{};
    std::uint64_t checkpoint_generation{};
    // Owner-thread state, except shutdown checks protected by mutex. Native
    // operations run on the caller only after the pause command acknowledges.
    bool is_native_paused{};
    bool is_drained_pause{};
    std::uint64_t native_generation{};
    const std::shared_ptr<const void> native_identity{std::make_shared<int>(0)};

    Implementation(CUcontext context_value, CudaQueueWorkerDispatch dispatch_value,
                   std::shared_ptr<CudaQueueCoordinator> coordinator_value,
                   CudaQueueWorkerOptions options_value)
        : context(context_value),
          dispatch(dispatch_value),
          coordinator(std::move(coordinator_value)),
          options(std::move(options_value)),
          control(options.control_gate) {
        if (control.enabled() && !options.progress_signal) {
            options.progress_signal = options.control_gate->subscribe_progress();
            if (!options.progress_signal)
                throw std::runtime_error("CPU actor signal subscription rejected");
        }
    }

    void wake() noexcept {
        condition.notify_all();
        if (options.progress_signal) {
            options.progress_signal->notify();
        }
    }

    void publish_initialized(CUresult value) noexcept {
        {
            const std::lock_guard lock(mutex);
            result = value;
            owner = std::this_thread::get_id();
            is_initialized = true;
        }
        condition.notify_all();
    }

    void finish(CUresult value) noexcept {
        std::deque<std::shared_ptr<PendingCommand>> rejected;
        {
            const std::lock_guard lock(mutex);
            result = value;
            is_finished = true;
            is_closing = true;
            rejected.swap(commands);
            for (const auto& pending : rejected) {
                pending->result = value == CUDA_SUCCESS ? CUDA_ERROR_DEINITIALIZED : value;
                pending->is_done = true;
            }
            for (const auto& pending : native_waits) {
                pending->result = value == CUDA_SUCCESS ? CUDA_ERROR_DEINITIALIZED : value;
                pending->is_done = true;
            }
            native_waits.clear();
        }
        condition.notify_all();
    }

    CUresult execute_commands(CudaOperationQueue& queue) noexcept {
        // Bound each batch so continuous callers cannot starve GPU polling.
        for (std::size_t count = 0; count < options.max_queued_commands; ++count) {
            std::shared_ptr<PendingCommand> pending;
            {
                const std::lock_guard lock(mutex);
                if (commands.empty()) {
                    return CUDA_SUCCESS;
                }
                pending = std::move(commands.front());
                commands.pop_front();
            }
            CUresult value = CUDA_ERROR_UNKNOWN;
            bool has_exception = false;
            try {
                value = is_native_paused && !pending->can_run_paused ? CUDA_ERROR_NOT_READY
                                                                     : pending->command(queue);
            } catch (...) {
                has_exception = true;
            }
            {
                const std::lock_guard lock(mutex);
                pending->result = value;
                pending->is_done = true;
            }
            condition.notify_all();
            if (has_exception) {
                return CUDA_ERROR_UNKNOWN;
            }
        }
        return CUDA_SUCCESS;
    }

    CUresult drive(CudaOperationQueue& queue) noexcept {
        for (;;) {
            if (const auto value = control.progress(); value != CUDA_SUCCESS) return value;
            if (const auto value = execute_commands(queue); value != CUDA_SUCCESS) {
                return value;
            }
            {
                const std::lock_guard lock(mutex);
                if (is_closing && commands.empty()) {
                    return CUDA_SUCCESS;
                }
            }
            if (!is_native_paused) {
                if (const auto value = coordinator->progress(queue); value != CUDA_SUCCESS) {
                    return value;
                }
                if (const auto value = advance_native_wait(queue); value != CUDA_SUCCESS) {
                    return value;
                }
            }
            if (control.enabled()) {
                const auto delay =
                    is_native_paused ? std::chrono::microseconds{25000} : options.poll_interval;
                if (const auto value = control.wait_until(std::chrono::steady_clock::now() + delay,
                                                          options.progress_signal);
                    value != CUDA_SUCCESS)
                    return value;
            } else if (options.progress_signal) {
                (void)options.progress_signal->wait_for(options.poll_interval);
            } else {
                std::unique_lock lock(mutex);
                condition.wait_for(lock, options.poll_interval,
                                   [&] { return is_closing || !commands.empty(); });
            }
        }
    }

    CUresult advance_native_wait(CudaOperationQueue& queue) noexcept {
        std::size_t index = 0;
        // A missing grant is not FIFO authority. Scan other bounded waiters so
        // a lower-priority producer cannot strand an independently granted one.
        for (std::size_t count = 0; count < options.max_queued_commands; ++count) {
            std::shared_ptr<PendingCommand> pending;
            {
                const std::lock_guard lock(mutex);
                if (is_closing || index >= native_waits.size()) return CUDA_SUCCESS;
                pending = native_waits[index];
            }
            CUresult value = CUDA_ERROR_UNKNOWN;
            bool has_exception = false;
            try {
                value = pending->command(queue);
            } catch (...) {
                has_exception = true;
            }
            if (value == CUDA_ERROR_NOT_READY) {
                ++index;
                continue;
            }
            {
                const std::lock_guard lock(mutex);
                pending->result = value;
                pending->is_done = true;
                native_waits.erase(native_waits.begin() + static_cast<std::ptrdiff_t>(index));
            }
            condition.notify_all();
            if (has_exception) return CUDA_ERROR_UNKNOWN;
            // Success published a physical pause; no other prerequisite may
            // execute until the caller supplies its exact trailing fence.
            if (value == CUDA_SUCCESS) return CUDA_SUCCESS;
        }
        return CUDA_SUCCESS;
    }

    CUresult drain_stopped_control() noexcept {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
        for (;;) {
            if (const auto value = control.progress(); value != CUDA_SUCCESS) return value;
            const auto value = coordinator->stopped_progress();
            if (value != CUDA_ERROR_NOT_READY) return value;
            if (std::chrono::steady_clock::now() >= deadline) {
                options.control_gate->quarantine_owner_pump();
                return CUDA_ERROR_UNKNOWN;
            }
            if (const auto waited = control.wait_until(deadline, options.progress_signal);
                waited != CUDA_SUCCESS)
                return waited;
        }
    }

    void run() noexcept {
        if (static_cast<bool>(dispatch.enter_native_scope) !=
            static_cast<bool>(dispatch.leave_native_scope)) {
            publish_initialized(CUDA_ERROR_INVALID_VALUE);
            static_cast<void>(control.retire());
            finish(CUDA_ERROR_INVALID_VALUE);
            return;
        }
        const NativeScope native_scope(dispatch);
        if (!context || !dispatch.context_push || !dispatch.context_pop || !coordinator ||
            options.max_queued_commands == 0 || options.poll_interval.count() <= 0 ||
            (control.enabled() && options.poll_interval > std::chrono::seconds{1})) {
            publish_initialized(CUDA_ERROR_INVALID_VALUE);
            static_cast<void>(control.retire());
            finish(CUDA_ERROR_INVALID_VALUE);
            return;
        }
        auto value = dispatch.context_push(context);
        if (value != CUDA_SUCCESS) {
            publish_initialized(value);
            static_cast<void>(control.retire());
            finish(value);
            return;
        }
        try {
            if (owned_queue) {
                value = owned_queue->adopt_drained_owner();
                if (value != CUDA_SUCCESS) {
                    publish_initialized(value);
                }
            } else {
                owned_queue = std::make_unique<CudaOperationQueue>(context, dispatch.queue,
                                                                   options.queue_limits);
            }
            if (value == CUDA_SUCCESS) {
                {
                    const std::lock_guard lock(mutex);
                    has_checkpoint_request = false;
                    has_checkpoint = false;
                }
                publish_initialized(CUDA_SUCCESS);
                value = drive(*owned_queue);
            }
            if (!has_checkpoint_request) {
                const auto shutdown_result = owned_queue->shutdown();
                coordinator->stopped(*owned_queue, shutdown_result);
                if (control.enabled() && shutdown_result == CUDA_SUCCESS) {
                    const auto drained = drain_stopped_control();
                    if (value == CUDA_SUCCESS) value = drained;
                } else if (control.enabled()) {
                    // Unknown physical stop is never terminal authority. Keep
                    // the queue/resources and fail closed the shared CPU
                    // session before another actor can accept successor work.
                    options.control_gate->quarantine_owner_pump();
                }
                if (value == CUDA_SUCCESS) {
                    value = shutdown_result;
                }
            }
        } catch (...) {
            value = CUDA_ERROR_UNKNOWN;
            // Queue construction can fail before initialization publication.
            bool needs_publication = false;
            {
                const std::lock_guard lock(mutex);
                needs_publication = !is_initialized;
            }
            if (needs_publication) {
                publish_initialized(value);
            }
        }
        CUcontext popped = nullptr;
        const auto retired = control.retire();
        if (value == CUDA_SUCCESS) value = retired;
        const auto pop_result = dispatch.context_pop(&popped);
        if (value == CUDA_SUCCESS) {
            value = pop_result;
            if (value == CUDA_SUCCESS && popped != context) {
                value = CUDA_ERROR_INVALID_CONTEXT;
            }
        }
        {
            const std::lock_guard lock(mutex);
            has_checkpoint = has_checkpoint_request && value == CUDA_SUCCESS;
        }
        finish(value);
    }
};

CudaQueueWorker::CudaQueueWorker(CUcontext context, CudaQueueWorkerDispatch dispatch,
                                 std::shared_ptr<CudaQueueCoordinator> coordinator,
                                 CudaQueueWorkerOptions options)
    : implementation_(std::make_unique<Implementation>(context, dispatch, std::move(coordinator),
                                                       std::move(options))) {
    auto& state = *implementation_;
    if (state.control.attach() != CUDA_SUCCESS) {
        throw std::runtime_error("CPU control owner attach rejected");
    }
    state.thread = std::thread([&state] { state.run(); });
    std::unique_lock lock(state.mutex);
    state.condition.wait(lock, [&] { return state.is_initialized; });
}

CudaQueueWorker::~CudaQueueWorker() {
    if (shutdown() == CUDA_ERROR_NOT_READY || implementation_->has_checkpoint_request) {
        // Caller did not finish its native transaction. No CUDA teardown may
        // race it, and a joinable thread must not be destroyed.
        (void)implementation_.release();
        report_diagnostic("unfinished native CUDA boundary; worker quarantined");
    }
}

CUresult CudaQueueWorker::status() const noexcept {
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    if (state.is_closing && state.result == CUDA_SUCCESS) {
        return CUDA_ERROR_DEINITIALIZED;
    }
    return state.result;
}

CUresult CudaQueueWorker::invoke(Command command) noexcept {
    return enqueue(std::move(command), false);
}

CUresult CudaQueueWorker::enqueue(Command command, bool can_run_paused,
                                  const CudaOperationToken& reservation) noexcept {
    if (!command) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        auto pending = std::make_shared<Implementation::PendingCommand>();
        pending->command = std::move(command);
        pending->can_run_paused = can_run_paused;
        pending->native_reservation = reservation;
        const bool is_native_wait = static_cast<bool>(reservation);
        auto& state = *implementation_;
        std::unique_lock lock(state.mutex);
        if (std::this_thread::get_id() == state.owner) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (state.result != CUDA_SUCCESS) {
            return state.result;
        }
        if (state.is_closing) {
            return CUDA_ERROR_DEINITIALIZED;
        }
        if (is_native_wait && (state.is_native_paused ||
                               std::ranges::any_of(state.native_waits, [&](const auto& value) {
                                   return value->native_reservation == reservation;
                               }))) {
            return CUDA_ERROR_NOT_READY;
        }
        if (state.commands.size() + state.native_waits.size() >=
            state.options.max_queued_commands) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (is_native_wait)
            state.native_waits.push_back(pending);
        else
            state.commands.push_back(pending);
        state.wake();
        state.condition.wait(lock, [&] { return pending->is_done; });
        return pending->result;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult CudaQueueWorker::try_pause_native(CudaNativeBoundaryToken& token) noexcept {
    return try_pause_native_boundary(
        [&](CudaOperationQueue& queue) {
            const auto snapshot = queue.context_snapshot();
            const auto result = queue.query_native_boundary(snapshot);
            return result != CUDA_SUCCESS ? result
                   : queue.pending_count() != 0
                       ? CUDA_ERROR_NOT_READY
                       : implementation_->coordinator->native_boundary_status();
        },
        token, true);
}
CUresult CudaQueueWorker::try_pause_native_reservation(const CudaOperationToken& reservation,
                                                       CudaNativeBoundaryToken& token) noexcept {
    return try_pause_native_reservation(reservation, token,
                                        [](CudaOperationQueue&) { return CUDA_SUCCESS; });
}
CUresult CudaQueueWorker::try_pause_native_reservation(const CudaOperationToken& reservation,
                                                       CudaNativeBoundaryToken& token,
                                                       Command acquire_grant) noexcept {
    if (!acquire_grant) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return try_pause_native_boundary(
        [&](CudaOperationQueue& queue) {
            const auto result = acquire_grant(queue);
            return result == CUDA_SUCCESS ? queue.query_native_reservation_boundary(reservation)
                                          : result;
        },
        token, false);
}
CUresult CudaQueueWorker::await_native_reservation(const CudaOperationToken& reservation,
                                                   CudaNativeBoundaryToken& token,
                                                   Command acquire_grant,
                                                   Command prepare_pause) noexcept {
    if (!reservation || token || !acquire_grant) return CUDA_ERROR_INVALID_VALUE;
    return try_pause_native_boundary(
        [&](CudaOperationQueue& queue) {
            const auto granted = acquire_grant(queue);
            if (granted != CUDA_SUCCESS) return granted;
            const auto physical = queue.query_native_reservation_boundary(reservation);
            if (physical != CUDA_SUCCESS) return physical;
            return prepare_pause ? prepare_pause(queue) : CUDA_SUCCESS;
        },
        token, false, true, reservation);
}
CUresult CudaQueueWorker::try_pause_native_boundary(
    Command prerequisite, CudaNativeBoundaryToken& token, bool is_drained, bool should_wait,
    const CudaOperationToken& reservation) noexcept {
    try {
        return enqueue(
            [&](CudaOperationQueue& queue) {
                auto& state = *implementation_;
                if (state.is_native_paused) {
                    return CUDA_ERROR_NOT_READY;
                }
                const auto result = prerequisite(queue);
                if (result != CUDA_SUCCESS) {
                    return result;
                }
                const std::lock_guard lock(state.mutex);
                if (state.is_closing) {
                    return CUDA_ERROR_DEINITIALIZED;
                }
                if (state.native_generation == std::numeric_limits<std::uint64_t>::max()) {
                    return CUDA_ERROR_UNKNOWN;
                }
                ++state.native_generation;
                state.is_native_paused = true;
                state.is_drained_pause = is_drained;
                token.owner_ = state.native_identity;
                token.generation_ = state.native_generation;
                return CUDA_SUCCESS;
            },
            true, should_wait ? reservation : CudaOperationToken{});
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult CudaQueueWorker::invoke_drained_paused(const CudaNativeBoundaryToken& token,
                                                Command command) noexcept {
    if (!command) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        return enqueue(
            [&, operation = std::move(command)](CudaOperationQueue& queue) {
                auto& state = *implementation_;
                if (!state.is_native_paused || !state.is_drained_pause ||
                    token.owner_ != state.native_identity ||
                    token.generation_ != state.native_generation) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                if (queue.pending_count() != 0) {
                    return CUDA_ERROR_NOT_READY;
                }
                const auto terminal = state.coordinator->native_boundary_status();
                return terminal == CUDA_SUCCESS ? operation(queue) : terminal;
            },
            true);
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

CUresult CudaQueueWorker::finish_native(const CudaNativeBoundaryToken& token,
                                        Command import_native_fence) noexcept {
    if (!import_native_fence) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        return enqueue(
            [&, command = std::move(import_native_fence)](CudaOperationQueue& queue) {
                auto& state = *implementation_;
                if (!state.is_native_paused || token.owner_ != state.native_identity ||
                    token.generation_ != state.native_generation) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                CUresult result = CUDA_ERROR_UNKNOWN;
                try {
                    result = command(queue);
                } catch (...) {
                    // Do not enter CUDA teardown while a native transaction
                    // or capture may still own this paused context.
                    return CUDA_ERROR_UNKNOWN;
                }
                if (result == CUDA_SUCCESS) {
                    const std::lock_guard lock(state.mutex);
                    state.is_native_paused = false;
                }
                return result;
            },
            true);
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}

bool CudaQueueWorker::native_boundary_is_paused(
    const CudaNativeBoundaryToken& token) const noexcept {
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    return state.is_native_paused && token.owner_ == state.native_identity &&
           token.generation_ == state.native_generation;
}

CUresult CudaQueueWorker::shutdown() noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        if (std::this_thread::get_id() == state.owner) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if (state.is_native_paused || state.has_checkpoint_request) {
            return CUDA_ERROR_NOT_READY;
        }
        state.is_closing = true;
    }
    state.wake();
    if (state.thread.joinable()) {
        state.thread.join();
    }
    const std::lock_guard lock(state.mutex);
    return state.result;
}

CUresult CudaQueueWorker::checkpoint(CudaWorkerCheckpoint& output) noexcept {
    if (output) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto& state = *implementation_;
    const auto admitted = enqueue(
        [&](CudaOperationQueue& queue) {
            if (state.is_native_paused || queue.pending_count() != 0) {
                return CUDA_ERROR_NOT_READY;
            }
            auto status = queue.query_native_boundary({});
            if (status == CUDA_SUCCESS) {
                status = state.coordinator->native_boundary_status();
            }
            if (status != CUDA_SUCCESS) {
                return status;
            }
            const std::lock_guard lock(state.mutex);
            if (!state.commands.empty() || !state.native_waits.empty()) {
                return CUDA_ERROR_NOT_READY;
            }
            if (state.checkpoint_generation == std::numeric_limits<std::uint64_t>::max()) {
                return CUDA_ERROR_UNKNOWN;
            }
            ++state.checkpoint_generation;
            state.has_checkpoint_request = true;
            state.is_closing = true;
            return CUDA_SUCCESS;
        },
        false);
    if (admitted != CUDA_SUCCESS) {
        return admitted;
    }
    state.wake();
    state.thread.join();
    const std::lock_guard lock(state.mutex);
    state.is_checkpoint_joined = true;
    if (state.result == CUDA_SUCCESS && state.has_checkpoint) {
        output.owner_ = state.native_identity;
        output.generation_ = state.checkpoint_generation;
    }
    return state.result;
}

CUresult CudaQueueWorker::resume_checkpoint(const CudaWorkerCheckpoint& checkpoint) noexcept {
    auto& state = *implementation_;
    {
        const std::lock_guard lock(state.mutex);
        if (!checkpoint || checkpoint.owner_ != state.native_identity ||
            checkpoint.generation_ != state.checkpoint_generation || !state.has_checkpoint ||
            state.result != CUDA_SUCCESS || !state.is_checkpoint_joined) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    CUcontext current{};
    const Implementation::NativeScope native_scope(state.dispatch);
    const auto status = state.dispatch.queue.context_get_current(&current);
    if (status != CUDA_SUCCESS || current != state.context) {
        return status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : status;
    }
    if (const auto attached = state.control.attach(); attached != CUDA_SUCCESS) return attached;
    std::unique_lock lock(state.mutex);
    state.is_initialized = false;
    state.is_finished = false;
    state.is_closing = false;
    state.is_checkpoint_joined = false;
    // Keep checkpoint ownership until the new native owner is published.
    try {
        state.thread = std::thread([&state] { state.run(); });
    } catch (...) {
        state.is_initialized = true;
        state.is_finished = true;
        state.is_closing = true;
        state.is_checkpoint_joined = true;
        lock.unlock();
        static_cast<void>(state.control.retire());
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    state.condition.wait(lock, [&] { return state.is_initialized; });
    const auto result = state.result;
    lock.unlock();
    if (result != CUDA_SUCCESS && state.thread.joinable()) {
        state.thread.join();
    }
    return result;
}

bool CudaQueueWorker::owns_checkpoint(const CudaWorkerCheckpoint& checkpoint) const noexcept {
    const auto& state = *implementation_;
    const std::lock_guard lock(state.mutex);
    return checkpoint && checkpoint.owner_ == state.native_identity &&
           checkpoint.generation_ == state.checkpoint_generation && state.has_checkpoint &&
           state.has_checkpoint_request && state.result == CUDA_SUCCESS &&
           state.is_checkpoint_joined;
}

CUresult CudaQueueWorker::retire_checkpoint_after_context_destroy(
    const CudaWorkerCheckpoint& checkpoint) noexcept {
    auto& state = *implementation_;
    if (!owns_checkpoint(checkpoint)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto terminal = state.coordinator->native_boundary_status();
    if (terminal != CUDA_SUCCESS) {
        return terminal;
    }
    // The checkpoint joined the native owner and proved an empty queue plus
    // terminal control state. Never call shutdown on reset-destroyed handles.
    state.owned_queue->retire_after_context_destroy();
    state.coordinator->stopped(*state.owned_queue, CUDA_SUCCESS);
    {
        const std::lock_guard lock(state.mutex);
        state.has_checkpoint_request = false;
        state.has_checkpoint = false;
        state.is_checkpoint_joined = false;
        state.owner = {};
    }
    return CUDA_SUCCESS;
}

}  // namespace glimmer::interceptor
