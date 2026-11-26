#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace glimmer::interceptor {

// CPU-only exclusion for process capture transitions. The owner supplies every
// physical pause/completion proof; this gate never calls CUDA or user code.
// Thread-safe. Scopes and their methods are caller-thread-bound and lexically
// nested; the gate must outlive them. No scope may be moved to another thread.
class ProcessCallGate final {
   public:
    struct Snapshot {
        std::size_t active_calls{};
        std::size_t admitted_callers{};
        std::size_t waiting_transitions{};
        bool owns_transition{};
        bool is_native_only{};
    };
    // CPU-only observation; not stable issue permission or physical proof.
    // No enclosing Call required. Never use it to bypass lexical admission.
    [[nodiscard]] Snapshot snapshot();

    class Call final {
       public:
        explicit Call(ProcessCallGate& gate);
        ~Call();
        Call(const Call&) = delete;
        Call& operator=(const Call&) = delete;
        [[nodiscard]] bool is_native_only() const noexcept;

       private:
        ProcessCallGate& gate_;
    };

    // Requires an enclosing Call on this thread. Removes that thread's count
    // before waiting, excluding new outer calls and avoiding upgrade deadlock.
    // Its native callbacks may nest Call but may not nest another Transition.
    class Transition final {
       public:
        explicit Transition(ProcessCallGate& gate);
        ~Transition();
        Transition(const Transition&) = delete;
        Transition& operator=(const Transition&) = delete;
        void set_native_only(bool is_native_only) noexcept;

       private:
        ProcessCallGate& gate_;
    };

    // Temporarily detach only for CPU/control admission waiting. No CUDA/source
    // callback or transition may run until resume refreshes the stable phase.
    // Must nest within Call and outlive no other gate scope on this thread.
    class AdmissionWait final {
       public:
        explicit AdmissionWait(ProcessCallGate& gate);
        ~AdmissionWait();
        AdmissionWait(const AdmissionWait&) = delete;
        AdmissionWait& operator=(const AdmissionWait&) = delete;
        void resume();

       private:
        ProcessCallGate& gate_;
        bool has_resumed_{};
    };

    ProcessCallGate() = default;
    ProcessCallGate(const ProcessCallGate&) = delete;
    ProcessCallGate& operator=(const ProcessCallGate&) = delete;
    // Caller-local metadata only; no mutex, native status or physical proof.
    [[nodiscard]] bool is_entered_by_current_thread() const noexcept;
    [[nodiscard]] bool is_native_only_on_current_thread() const noexcept;

   private:
    struct Caller;
    static thread_local Caller current_caller;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t active_calls_{};
    // Reserve one count even while a caller upgrades or waits for admission.
    std::size_t admitted_callers_{};
    std::size_t waiting_transitions_{};
    bool owns_transition_{};
    bool is_native_only_{};
};

}  // namespace glimmer::interceptor
