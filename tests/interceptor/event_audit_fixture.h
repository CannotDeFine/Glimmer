#pragma once

#include <array>

namespace glimmer::tests {
enum class EventCall : unsigned int {
    kCreate,
    kCreateWithFlags,
    kRecord,
    kRecordPtds,
    kRecordFlags,
    kRecordFlagsPtds,
    kQuery,
    kSynchronize,
    kDestroy,
    kElapsed,
    kWait,
    kWaitPtds,
    kCount
};
struct EventAudit {
    std::array<unsigned int, static_cast<unsigned int>(EventCall::kCount)> calls{};
    void* output{};
    void* event{};
    void* stream{};
    void* end{};
    unsigned int flags{};
    int result{};
};
inline void record_event_call(EventAudit& audit, EventCall call, void* output, void* event,
                              void* stream, void* end, unsigned int flags) noexcept {
    ++audit.calls[static_cast<unsigned int>(call)];
    audit.output = output;
    audit.event = event;
    audit.stream = stream;
    audit.end = end;
    audit.flags = flags;
}
}  // namespace glimmer::tests
