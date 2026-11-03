#include "layout.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

bool accept_resource(void*, glimmer::core::TaskId) noexcept {
    return true;
}

}  // namespace

extern "C" bool glimmer_abi_v1_consumer(const std::size_t* current, std::size_t count) {
    using namespace glimmer::control;
    const auto legacy = control_layout();
    if (current == nullptr || count != legacy.size() ||
        !std::equal(legacy.begin(), legacy.end(), current)) {
        std::cerr << "Control ABI layout changed\n";
        return false;
    }

    glimmer::core::Scheduler scheduler(1024);
    TaskAdmissionService admission(scheduler);
    TaskControlEndpoint endpoint(admission, accept_resource, nullptr);
    if (admission.cancel(0) || admission.cancel_queued(0) || admission.claim_next() ||
        admission.claim(0) || admission.wait_claim(0, std::chrono::milliseconds(1)) ||
        admission.heartbeat(0) || admission.complete(0) || admission.fail(0) ||
        admission.reap_expired() || admission.find(0)) {
        return false;
    }
    static_cast<void>(admission.stats());
    admission.observe_external_dispatch(0);
    admission.observe_external_completion(0, glimmer::core::TaskState::kFailed);
    admission.notify_scheduler_change();
    const auto request = parse_task_protocol_request("GLIMMER_TASK_V1 SUBMIT tenant 1 1 1\n");
    if (!request.parsed() || !request.request.has_value() ||
        !format_task_protocol_request(*request.request)) {
        return false;
    }
    const auto wire_response = endpoint.handle("GLIMMER_TASK_V1 SUBMIT tenant 1 1 1\n");
    if (!wire_response) {
        return false;
    }
    const auto response = parse_task_protocol_response(*wire_response);
    if (!response.parsed() || !response.response.has_value() ||
        response.response->kind != TaskProtocolResponseKind::kAccepted ||
        !format_task_protocol_response(*response.response) ||
        !admission.cancel_queued(response.response->task_id)) {
        return false;
    }

    if (!endpoint.handle("GLIMMER_TASK_V1 STATS\n", std::nullopt)) {
        return false;
    }
    char directory[] = "/tmp/glimmer-abi-v1-XXXXXX";
    if (::mkdtemp(directory) == nullptr) {
        return false;
    }
    struct RemoveDirectory {
        const char* path;
        ~RemoveDirectory() {
            static_cast<void>(::rmdir(path));
        }
    } remove_directory{directory};
    UnixSocketControlConfig config;
    config.socket_path = std::string(directory) + "/control.sock";
    UnixSocketControlServer server(endpoint, config);
    if (server.running() || server.serve_one() || !server.start() || !server.running() ||
        server.serve_one_for(1) != UnixSocketServeStatus::kTimedOut ||
        server.served_request_count() != 0) {
        return false;
    }
    server.stop();
    struct GuardedTiming {
        ControlRequestTiming timing;
        std::array<std::uint8_t, 8> guard;
    } timing{.timing = {}, .guard = {1, 2, 3, 4, 5, 6, 7, 8}};
    UnixSocketControlClient client(std::string{});
    if (client.request("GLIMMER_TASK_V1 STATS\n", &timing.timing)) {
        return false;
    }
    if (timing.guard != std::array<std::uint8_t, 8>{1, 2, 3, 4, 5, 6, 7, 8}) {
        return false;
    }

    // Leave both lease maps populated for destruction compiled with old headers.
    const TaskAdmissionRequest next{.tenant_id = "tenant", .memory_bytes = 1};
    const auto running = admission.submit(next, accept_resource, nullptr);
    const auto pending = admission.submit(next, accept_resource, nullptr);
    return running.accepted() && pending.accepted() && admission.claim(running.task_id).has_value();
}
