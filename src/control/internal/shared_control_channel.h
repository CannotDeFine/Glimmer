#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <sys/types.h>

namespace glimmer::control::internal {

inline constexpr std::string_view kSharedControlOpen = "GLIMMER_SHM_V2 OPEN\n";
inline constexpr std::string_view kSharedControlReady = "GLIMMER_SHM_V2 READY\n";
inline constexpr std::string_view kSharedControlUnsupported = "GLIMMER_SHM_V2 UNSUPPORTED\n";
inline constexpr std::string_view kOwnerControlOpen = "GLIMMER_SHM_V3 OPEN\n";
inline constexpr std::string_view kOwnerControlReady = "GLIMMER_SHM_V3 READY\n";
inline constexpr std::string_view kOwnerControlUnsupported = "GLIMMER_SHM_V3 UNSUPPORTED\n";

enum class SharedControlMode : std::uint8_t { kV2, kReplyHintV3 };
enum class SharedControlPublication : std::uint8_t { kRejected, kPublished, kAmbiguous };

using ServiceIncarnation = std::array<std::uint64_t, 2>;
// Empty on unavailable entropy; both words must be nonzero. Never a PID/time
// fallback. The caller owns the identity for one complete server start.
[[nodiscard]] ServiceIncarnation new_service_incarnation() noexcept;

// Optional CPU/latency tradeoff, captured at negotiation. Zero keeps sleeping;
// invalid/out-of-range configuration rejects before channel publication.
[[nodiscard]] std::optional<std::chrono::microseconds> parse_shared_control_spin_budget(
    std::string_view value) noexcept;

enum class SharedControlReadStatus : std::uint8_t { kLine, kIdle, kError };

// One owner process and externally serialized thread per endpoint. A fork child
// may reset inherited mappings but cannot use them. One outstanding request, fresh sealed backing
// per authenticated socket, no retry. The socket is borrowed for liveness;
// descriptors and mappings are owned here. Neither endpoint owns domain state.
class SharedControlChannel final {
   public:
    SharedControlChannel() noexcept = default;
    ~SharedControlChannel();
    SharedControlChannel(const SharedControlChannel&) = delete;
    SharedControlChannel& operator=(const SharedControlChannel&) = delete;

    void reset() noexcept;
    [[nodiscard]] bool create_server() noexcept;
    [[nodiscard]] bool create_server(ServiceIncarnation incarnation,
                                     SharedControlMode mode = SharedControlMode::kV2) noexcept;
    // Caller has sent READY / received READY, respectively. These methods
    // exchange a one-byte acknowledgment and an exact SCM_RIGHTS offer.
    [[nodiscard]] bool send_offer(int socket_fd, std::chrono::milliseconds timeout) noexcept;
    [[nodiscard]] bool receive_offer(int socket_fd, std::chrono::milliseconds timeout,
                                     SharedControlMode mode = SharedControlMode::kV2) noexcept;
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] ServiceIncarnation service_incarnation() const noexcept;

    [[nodiscard]] bool publish_request(std::string_view line) noexcept;
    [[nodiscard]] bool receive_response(int socket_fd, std::uint32_t timeout_ms,
                                        std::string& line) noexcept;
    [[nodiscard]] SharedControlReadStatus receive_request(int socket_fd, std::uint32_t timeout_ms,
                                                          std::string& line) noexcept;
    [[nodiscard]] bool publish_response(std::string_view line) noexcept;
    [[nodiscard]] bool sends_reply_hints() const noexcept;

    // V3 only. Cold negotiation precedes these calls. Externally serialize the
    // endpoint; retain the borrowed socket through completion/reset. No waits,
    // retries or deadlines here. Idle retains the one outstanding request.
    // Rejected means no publication; ambiguous MUST NOT be replayed. A hint is
    // never authority. The caller owns deadlines and fail-closed result mapping.
    [[nodiscard]] SharedControlPublication begin_request(int socket_fd,
                                                         std::string_view line) noexcept;
    [[nodiscard]] SharedControlReadStatus poll_response(int socket_fd, std::string& line) noexcept;
    [[nodiscard]] bool publish_response_with_hint(int socket_fd, std::string_view line) noexcept;

   private:
    std::array<int, 2> descriptors_{-1, -1};
    void* request_mapping_ = nullptr;
    void* response_mapping_ = nullptr;
    std::uint64_t sequence_ = 0;
    bool is_server_ = false;
    bool is_pending_ = false;
    ServiceIncarnation incarnation_{};
    std::chrono::microseconds spin_budget_{};
    SharedControlMode mode_{SharedControlMode::kV2};
    int negotiated_socket_ = -1;
    pid_t process_id_ = 0;
    std::uint64_t received_hints_ = 0;
    bool has_transport_error_ = false;
};

}  // namespace glimmer::control::internal
