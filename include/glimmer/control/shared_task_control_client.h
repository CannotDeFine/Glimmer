#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

// A process-local, serialized logical session on the authenticated local
// shared channel. The first successful negotiation pins one service incarnation
// for the lifetime of this object, including across reconnects. A lost reply is
// ambiguous: requests are never replayed and no other transport is selected.
// The first same-UID service at the path is trusted on first use; keep the
// socket path in a trusted directory. All request calls must finish before
// destruction. Fork children must not use inherited clients or lease identities
// and must promptly exec or _exit.
class SharedTaskControlClient final {
   public:
    explicit SharedTaskControlClient(std::string socket_path, std::uint32_t io_timeout_ms = 1000);
    ~SharedTaskControlClient();
    SharedTaskControlClient(const SharedTaskControlClient&) = delete;
    SharedTaskControlClient& operator=(const SharedTaskControlClient&) = delete;
    SharedTaskControlClient(SharedTaskControlClient&&) = delete;
    SharedTaskControlClient& operator=(SharedTaskControlClient&&) = delete;

    [[nodiscard]] std::optional<std::string> request(std::string_view line) noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace glimmer::control
