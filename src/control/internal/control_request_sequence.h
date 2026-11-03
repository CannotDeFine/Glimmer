#pragma once

#include <cstdint>

namespace glimmer::control::internal {

// One process-lifetime correlation source for all synchronous/nonblocking
// native control adapters. Zero means exhaustion; never wrap or reset on
// reconnect/gate destruction. A tag is neither authentication nor authority.
[[nodiscard]] std::uint64_t next_epoch_request_sequence() noexcept;

}  // namespace glimmer::control::internal
