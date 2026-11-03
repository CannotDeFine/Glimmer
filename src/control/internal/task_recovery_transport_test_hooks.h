#pragma once

namespace glimmer::control::testing {
// Test builds only: drop one committed recovery reply before publication.
void drop_next_recovery_response() noexcept;
}  // namespace glimmer::control::testing
