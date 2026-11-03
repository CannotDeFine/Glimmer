#pragma once

namespace glimmer::control::testing {

// Available only in the endpoint test build; affects the calling thread once.
void fail_next_endpoint_response_format() noexcept;
void throw_next_endpoint_response_format() noexcept;
void mismatch_next_epoch_grant_extent() noexcept;
void error_next_epoch_grant() noexcept;

}  // namespace glimmer::control::testing
