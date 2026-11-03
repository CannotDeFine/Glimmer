#pragma once

namespace glimmer::control::testing {

// Available only in the endpoint test build; affects the calling thread once.
void fail_next_endpoint_response_format() noexcept;
void throw_next_endpoint_response_format() noexcept;

}  // namespace glimmer::control::testing
