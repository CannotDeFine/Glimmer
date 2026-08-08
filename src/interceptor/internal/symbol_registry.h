#pragma once

namespace glimmer::interceptor {

[[nodiscard]] void* find_interceptor_symbol(const char* name) noexcept;

}  // namespace glimmer::interceptor
