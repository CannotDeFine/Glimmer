#pragma once

namespace glimmer::interceptor {

[[nodiscard]] void* find_interceptor_symbol(const char* name,
                                            bool per_thread_default_stream = false) noexcept;

}  // namespace glimmer::interceptor
