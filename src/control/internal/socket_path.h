#pragma once

#include <cstdint>
#include <string>

namespace glimmer::control::internal {

[[nodiscard]] bool stat_socket_path(const std::string& path, std::uint64_t* device,
                                    std::uint64_t* inode) noexcept;
void unlink_socket_path_if_owned(int socket_fd, const std::string& path) noexcept;

}  // namespace glimmer::control::internal
