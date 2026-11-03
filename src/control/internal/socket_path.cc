#include "socket_path.h"

#include <sys/stat.h>
#include <unistd.h>

namespace glimmer::control::internal {

bool stat_socket_path(const std::string& path, std::uint64_t* device,
                      std::uint64_t* inode) noexcept {
    if (device == nullptr || inode == nullptr) {
        return false;
    }
    struct stat path_stat {};
    if (::lstat(path.c_str(), &path_stat) != 0) {
        return false;
    }
    *device = static_cast<std::uint64_t>(path_stat.st_dev);
    *inode = static_cast<std::uint64_t>(path_stat.st_ino);
    return true;
}

void unlink_socket_path_if_owned(int socket_fd, const std::string& path) noexcept {
    struct stat socket_stat {};
    struct stat path_stat {};
    if (::fstat(socket_fd, &socket_stat) != 0 || ::lstat(path.c_str(), &path_stat) != 0 ||
        socket_stat.st_dev != path_stat.st_dev || socket_stat.st_ino != path_stat.st_ino) {
        return;
    }
    static_cast<void>(::unlink(path.c_str()));
}

}  // namespace glimmer::control::internal
