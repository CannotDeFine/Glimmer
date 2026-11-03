#include "socket_path.h"

#include <cerrno>
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
    if (!S_ISSOCK(path_stat.st_mode)) {
        errno = EINVAL;
        return false;
    }
    *device = static_cast<std::uint64_t>(path_stat.st_dev);
    *inode = static_cast<std::uint64_t>(path_stat.st_ino);
    return true;
}

void unlink_socket_path_if_matches(const std::string& path, std::uint64_t device,
                                   std::uint64_t inode) noexcept {
    std::uint64_t current_device = 0;
    std::uint64_t current_inode = 0;
    if ((device != 0 || inode != 0) && stat_socket_path(path, &current_device, &current_inode) &&
        current_device == device && current_inode == inode) {
        static_cast<void>(::unlink(path.c_str()));
    }
}

}  // namespace glimmer::control::internal
