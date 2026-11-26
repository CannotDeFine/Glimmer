#include "profile_archive.h"
#include "internal/cuda_profile_cache.h"
#include "glimmer/control/execution_socket.h"

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::app {
namespace {
using control::ExecutionOwnedDescriptor;
struct Parent {
    ExecutionOwnedDescriptor directory;
    std::string name;
};
std::optional<Parent> open_parent(std::string_view path) {
    if (path.size() < 2 || path.size() > 4096 || path.front() != '/' ||
        path.find('\0') != std::string_view::npos)
        return std::nullopt;
    ExecutionOwnedDescriptor directory{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
    if (directory.get() < 0) return std::nullopt;
    path.remove_prefix(1);
    while (true) {
        const auto slash = path.find('/');
        const auto part = path.substr(0, slash);
        if (part.empty() || part == "." || part == ".." || part.size() > 255) return std::nullopt;
        std::string name{part};
        if (slash == std::string_view::npos) {
            struct stat status {};
            if (::fstat(directory.get(), &status) != 0 || status.st_uid != ::geteuid() ||
                (status.st_mode & 0022) != 0)
                return std::nullopt;
            return Parent{std::move(directory), std::move(name)};
        }
        ExecutionOwnedDescriptor next{::openat(directory.get(), name.c_str(),
                                               O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)};
        if (next.get() < 0) return std::nullopt;
        directory = std::move(next);
        path.remove_prefix(slash + 1);
    }
}
bool same_file(const struct stat& before, const struct stat& after) noexcept {
    return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
           before.st_size == after.st_size && before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
           before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
           before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
           before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}
struct Temporary {
    int parent;
    std::string name;
    bool is_published{};
    ~Temporary() {
        if (!is_published && ::unlinkat(parent, name.c_str(), 0) != 0) {
            try {
                spdlog::error("Execution archive temporary cleanup failed: errno={}", errno);
            } catch (...) {
            }
        }
    }
};
}  // namespace

std::optional<std::string> profile_runtime_identity() noexcept {
    try {
        // Follow only this kernel-owned procfs executable reference.
        ExecutionOwnedDescriptor executable{::open("/proc/self/exe", O_RDONLY | O_CLOEXEC)};
        struct stat status {};
        if (executable.get() < 0 || ::fstat(executable.get(), &status) != 0 ||
            !S_ISREG(status.st_mode) || status.st_size <= 0)
            return std::nullopt;
        return "glimmer-profile-v1:" + std::to_string(status.st_dev) + ":" +
               std::to_string(status.st_ino) + ":" + std::to_string(status.st_size) + ":" +
               std::to_string(status.st_mtim.tv_sec) + ":" +
               std::to_string(status.st_mtim.tv_nsec) + ":" +
               std::to_string(status.st_ctim.tv_sec) + ":" + std::to_string(status.st_ctim.tv_nsec);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::vector<std::byte>> read_execution_archive(std::string_view path,
                                                             std::size_t maximum_bytes) noexcept {
    if (!maximum_bytes || maximum_bytes > backend::cuda::kCudaProfileArchiveMaxBytes)
        return std::nullopt;
    try {
        auto parent = open_parent(path);
        if (!parent) return std::nullopt;
        ExecutionOwnedDescriptor file{::openat(parent->directory.get(), parent->name.c_str(),
                                               O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)};
        struct stat before {
        }, after{};
        if (file.get() < 0 || ::fstat(file.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
            before.st_uid != ::geteuid() || (before.st_mode & 0077) != 0 || before.st_size <= 0 ||
            static_cast<std::uintmax_t>(before.st_size) > maximum_bytes)
            return std::nullopt;
        std::vector<std::byte> bytes(static_cast<std::size_t>(before.st_size));
        std::size_t offset{};
        while (offset < bytes.size()) {
            const auto count = ::read(file.get(), bytes.data() + offset, bytes.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) return std::nullopt;
            offset += static_cast<std::size_t>(count);
        }
        if (::fstat(file.get(), &after) != 0 || !same_file(before, after)) return std::nullopt;
        if (::close(file.release()) != 0) return std::nullopt;
        return bytes;
    } catch (...) {
        return std::nullopt;
    }
}

bool write_execution_archive(std::string_view path, std::span<const std::byte> bytes,
                             std::size_t maximum_bytes) noexcept {
    if (!maximum_bytes || maximum_bytes > backend::cuda::kCudaProfileArchiveMaxBytes ||
        bytes.empty() || !bytes.data() || bytes.size() > maximum_bytes)
        return false;
    try {
        auto parent = open_parent(path);
        if (!parent) return false;
        // Distinct exclusive names; no random dependency and no unlink of collisions.
        static std::atomic<std::uint64_t> next{1};
        for (unsigned int attempt = 0; attempt < 32; ++attempt) {
            auto sequence = next.load(std::memory_order_relaxed);
            do {
                if (sequence == std::numeric_limits<std::uint64_t>::max()) return false;
            } while (
                !next.compare_exchange_weak(sequence, sequence + 1, std::memory_order_relaxed));
            auto name =
                ".glimmer-profile-" + std::to_string(::getpid()) + "-" + std::to_string(sequence);
            ExecutionOwnedDescriptor file{
                ::openat(parent->directory.get(), name.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600)};
            if (file.get() < 0) {
                if (errno == EEXIST) continue;
                return false;
            }
            Temporary temporary{parent->directory.get(), std::move(name), false};
            std::size_t offset{};
            while (offset < bytes.size()) {
                const auto count =
                    ::write(file.get(), bytes.data() + offset, bytes.size() - offset);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) return false;
                offset += static_cast<std::size_t>(count);
            }
            if (::fsync(file.get()) != 0 || ::close(file.release()) != 0) return false;
            if (::syscall(SYS_renameat2, parent->directory.get(), temporary.name.c_str(),
                          parent->directory.get(), parent->name.c_str(), RENAME_NOREPLACE) != 0)
                return false;
            temporary.is_published = true;
            return ::fsync(parent->directory.get()) == 0;
        }
    } catch (...) {
        return false;
    }
    return false;
}
std::optional<std::vector<std::byte>> read_profile_archive(std::string_view path) noexcept {
    return read_execution_archive(path, backend::cuda::kCudaProfileArchiveMaxBytes);
}
bool write_profile_archive(std::string_view path, std::span<const std::byte> bytes) noexcept {
    return write_execution_archive(path, bytes, backend::cuda::kCudaProfileArchiveMaxBytes);
}
}  // namespace glimmer::app
