#include "profile_archive.h"
#include "internal/cuda_profile_cache.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checks(const std::string& root) {
    using namespace glimmer::app;
    const std::array bytes{std::byte{1}, std::byte{0}, std::byte{255}};
    const auto path = root + "/profile";
    const auto identity = profile_runtime_identity();
    expect(identity && !identity->empty() && profile_runtime_identity() == identity,
           "local execution identity is stable across startup reads");
    expect(write_profile_archive(path, bytes), "publish a complete owned private archive");
    struct stat status {};
    expect(::stat(path.c_str(), &status) == 0 && (status.st_mode & 0777) == 0600,
           "cache files never grant group/world access");
    auto restored = read_profile_archive(path);
    expect(restored && *restored == std::vector(bytes.begin(), bytes.end()),
           "read every stored byte including embedded zeros");
    expect(!read_execution_archive(path, 0) && !read_execution_archive(path, bytes.size() - 1) &&
               !write_execution_archive(root + "/too-small", bytes, bytes.size() - 1) &&
               write_execution_archive(root + "/preparation", bytes, bytes.size()) &&
               read_execution_archive(root + "/preparation", bytes.size()) == restored,
           "preparation and measurement archives share strict pre-allocation/publication bounds");
    const std::array changed{std::byte{7}};
    expect(!write_profile_archive(path, changed) && read_profile_archive(path) == restored,
           "no-replace publication preserves existing bytes and discards its temporary");
    expect(::symlink("profile", (root + "/alias").c_str()) == 0 &&
               !read_profile_archive(root + "/alias") &&
               !write_profile_archive(root + "/alias", bytes),
           "input and output symlinks never acquire file authority");
    expect(::symlink(root.c_str(), (root + "/directory-link").c_str()) == 0 &&
               !read_profile_archive(root + "/directory-link/profile") &&
               !write_profile_archive(root + "/directory-link/new", bytes),
           "no symlink component is traversed");
    expect(::mkfifo((root + "/pipe").c_str(), 0600) == 0 && !read_profile_archive(root + "/pipe"),
           "nonblocking open rejects a FIFO without waiting for a writer");
    expect(::chmod(path.c_str(), 0644) == 0 && !read_profile_archive(path) &&
               ::chmod(path.c_str(), 0600) == 0,
           "nonprivate inputs reject before reading code identities");
    expect(::mkdir((root + "/unsafe").c_str(), 0700) == 0 &&
               ::chmod((root + "/unsafe").c_str(), 0777) == 0 &&
               !write_profile_archive(root + "/unsafe/new", bytes),
           "writable-by-others output parent rejects");
    expect(!write_profile_archive("relative", bytes) &&
               !write_profile_archive(root + "/../outside", bytes) &&
               !write_profile_archive(root + "//new", bytes) &&
               !write_profile_archive(root + "/empty", {}) &&
               !read_profile_archive(root + "/missing"),
           "ambiguous paths, empty output and missing inputs reject without side effects");
    const auto file =
        ::open((root + "/large").c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    expect(file >= 0, "create owned sparse oversized input");
    const auto sized = ::ftruncate(
        file, static_cast<off_t>(glimmer::backend::cuda::kCudaProfileArchiveMaxBytes + 1));
    const auto closed = ::close(file);
    expect(sized == 0 && closed == 0 && !read_profile_archive(root + "/large"),
           "input file bound is checked before allocation");
    auto* directory = ::opendir(root.c_str());
    expect(directory != nullptr, "inspect only the private fixture directory");
    bool has_temporary{};
    while (const auto* entry = ::readdir(directory))
        if (std::string_view(entry->d_name).starts_with(".glimmer-profile-")) has_temporary = true;
    const auto directory_closed = ::closedir(directory);
    expect(directory_closed == 0 && !has_temporary,
           "successful and failed publication leaves no temporaries");
}
}  // namespace

int main() {
    char pattern[] = "/tmp/glimmer-profile-archive-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    const std::string root{directory};
    int result = EXIT_SUCCESS;
    try {
        checks(root);
        std::cout << "Bounded private profile archive I/O checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    for (const auto* name : {"profile", "preparation", "alias", "directory-link", "pipe", "large"})
        if (::unlink((root + "/" + name).c_str()) != 0 && errno != ENOENT) result = EXIT_FAILURE;
    if (::rmdir((root + "/unsafe").c_str()) != 0 && errno != ENOENT) result = EXIT_FAILURE;
    if (::rmdir(root.c_str()) != 0) result = EXIT_FAILURE;
    return result;
}
