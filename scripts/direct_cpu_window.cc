// Optional CPU-only diagnostic preload. The controller owns its pipe and files;
// the immutable main TID is the only data accessed by the sampling filter.
#include <dirent.h>
#include <fcntl.h>
#include <gperftools/profiler.h>
#include <poll.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
constexpr std::uint64_t kMagic = 0x474c435055303031ULL;
constexpr std::uint64_t kSecond = 1000000000;
struct Controller {
    pthread_t thread{};
    int pipe_fd{-1};
    bool started{};
    std::atomic<bool> profiler_owned{false};
    long main_tid{};
    std::uint64_t start_ticks{};
    std::array<char, 1024> profile{};
    std::array<char, 1024> receipt{};
};
constinit Controller controller;

[[noreturn]] void fail(const char* message) {
    if (controller.profiler_owned.exchange(false)) {
        ProfilerStop();
    }
    std::fprintf(stderr, "DIRECT_CPU_ERROR: %s\n", message);
    std::_Exit(86);
}

std::uint64_t clock_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        fail("monotonic clock failed");
    }
    return static_cast<std::uint64_t>(value.tv_sec) * kSecond +
           static_cast<std::uint64_t>(value.tv_nsec);
}

void sleep_until(std::uint64_t time_ns) {
    const timespec time{static_cast<time_t>(time_ns / kSecond),
                        static_cast<long>(time_ns % kSecond)};
    int error = 0;
    do {
        error = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &time, nullptr);
    } while (error == EINTR);
    if (error != 0) {
        fail("absolute sleep failed");
    }
}

std::uint64_t own_start_ticks() {
    const int fd = open("/proc/self/stat", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail("self identity open failed");
    }
    std::array<char, 4096> bytes{};
    const auto size = read(fd, bytes.data(), bytes.size() - 1);
    const int closed = close(fd);
    if (size <= 0 || closed != 0) {
        fail("self identity read failed");
    }
    char* tail = std::strrchr(bytes.data(), ')');
    if (tail == nullptr || tail[1] != ' ') {
        fail("self identity format failed");
    }
    tail += 2;
    for (int field = 3; field < 22; ++field) {
        tail = std::strchr(tail, ' ');
        if (tail == nullptr) {
            fail("self identity field missing");
        }
        ++tail;
    }
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(tail, &end, 10);
    if (errno != 0 || end == tail || *end != ' ' || value == 0) {
        fail("self identity start invalid");
    }
    return value;
}

int main_thread_filter(void* argument) {
#if defined(__linux__) && defined(__x86_64__)
    // A raw, register-only Linux syscall: no libc, loader, TLS or locks.
    long tid = SYS_gettid;
    asm volatile("syscall" : "+a"(tid) : : "rcx", "r11", "cc", "memory");
    return tid == *static_cast<const long*>(argument);
#else
#error "This diagnostic is qualified only for Linux x86_64"
#endif
}

std::array<std::uint64_t, 8> read_command() {
    std::array<unsigned char, 65> bytes{};
    std::size_t used = 0;
    const auto deadline = clock_ns() + 300 * kSecond;
    for (;;) {
        const auto now = clock_ns();
        if (now >= deadline) {
            fail("command timeout");
        }
        pollfd descriptor{controller.pipe_fd, POLLIN, 0};
        const auto remaining_ms = static_cast<int>((deadline - now) / 1000000 + 1);
        const int ready = poll(&descriptor, 1, remaining_ms);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready <= 0 || (descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
            fail("command poll failed");
        }
        const auto count = read(controller.pipe_fd, bytes.data() + used, bytes.size() - used);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            fail("command read failed");
        }
        if (count == 0) {
            break;
        }
        used += static_cast<std::size_t>(count);
        if (used > 64) {
            fail("extra command data");
        }
    }
    if (close(controller.pipe_fd) != 0) {
        fail("command close failed");
    }
    controller.pipe_fd = -1;
    if (used != 64) {
        fail("truncated command");
    }
    std::array<std::uint64_t, 8> command{};
    std::memcpy(command.data(), bytes.data(), 64);
    const auto now = clock_ns();
    if (command[0] != kMagic || command[1] != 1 ||
        command[2] != static_cast<std::uint64_t>(controller.main_tid) ||
        command[3] != controller.start_ticks || command[4] < now + kSecond / 10 ||
        command[4] > now + 300 * kSecond || command[5] < command[4] ||
        command[5] - command[4] != 5 * kSecond || command[6] > 1 || command[7] != 0) {
        fail("invalid command identity or window");
    }
    return command;
}

struct stat profile_stat() {
    struct stat info {};
    if (lstat(controller.profile.data(), &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_uid != getuid() || info.st_nlink != 1 || info.st_size < 64 ||
        info.st_size > 16L * 1024 * 1024) {
        fail("invalid profile file");
    }
    return info;
}

void verify_closed_file(const struct stat& info) {
    DIR* entries = opendir("/proc/self/fd");
    if (entries == nullptr) {
        fail("fd inspection failed");
    }
    for (;;) {
        errno = 0;
        const auto* entry = readdir(entries);
        if (entry == nullptr) {
            if (errno != 0) {
                fail("fd directory read failed");
            }
            break;
        }
        char* end = nullptr;
        const long fd = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0' || fd < 0 ||
            fd > std::numeric_limits<int>::max()) {
            continue;
        }
        struct stat opened {};
        if (fstat(static_cast<int>(fd), &opened) != 0) {
            if (errno == EBADF) {
                continue;
            }
            fail("fd identity failed");
        }
        if (opened.st_dev == info.st_dev && opened.st_ino == info.st_ino) {
            fail("profile still open after Stop");
        }
    }
    if (closedir(entries) != 0) {
        fail("fd directory close failed");
    }
}

void write_receipt(const std::array<std::uint64_t, 24>& receipt) {
    const int fd =
        open(controller.receipt.data(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        fail("receipt create failed");
    }
    const auto* cursor = reinterpret_cast<const char*>(receipt.data());
    std::size_t remaining = sizeof(receipt);
    while (remaining > 0) {
        const auto count = write(fd, cursor, remaining);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            fail("receipt write failed");
        }
        cursor += count;
        remaining -= static_cast<std::size_t>(count);
    }
    if (fsync(fd) != 0 || close(fd) != 0) {
        fail("receipt finalize failed");
    }
}

void* run_controller(void*) {
    const auto command = read_command();
    std::array<std::uint64_t, 24> receipt{};
    std::copy(command.begin(), command.end(), receipt.begin());
    ProfilerState state{};
    ProfilerGetCurrentState(&state);
    if (state.enabled != 0) {
        fail("profiler already enabled");
    }
    receipt[8] = clock_ns();
    sleep_until(command[4]);
    receipt[9] = clock_ns();
    ProfilerOptions options{main_thread_filter, &controller.main_tid};
    if (command[6] != 0) {
        if (ProfilerStartWithOptions(controller.profile.data(), &options) == 0) {
            fail("Start failed");
        }
        controller.profiler_owned.store(true);
    }
    receipt[10] = clock_ns();
    ProfilerGetCurrentState(&state);
    receipt[11] = static_cast<std::uint64_t>(state.enabled);
    if (receipt[11] != command[6] || receipt[10] - command[4] > 50000000 ||
        (command[6] != 0 && std::strcmp(state.profile_name, controller.profile.data()) != 0)) {
        fail("Start state or boundary failed");
    }
    sleep_until(command[5]);
    ProfilerGetCurrentState(&state);
    receipt[12] = static_cast<std::uint64_t>(state.samples_gathered);
    receipt[13] = clock_ns();
    if (command[6] != 0) {
        ProfilerStop();
    }
    receipt[14] = clock_ns();
    ProfilerGetCurrentState(&state);
    receipt[15] = static_cast<std::uint64_t>(state.enabled);
    if (state.enabled != 0 || receipt[14] - command[5] > 50000000) {
        fail("Stop state or boundary failed");
    }
    controller.profiler_owned.store(false);
    if (command[6] != 0) {
        const auto before = profile_stat();
        verify_closed_file(before);
        sleep_until(clock_ns() + 10000000);
        const auto after = profile_stat();
        if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
            before.st_size != after.st_size || before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
            before.st_mtim.tv_nsec != after.st_mtim.tv_nsec) {
            fail("profile changed after Stop");
        }
        receipt[16] = static_cast<std::uint64_t>(after.st_size);
        receipt[17] = static_cast<std::uint64_t>(after.st_dev);
        receipt[18] = static_cast<std::uint64_t>(after.st_ino);
    }
    receipt[19] = clock_ns();
    write_receipt(receipt);
    return nullptr;
}

__attribute__((constructor)) void begin_controller() {
    const char* descriptor = std::getenv("GLIMMER_CPU_COMMAND_FD");
    if (descriptor == nullptr) {
        return;
    }
    const char* directory = std::getenv("GLIMMER_CPU_DIRECTORY");
    if (directory == nullptr || directory[0] != '/' || std::getenv("CPUPROFILE") != nullptr ||
        std::getenv("CPUPROFILESIGNAL") != nullptr) {
        fail("invalid startup environment");
    }
    struct stat info {};
    if (lstat(directory, &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != getuid() ||
        (info.st_mode & 0777) != 0700) {
        fail("private output directory required");
    }
    for (auto* output : {&controller.profile, &controller.receipt}) {
        const char* name = output == &controller.profile ? "owner.cpu.prof" : "receipt.bin";
        const auto length = std::snprintf(output->data(), output->size(), "%s/%s", directory, name);
        if (length <= 0 || static_cast<std::size_t>(length) >= output->size() ||
            lstat(output->data(), &info) == 0 || errno != ENOENT) {
            fail("fresh bounded output required");
        }
    }
    char* end = nullptr;
    errno = 0;
    const long fd = std::strtol(descriptor, &end, 10);
    if (errno != 0 || end == descriptor || *end != '\0' || fd < 3 ||
        fd > std::numeric_limits<int>::max()) {
        fail("invalid command fd");
    }
    controller.pipe_fd = static_cast<int>(fd);
    if (fstat(controller.pipe_fd, &info) != 0 || !S_ISFIFO(info.st_mode) ||
        info.st_uid != getuid() || fcntl(controller.pipe_fd, F_SETFD, FD_CLOEXEC) != 0) {
        fail("owned pipe required");
    }
    controller.main_tid = getpid();
    controller.start_ticks = own_start_ticks();
    if (main_thread_filter(&controller.main_tid) == 0 ||
        pthread_create(&controller.thread, nullptr, run_controller, nullptr) != 0) {
        fail("controller initialization failed");
    }
    controller.started = true;
}

__attribute__((destructor)) void finish_controller() {
    if (controller.started && pthread_join(controller.thread, nullptr) != 0) {
        fail("controller join failed");
    }
}
}  // namespace
