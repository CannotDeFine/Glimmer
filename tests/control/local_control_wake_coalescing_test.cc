#include "internal/local_control_wake.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <future>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace {
using namespace glimmer::control::internal;
using namespace std::chrono_literals;
using Clock = OwnedControlExchange::Clock;
enum class SendMode : std::uint8_t { kNormal, kBlocked, kFill, kFail };
std::atomic<SendMode> mode{SendMode::kNormal};
std::atomic<std::size_t> sends{};
std::atomic<std::size_t> filled{};
std::atomic<bool> is_blocked{};
std::atomic<bool> is_released{};
std::atomic<bool> should_block_receive{};
std::atomic<bool> is_receive_blocked{};
std::atomic<bool> is_receive_released{};

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::size_t descriptors() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "owned descriptor inventory opens");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "owned descriptor inventory closes");
    return count;
}

void test_duplicates_and_rearming() {
    LocalControlWake wake;
    sends.store(0);
    require(wake.notify(), "first owned activity publishes a real byte");
    std::array<std::thread, 4> producers;
    for (auto& producer : producers)
        producer = std::thread([&] {
            for (std::size_t count = 0; count < 1000; ++count)
                require(wake.notify(), "concurrent duplicate activity remains accepted");
        });
    for (auto& producer : producers) producer.join();
    require(sends.load() == 1, "coalesced duplicates must execute exactly one real send");
    require(wake.wait_until({Clock::now() + 1s, {}}).has_activity,
            "one real byte requests complete state reinspection");
    require(wake.wait_until({Clock::now(), {}}).status == ControlWaitStatus::kDue,
            "a consumed hint cannot fabricate a second activity");
    require(
        wake.notify() && sends.load() == 2 && wake.wait_until({Clock::now() + 1s, {}}).has_activity,
        "consumption rearms the next actual publication");
    std::atomic<std::size_t> published{};
    std::atomic<std::size_t> inspected{};
    std::thread producer([&] {
        const auto limit = Clock::now() + 5s;
        for (std::size_t count = 1; count <= 2000; ++count) {
            while (inspected.load(std::memory_order_acquire) != count - 1) {
                require(Clock::now() < limit, "publication/inspection producer terminates");
                std::this_thread::yield();
            }
            published.store(count, std::memory_order_release);
            require(wake.notify(), "new state publication requires an observable hint");
        }
    });
    for (std::size_t count = 1; count <= 2000; ++count) {
        require(wake.wait_until({Clock::now() + 1s, {}}).has_activity &&
                    published.load(std::memory_order_acquire) == count,
                "racing consumer always reinspects the actual published state");
        inspected.store(count, std::memory_order_release);
    }
    producer.join();
    require(sends.load() == 2002, "each consumed generation requires exactly one fresh send");
}

void test_incomplete_send() {
    LocalControlWake wake;
    sends.store(0);
    is_blocked.store(false);
    is_released.store(false);
    mode.store(SendMode::kBlocked);
    std::thread first([&] { require(wake.notify(), "blocked winner completes its real send"); });
    const auto limit = Clock::now() + 1s;
    while (!is_blocked.load(std::memory_order_acquire)) {
        require(Clock::now() < limit, "first send enters the owned test barrier");
        std::this_thread::yield();
    }
    require(wake.notify() && sends.load() == 1, "duplicate coalesces while winner is sending");
    auto consumer =
        std::async(std::launch::async, [&] { return wake.wait_until({Clock::now() + 1s, {}}); });
    require(consumer.wait_for(2ms) == std::future_status::timeout,
            "pending latch without real bytes cannot fabricate readiness");
    is_released.store(true, std::memory_order_release);
    first.join();
    require(consumer.get().has_activity && sends.load() == 1,
            "actual winner byte wakes the waiting consumer");
    require(
        wake.notify() && sends.load() == 2 && wake.wait_until({Clock::now() + 1s, {}}).has_activity,
        "incomplete-send consumption safely rearms");
}

void test_real_backpressure_and_failure() {
    {
        LocalControlWake wake;
        filled.store(0);
        mode.store(SendMode::kFill);
        require(wake.notify() && filled.load() > 0,
                "actual full socket EAGAIN is readable activity");
        std::size_t drains = 0;
        for (;;) {
            const auto outcome = wake.wait_until({Clock::now(), {}});
            if (outcome.status == ControlWaitStatus::kDue) break;
            require(outcome.has_activity && ++drains <= 8192,
                    "actual buffered W bytes drain in bounded calls without authority");
        }
        require(
            drains > 0 && wake.notify() && wake.wait_until({Clock::now() + 1s, {}}).has_activity,
            "real buffer saturation cannot lose a later publication");
    }
    {
        LocalControlWake wake;
        sends.store(0);
        mode.store(SendMode::kFail);
        require(!wake.notify() && !wake.notify() && sends.load() == 1 &&
                    wake.wait_until({Clock::now(), {}}).status == ControlWaitStatus::kError,
                "failed winning send remains sticky and coalescing cannot hide it");
    }
}

void test_read_rearm_window() {
    LocalControlWake wake;
    sends.store(0);
    std::atomic<unsigned> published{1};
    require(wake.notify(), "first read/rearm state publishes a real byte");
    is_receive_blocked.store(false);
    is_receive_released.store(false);
    should_block_receive.store(true);
    auto consumer = std::async(std::launch::async, [&] {
        const auto outcome = wake.wait_until({Clock::now() + 1s, {}});
        return std::pair{outcome, published.load(std::memory_order_acquire)};
    });
    const auto limit = Clock::now() + 1s;
    while (!is_receive_blocked.load(std::memory_order_acquire)) {
        require(Clock::now() < limit, "owned positive receive enters read/rearm barrier");
        std::this_thread::yield();
    }
    published.store(2, std::memory_order_release);
    const bool is_coalesced = wake.notify() && sends.load() == 1;
    is_receive_released.store(true, std::memory_order_release);
    const auto [outcome, inspected] = consumer.get();
    require(is_coalesced && outcome.has_activity && inspected == 2,
            "read/rearm window must coalesce before complete state reinspection");
    require(wake.wait_until({Clock::now(), {}}).status == ControlWaitStatus::kDue,
            "second state is covered by inspection, not an invented extra byte");
    published.store(3, std::memory_order_release);
    require(wake.notify() && sends.load() == 2 &&
                wake.wait_until({Clock::now() + 1s, {}}).has_activity &&
                published.load(std::memory_order_acquire) == 3,
            "publication after rearm has a new real byte and observable state");
}

void test_pending_fork_and_close() {
    LocalControlWake wake;
    sends.store(0);
    require(wake.notify(), "parent owns a pending real byte");
    const pid_t child = ::fork();
    require(child >= 0, "owned child creates");
    if (child == 0) {
        const bool is_rejected =
            !wake.notify() && sends.load() == 1 &&
            wake.wait_until({Clock::now(), {}}).status == ControlWaitStatus::kRejected &&
            wake.close();
        ::_exit(is_rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "child rejects before latch/I/O and is reaped");
    require(
        wake.notify() && sends.load() == 1 && wake.wait_until({Clock::now() + 1s, {}}).has_activity,
        "child close cannot consume or shut down the parent's pending byte");
    require(wake.close() && wake.close() && !wake.notify() &&
                wake.wait_until({Clock::now(), {}}).status == ControlWaitStatus::kError,
            "closed wake rejects before coalescing and retires idempotently");
}
}  // namespace

extern "C" ssize_t real_local_send(int descriptor, const void* bytes, size_t size,
                                   int flags) __asm__("__real_send");
extern "C" ssize_t wrapped_local_send(int descriptor, const void* bytes, size_t size,
                                      int flags) __asm__("__wrap_send");
extern "C" ssize_t wrapped_local_send(int descriptor, const void* bytes, size_t size, int flags) {
    require(size == 1 && *static_cast<const char*>(bytes) == 'W' &&
                flags == (MSG_DONTWAIT | MSG_NOSIGNAL),
            "wrapper intercepts only fixture-owned exact local wake sends");
    sends.fetch_add(1);
    const auto selected = mode.exchange(SendMode::kNormal);
    if (selected == SendMode::kBlocked) {
        is_blocked.store(true, std::memory_order_release);
        const auto limit = Clock::now() + 2s;
        while (!is_released.load(std::memory_order_acquire)) {
            require(Clock::now() < limit, "owned send barrier releases within finite bound");
            std::this_thread::yield();
        }
    } else if (selected == SendMode::kFill) {
        for (std::size_t count = 0; count < 8192; ++count) {
            const auto result = real_local_send(descriptor, bytes, size, flags);
            if (result == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) return result;
            require(result == 1, "owned real buffer fill succeeds or reaches EAGAIN");
            filled.fetch_add(1);
        }
        require(false, "owned buffer fill must establish actual saturation");
    } else if (selected == SendMode::kFail) {
        errno = EIO;
        return -1;
    }
    return real_local_send(descriptor, bytes, size, flags);
}

extern "C" ssize_t real_local_receive(int descriptor, void* bytes, size_t size,
                                      int flags) __asm__("__real_recv");
extern "C" ssize_t wrapped_local_receive(int descriptor, void* bytes, size_t size,
                                         int flags) __asm__("__wrap_recv");
extern "C" ssize_t wrapped_local_receive(int descriptor, void* bytes, size_t size, int flags) {
    require(size == 64 && flags == MSG_DONTWAIT,
            "receive wrapper observes only fixture-owned exact nonblocking wake reads");
    const auto result = real_local_receive(descriptor, bytes, size, flags);
    if (should_block_receive.exchange(false)) {
        require(result > 0, "receive barrier follows a real positive kernel read");
        is_receive_blocked.store(true, std::memory_order_release);
        const auto limit = Clock::now() + 2s;
        while (!is_receive_released.load(std::memory_order_acquire)) {
            require(Clock::now() < limit, "owned read/rearm barrier releases within finite bound");
            std::this_thread::yield();
        }
    }
    return result;
}

int main(int argc, char** argv) {
    const bool is_boundary_only = argc == 2 && std::strcmp(argv[1], "--read-rearm-only") == 0;
    require(argc == 1 || is_boundary_only, "only the explicit owned boundary selector is valid");
    const auto before = descriptors();
    if (!is_boundary_only) {
        test_duplicates_and_rearming();
        test_incomplete_send();
        test_real_backpressure_and_failure();
        test_pending_fork_and_close();
    }
    test_read_rearm_window();
    require(descriptors() == before, "all fixture-owned descriptors retire");
}
