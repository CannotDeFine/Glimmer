#include "glimmer/control/shared_memory_quota.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <sys/mman.h>
#include <string>
#include <string_view>
#include <thread>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using glimmer::control::SharedMemoryQuota;
using glimmer::control::SharedMemoryQuotaConfig;
using glimmer::control::SharedMemoryQuotaError;
using glimmer::core::MemoryBytes;

struct SharedRegionMutexPrefix {
    std::uint64_t magic;
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint64_t region_size;
    std::uint32_t state;
    std::uint32_t tenant_length;
    std::uint64_t generation;
    std::uint32_t max_devices;
    std::uint32_t max_processes;
    std::uint8_t tenant_id[SharedMemoryQuota::max_tenant_id_bytes];
    pthread_mutex_t mutex;
};

static_assert(offsetof(SharedRegionMutexPrefix, mutex) > 0);

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

[[nodiscard]] std::string unique_tenant(std::string_view suffix) {
    return "shared-memory-test-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
           std::string(suffix);
}

[[nodiscard]] SharedMemoryQuotaConfig make_config(const std::string& tenant_id,
                                                  MemoryBytes limit_bytes = 100) {
    return SharedMemoryQuotaConfig{
        .tenant_id = tenant_id,
        .device = 0,
        .limit_bytes = limit_bytes,
    };
}

[[nodiscard]] std::string region_name(std::string_view tenant_id) {
    constexpr std::uint64_t k_fnv_offset = 1469598103934665603ULL;
    constexpr std::uint64_t k_fnv_prime = 1099511628211ULL;
    constexpr char k_hex_digits[] = "0123456789abcdef";
    std::uint64_t hash = k_fnv_offset;
    for (const unsigned char character : tenant_id) {
        hash ^= character;
        hash *= k_fnv_prime;
    }

    std::string name = "/glimmer-quota-v1-";
    name.reserve(name.size() + 16);
    for (std::size_t index = 0; index < 16; ++index) {
        const std::size_t shift = (15 - index) * 4;
        name.push_back(k_hex_digits[(hash >> shift) & 0x0fU]);
    }
    return name;
}

void write_result(int file_descriptor, int result) {
    const char* data = reinterpret_cast<const char*>(&result);
    std::size_t written = 0;
    while (written < sizeof(result)) {
        const ssize_t count = ::write(file_descriptor, data + written, sizeof(result) - written);
        if (count > 0) {
            written += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        _exit(EXIT_FAILURE);
    }
}

[[nodiscard]] bool read_result(int file_descriptor, int* result) {
    char* data = reinterpret_cast<char*>(result);
    std::size_t received = 0;
    while (received < sizeof(*result)) {
        const ssize_t count = ::read(file_descriptor, data + received, sizeof(*result) - received);
        if (count > 0) {
            received += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

void test_local_shared_accounting() {
    const std::string tenant_id = unique_tenant("local");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    SharedMemoryQuotaError error = SharedMemoryQuotaError::kUnknown;
    auto quota = SharedMemoryQuota::open(make_config(tenant_id), &error);
    expect(quota != nullptr && error == SharedMemoryQuotaError::kNone, "shared quota should open");
    expect(!quota->try_reserve(-1, 1).has_value(),
           "shared quota should reject a negative device identity");
    expect(quota->usage(-1).limit_bytes == 0,
           "invalid shared device usage should not expose the default quota");

    auto reservation = quota->try_reserve(60);
    expect(reservation.has_value(), "shared reservation should be admitted");
    if (!reservation.has_value()) {
        quota.reset();
        (void)SharedMemoryQuota::remove_region(tenant_id);
        return;
    }
    expect(reservation->commit(), "shared reservation should commit");
    expect(!SharedMemoryQuota::remove_region(tenant_id),
           "active shared quota regions must not be removed");
    expect(!quota->try_reserve(41).has_value(), "shared quota should reject an over-limit request");

    const auto memory_info = quota->get_memory_info(100, 90);
    expect(memory_info.total_bytes == 100, "shared quota should expose its configured total");
    expect(memory_info.free_bytes == 40, "shared quota should expose committed usage");
    const auto reduced_physical_info = quota->get_memory_info(50, 50);
    expect(reduced_physical_info.total_bytes == 50,
           "shared quota should clamp visible total to physical capacity");
    expect(reduced_physical_info.free_bytes == 0,
           "shared quota should not report free bytes below committed usage");
    expect(quota->release(60), "shared release should succeed");
    expect(quota->usage().used_bytes() == 0, "shared release should restore usage");
    expect(quota->is_healthy(), "shared quota should remain healthy after accounting operations");
    expect(SharedMemoryQuota::remove_region(tenant_id),
           "idle shared quota regions should be removable");
    const auto removed_info = quota->get_memory_info(100, 90);
    expect(!quota->is_healthy() && removed_info.total_bytes == 0 && removed_info.free_bytes == 0,
           "shared quota queries should report an unusable removed region");

    quota.reset();
}

void test_incompatible_region_is_rejected() {
    const std::string tenant_id = unique_tenant("incompatible");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    const std::string name = region_name(tenant_id);
    const int file_descriptor = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    expect(file_descriptor >= 0, "incompatible test region should be created");
    if (file_descriptor < 0) {
        return;
    }
    expect(::ftruncate(file_descriptor, 1) == 0, "incompatible region should be truncated");
    (void)::close(file_descriptor);

    SharedMemoryQuotaError error = SharedMemoryQuotaError::kNone;
    auto quota = SharedMemoryQuota::open(make_config(tenant_id), &error);
    expect(quota == nullptr && error == SharedMemoryQuotaError::kIncompatibleRegion,
           "incompatible shared region should fail closed");
    expect(SharedMemoryQuota::remove_region(tenant_id), "incompatible region should be removable");
}

void test_robust_mutex_owner_death_is_recovered() {
    const std::string tenant_id = unique_tenant("owner-death");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    auto quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(quota != nullptr, "owner-death test quota should open");
    if (quota == nullptr) {
        return;
    }

    int pipe_descriptors[2] = {-1, -1};
    expect(::pipe(pipe_descriptors) == 0, "owner-death pipe should open");
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "owner-death child should fork");
    if (child_pid == 0) {
        (void)::close(pipe_descriptors[0]);
        const std::string name = region_name(tenant_id);
        const int file_descriptor = ::shm_open(name.c_str(), O_RDWR, 0600);
        if (file_descriptor < 0) {
            write_result(pipe_descriptors[1], 1);
            _exit(EXIT_FAILURE);
        }
        void* mapping = ::mmap(nullptr, sizeof(SharedRegionMutexPrefix), PROT_READ | PROT_WRITE,
                               MAP_SHARED, file_descriptor, 0);
        if (mapping == MAP_FAILED) {
            (void)::close(file_descriptor);
            write_result(pipe_descriptors[1], 2);
            _exit(EXIT_FAILURE);
        }
        auto* prefix = static_cast<SharedRegionMutexPrefix*>(mapping);
        const int lock_result = ::pthread_mutex_lock(&prefix->mutex);
        write_result(pipe_descriptors[1], lock_result == 0 ? 0 : 3);
        // Deliberately exit while owning the robust process-shared mutex.
        _exit(lock_result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    (void)::close(pipe_descriptors[1]);
    int child_result = -1;
    expect(read_result(pipe_descriptors[0], &child_result),
           "parent should observe owner-death lock acquisition");
    (void)::close(pipe_descriptors[0]);
    int child_status = 0;
    expect(::waitpid(child_pid, &child_status, 0) == child_pid,
           "parent should wait for owner-death child");
    expect(child_result == 0 && WIFEXITED(child_status),
           "child should exit while holding the shared mutex");

    auto reservation = quota->try_reserve(100);
    expect(reservation.has_value() && reservation->commit(),
           "robust mutex owner death should be recovered safely");
    expect(quota->release(100), "owner-death recovery allocation should release");

    quota.reset();
    expect(SharedMemoryQuota::remove_region(tenant_id),
           "owner-death test region should be removable");
}

void test_processes_share_one_limit_and_recover_committed_bytes() {
    const std::string tenant_id = unique_tenant("processes");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    auto quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(quota != nullptr, "parent shared quota should open");
    auto parent_reservation = quota->try_reserve(60);
    expect(parent_reservation.has_value() && parent_reservation->commit(),
           "parent allocation should commit");

    int pipe_descriptors[2] = {-1, -1};
    expect(::pipe(pipe_descriptors) == 0, "result pipe should open");
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "child process should fork");
    if (child_pid == 0) {
        (void)::close(pipe_descriptors[0]);
        int result = 0;
        {
            auto child_quota = SharedMemoryQuota::open(make_config(tenant_id));
            if (child_quota == nullptr || child_quota->try_reserve(41).has_value()) {
                result = 1;
            } else {
                auto child_reservation = child_quota->try_reserve(40);
                if (!child_reservation.has_value() || !child_reservation->commit()) {
                    result = 2;
                }
            }
        }
        write_result(pipe_descriptors[1], result);
        (void)::close(pipe_descriptors[1]);
        _exit(result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    (void)::close(pipe_descriptors[1]);
    int child_result = -1;
    expect(read_result(pipe_descriptors[0], &child_result), "parent should read child result");
    (void)::close(pipe_descriptors[0]);
    int child_status = 0;
    expect(::waitpid(child_pid, &child_status, 0) == child_pid,
           "parent should wait for child process");
    expect(
        child_result == 0 && WIFEXITED(child_status) && WEXITSTATUS(child_status) == EXIT_SUCCESS,
        "child should observe the shared quota boundary");

    auto recovered_in_place = quota->try_reserve(40);
    expect(!recovered_in_place.has_value(),
           "committed bytes should remain reserved during recovery grace period");

    auto recovered_quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(recovered_quota != nullptr, "parent should reopen the shared quota");
    expect(recovered_quota->usage().allocated_bytes == 100,
           "dead process committed bytes should remain during recovery grace period");

    const auto recovery_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (recovered_quota->usage().allocated_bytes != 60 &&
           std::chrono::steady_clock::now() < recovery_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    expect(recovered_quota->usage().allocated_bytes == 60,
           "dead process committed bytes should recover after the grace period");

    auto recovered_reservation = recovered_quota->try_reserve(40);
    expect(recovered_reservation.has_value() && recovered_reservation->commit(),
           "recovered quota should admit the reclaimed capacity");
    expect(recovered_quota->release(40), "recovered process should release its allocation");
    expect(quota->release(60), "parent should release its allocation");

    recovered_quota.reset();
    quota.reset();
    expect(SharedMemoryQuota::remove_region(tenant_id), "test region should be removable");
}

void test_forked_state_registers_a_new_process_slot() {
    const std::string tenant_id = unique_tenant("fork-rebind");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    auto quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(quota != nullptr, "fork-rebind quota should open");
    if (quota == nullptr) {
        return;
    }

    int pipe_descriptors[2] = {-1, -1};
    expect(::pipe(pipe_descriptors) == 0, "fork-rebind pipe should open");
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "fork-rebind child should fork");
    if (child_pid == 0) {
        (void)::close(pipe_descriptors[0]);
        int result = 0;
        auto reservation = quota->try_reserve(25);
        if (!reservation.has_value() || !reservation->commit() || !quota->release(25)) {
            result = 1;
        }
        write_result(pipe_descriptors[1], result);
        (void)::close(pipe_descriptors[1]);
        _exit(result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    (void)::close(pipe_descriptors[1]);
    int child_result = -1;
    expect(read_result(pipe_descriptors[0], &child_result),
           "parent should read fork-rebind result");
    (void)::close(pipe_descriptors[0]);
    int child_status = 0;
    expect(::waitpid(child_pid, &child_status, 0) == child_pid,
           "parent should wait for fork-rebind child");
    expect(
        child_result == 0 && WIFEXITED(child_status) && WEXITSTATUS(child_status) == EXIT_SUCCESS,
        "forked quota state should use a child process slot");
    expect(quota->usage().used_bytes() == 0,
           "forked quota activity should not leave committed bytes in the parent slot");

    quota.reset();
    expect(SharedMemoryQuota::remove_region(tenant_id), "fork-rebind region should be removable");
}

void test_unhealthy_state_rejects_new_reservations() {
    const std::string tenant_id = unique_tenant("unhealthy-admission");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    auto quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(quota != nullptr, "unhealthy-admission quota should open");
    if (quota == nullptr) {
        return;
    }

    auto inherited_reservation = quota->try_reserve(10);
    expect(inherited_reservation.has_value(), "inherited reservation should be admitted");
    if (!inherited_reservation.has_value()) {
        quota.reset();
        (void)SharedMemoryQuota::remove_region(tenant_id);
        return;
    }

    int pipe_descriptors[2] = {-1, -1};
    expect(::pipe(pipe_descriptors) == 0, "unhealthy-admission pipe should open");
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "unhealthy-admission child should fork");
    if (child_pid == 0) {
        (void)::close(pipe_descriptors[0]);
        int result = 0;
        if (inherited_reservation->commit()) {
            result = 1;
        }
        if (quota->try_reserve(1).has_value()) {
            result = 2;
        }
        write_result(pipe_descriptors[1], result);
        (void)::close(pipe_descriptors[1]);
        _exit(result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }

    (void)::close(pipe_descriptors[1]);
    int child_result = -1;
    expect(read_result(pipe_descriptors[0], &child_result),
           "parent should read unhealthy-admission result");
    (void)::close(pipe_descriptors[0]);
    int child_status = 0;
    expect(::waitpid(child_pid, &child_status, 0) == child_pid,
           "parent should wait for unhealthy-admission child");
    expect(
        child_result == 0 && WIFEXITED(child_status) && WEXITSTATUS(child_status) == EXIT_SUCCESS,
        "unhealthy shared state admitted a new reservation");

    inherited_reservation->cancel();
    quota.reset();
    expect(SharedMemoryQuota::remove_region(tenant_id),
           "unhealthy-admission region should be removable");
}

void test_devices_use_separate_counters() {
    const std::string tenant_id = unique_tenant("devices");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    auto quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(quota != nullptr, "shared quota should open for device accounting");
    auto first_device_reservation = quota->try_reserve(0, 60);
    expect(first_device_reservation.has_value(), "first device reservation should be admitted");
    if (!first_device_reservation.has_value()) {
        quota.reset();
        (void)SharedMemoryQuota::remove_region(tenant_id);
        return;
    }
    expect(first_device_reservation->commit(), "first device reservation should commit");
    expect(quota->usage(1).limit_bytes == 100 && quota->usage(1).used_bytes() == 0,
           "unregistered devices should expose the configured quota before allocation");

    auto second_device_reservation = quota->try_reserve(1, 100);
    expect(second_device_reservation.has_value(), "second device should have its own quota");
    if (!second_device_reservation.has_value()) {
        quota.reset();
        (void)SharedMemoryQuota::remove_region(tenant_id);
        return;
    }
    expect(second_device_reservation->commit(), "second device reservation should commit");
    expect(quota->usage(0).allocated_bytes == 60, "device zero usage should remain independent");
    expect(quota->usage(1).allocated_bytes == 100, "device one usage should remain independent");
    expect(!quota->try_reserve(0, 41).has_value(), "device zero should enforce its own limit");
    expect(quota->release(1, 100), "device one release should succeed");
    expect(quota->release(0, 60), "device zero release should succeed");

    quota.reset();
    expect(SharedMemoryQuota::remove_region(tenant_id), "test region should be removable");
}

void test_process_recovers_an_uncommitted_reservation() {
    const std::string tenant_id = unique_tenant("reservation-crash");
    expect(SharedMemoryQuota::remove_region(tenant_id), "stale test region should be removable");

    auto quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(quota != nullptr, "parent shared quota should open");

    int pipe_descriptors[2] = {-1, -1};
    expect(::pipe(pipe_descriptors) == 0, "result pipe should open");
    const pid_t child_pid = ::fork();
    expect(child_pid >= 0, "child process should fork");
    if (child_pid == 0) {
        (void)::close(pipe_descriptors[0]);
        auto child_quota = SharedMemoryQuota::open(make_config(tenant_id));
        if (child_quota == nullptr) {
            write_result(pipe_descriptors[1], 1);
            _exit(EXIT_FAILURE);
        }
        auto child_reservation = child_quota->try_reserve(70);
        if (!child_reservation.has_value()) {
            write_result(pipe_descriptors[1], 2);
            _exit(EXIT_FAILURE);
        }
        write_result(pipe_descriptors[1], 0);
        // Deliberately skip destructors: the reservation must be recovered from
        // the dead process slot rather than cancelled by RAII.
        _exit(EXIT_SUCCESS);
    }

    (void)::close(pipe_descriptors[1]);
    int child_result = -1;
    expect(read_result(pipe_descriptors[0], &child_result), "parent should read child result");
    (void)::close(pipe_descriptors[0]);
    int child_status = 0;
    expect(::waitpid(child_pid, &child_status, 0) == child_pid,
           "parent should wait for child process");
    expect(child_result == 0 && WIFEXITED(child_status),
           "child should leave an outstanding reservation");

    auto recovered_quota = SharedMemoryQuota::open(make_config(tenant_id));
    expect(recovered_quota != nullptr, "parent should reopen the shared quota");
    expect(recovered_quota->usage().reserved_bytes == 0,
           "dead process reservations should be recovered");
    auto reservation = recovered_quota->try_reserve(100);
    expect(reservation.has_value() && reservation->commit(),
           "recovered reservation capacity should be reusable");
    expect(recovered_quota->release(100), "recovered allocation should release");

    recovered_quota.reset();
    quota.reset();
    expect(SharedMemoryQuota::remove_region(tenant_id), "test region should be removable");
}

}  // namespace

int main() {
    test_local_shared_accounting();
    test_incompatible_region_is_rejected();
    test_robust_mutex_owner_death_is_recovered();
    test_processes_share_one_limit_and_recover_committed_bytes();
    test_forked_state_registers_a_new_process_slot();
    test_unhealthy_state_rejects_new_reservations();
    test_devices_use_separate_counters();
    test_process_recovers_an_uncommitted_reservation();
    return EXIT_SUCCESS;
}
