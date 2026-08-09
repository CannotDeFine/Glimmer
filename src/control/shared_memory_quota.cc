#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "glimmer/control/shared_memory_quota.h"

#include <atomic>
#include <algorithm>
#include <array>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <unordered_set>

namespace glimmer::control::detail {

namespace {

constexpr std::uint64_t kRegionMagic = 0x474C494D4D455201ULL;
constexpr std::uint32_t kRegionVersion = 2;
constexpr std::uint32_t kRegionInitializing = 1;
constexpr std::uint32_t kRegionReady = 2;
constexpr std::uint32_t kRegionRemoving = 3;
constexpr std::uint32_t kDeviceFree = 0;
constexpr std::uint32_t kDeviceActive = 1;
constexpr std::uint32_t kProcessFree = 0;
constexpr std::uint32_t kProcessActive = 1;
constexpr std::uint32_t kProcessOrphaned = 2;
constexpr std::size_t kMaxDevices = 16;
constexpr std::size_t kMaxProcessSlots = 64;
constexpr std::uint64_t kCommittedRecoveryGraceMilliseconds = 5000;

struct DeviceQuotaRecord {
    std::uint32_t state;
    DeviceId device;
    core::MemoryBytes limit_bytes;
    core::MemoryBytes reserved_bytes;
    core::MemoryBytes committed_bytes;
};

struct ProcessQuotaSlot {
    std::uint32_t state;
    std::int32_t process_id;
    std::uint64_t process_start_time;
    std::uint64_t generation;
    std::uint64_t heartbeat;
    std::uint64_t orphaned_since_milliseconds;
    core::MemoryBytes reserved_bytes[kMaxDevices];
    core::MemoryBytes committed_bytes[kMaxDevices];
};

struct SharedRegion {
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
    DeviceQuotaRecord devices[kMaxDevices];
    ProcessQuotaSlot processes[kMaxProcessSlots];
};

struct ReservationToken {
    std::uint32_t process_slot = 0;
    std::uint32_t device_slot = 0;
    std::uint64_t process_generation = 0;
    std::uint64_t reservation_id = 0;
    core::MemoryBytes memory_bytes = 0;
};

[[nodiscard]] bool checked_add(core::MemoryBytes left, core::MemoryBytes right,
                               core::MemoryBytes* result) noexcept {
    if (right > std::numeric_limits<core::MemoryBytes>::max() - left) {
        return false;
    }
    *result = left + right;
    return true;
}

[[nodiscard]] std::uint64_t monotonic_ticks() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(now.count());
}

[[nodiscard]] std::uint64_t monotonic_milliseconds() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

[[nodiscard]] bool read_process_start_time(pid_t process_id, std::uint64_t* start_time) noexcept {
    if (process_id <= 0 || start_time == nullptr) {
        return false;
    }

    try {
        std::ifstream stat_file("/proc/" + std::to_string(process_id) + "/stat");
        if (!stat_file) {
            return false;
        }

        std::string contents;
        if (!std::getline(stat_file, contents)) {
            return false;
        }

        const std::size_t command_end = contents.rfind(')');
        if (command_end == std::string::npos || command_end + 2 > contents.size()) {
            return false;
        }

        std::istringstream fields(contents.substr(command_end + 2));
        std::string field;
        for (std::size_t index = 0; index <= 19; ++index) {
            if (!(fields >> field)) {
                return false;
            }
            if (index != 19) {
                continue;
            }

            const auto [end, error] =
                std::from_chars(field.data(), field.data() + field.size(), *start_time);
            return error == std::errc{} && end == field.data() + field.size();
        }
    } catch (...) {
        return false;
    }
    return false;
}

struct ProcessIdentity {
    std::int32_t process_id = 0;
    std::uint64_t process_start_time = 0;
    std::uint64_t generation = 0;
};

[[nodiscard]] bool process_is_dead(const ProcessIdentity& identity) noexcept {
    if (identity.process_id <= 0 || identity.process_start_time == 0 || identity.generation == 0) {
        return false;
    }

    if (::kill(static_cast<pid_t>(identity.process_id), 0) != 0) {
        return errno == ESRCH;
    }

    std::uint64_t current_start_time = 0;
    if (!read_process_start_time(static_cast<pid_t>(identity.process_id), &current_start_time)) {
        return false;
    }
    return current_start_time != identity.process_start_time;
}

[[nodiscard]] bool valid_region(const SharedRegion& region, std::string_view tenant_id) noexcept {
    if (region.magic != kRegionMagic || region.version != kRegionVersion ||
        region.header_size != offsetof(SharedRegion, mutex) ||
        region.region_size != sizeof(SharedRegion) || region.state != kRegionReady ||
        region.max_devices != kMaxDevices || region.max_processes != kMaxProcessSlots ||
        region.tenant_length != tenant_id.size() ||
        tenant_id.size() > SharedMemoryQuota::max_tenant_id_bytes) {
        return false;
    }
    return std::memcmp(region.tenant_id, tenant_id.data(), tenant_id.size()) == 0;
}

[[nodiscard]] bool valid_accounting(const SharedRegion& region) noexcept {
    std::array<core::MemoryBytes, kMaxDevices> process_reserved{};
    std::array<core::MemoryBytes, kMaxDevices> process_committed{};
    for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
        const auto& device = region.devices[device_index];
        if (device.state == kDeviceFree) {
            if (device.reserved_bytes != 0 || device.committed_bytes != 0) {
                return false;
            }
            continue;
        }
        if (device.state != kDeviceActive) {
            return false;
        }
        core::MemoryBytes used_bytes = 0;
        if (!checked_add(device.reserved_bytes, device.committed_bytes, &used_bytes) ||
            used_bytes > device.limit_bytes) {
            return false;
        }
    }

    for (const auto& process : region.processes) {
        if (process.state == kProcessFree) {
            for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
                if (process.reserved_bytes[device_index] != 0 ||
                    process.committed_bytes[device_index] != 0) {
                    return false;
                }
            }
            continue;
        }
        if (process.state != kProcessActive && process.state != kProcessOrphaned) {
            return false;
        }
        if (process.state == kProcessActive && process.orphaned_since_milliseconds != 0) {
            return false;
        }
        if (process.process_id <= 0 || process.process_start_time == 0 || process.generation == 0) {
            return false;
        }
        for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
            const auto& device = region.devices[device_index];
            if (device.state != kDeviceActive) {
                if (process.reserved_bytes[device_index] != 0 ||
                    process.committed_bytes[device_index] != 0) {
                    return false;
                }
                continue;
            }
            if (process.reserved_bytes[device_index] > device.reserved_bytes ||
                process.committed_bytes[device_index] > device.committed_bytes) {
                return false;
            }
            if (!checked_add(process_reserved[device_index], process.reserved_bytes[device_index],
                             &process_reserved[device_index]) ||
                !checked_add(process_committed[device_index], process.committed_bytes[device_index],
                             &process_committed[device_index])) {
                return false;
            }
        }
    }
    for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
        const auto& device = region.devices[device_index];
        if (device.state == kDeviceActive &&
            (process_reserved[device_index] != device.reserved_bytes ||
             process_committed[device_index] != device.committed_bytes)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool lock_region(SharedRegion& region) noexcept {
    const auto region_is_valid = [&region] {
        return region.tenant_length <= SharedMemoryQuota::max_tenant_id_bytes &&
               valid_region(region,
                            std::string_view(reinterpret_cast<const char*>(region.tenant_id),
                                             region.tenant_length)) &&
               valid_accounting(region);
    };
    const int lock_result = ::pthread_mutex_lock(&region.mutex);
    if (lock_result == 0) {
        if (region_is_valid()) {
            return true;
        }
        if (::pthread_mutex_unlock(&region.mutex) != 0) {
            return false;
        }
        return false;
    }
    if (lock_result != EOWNERDEAD) {
        return false;
    }
    if (!region_is_valid() || ::pthread_mutex_consistent(&region.mutex) != 0) {
        if (::pthread_mutex_unlock(&region.mutex) != 0) {
            return false;
        }
        return false;
    }
    return true;
}

[[nodiscard]] bool unlock_region(SharedRegion& region) noexcept {
    return ::pthread_mutex_unlock(&region.mutex) == 0;
}

[[nodiscard]] std::string make_region_name(std::string_view tenant_id) {
    constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
    constexpr std::uint64_t fnv_prime = 1099511628211ULL;
    std::uint64_t hash = fnv_offset;
    for (const unsigned char character : tenant_id) {
        hash ^= character;
        hash *= fnv_prime;
    }

    constexpr char hex_digits[] = "0123456789abcdef";
    std::string name = "/glimmer-quota-v1-";
    name.reserve(name.size() + 16);
    for (std::size_t index = 0; index < 16; ++index) {
        const std::size_t shift = (15 - index) * 4;
        name.push_back(hex_digits[(hash >> shift) & 0x0fU]);
    }
    return name;
}

struct OpenRegion {
    int file_descriptor = -1;
    void* mapping = nullptr;
    bool created = false;
    std::string name;
};

[[nodiscard]] bool close_open_region(OpenRegion& region, bool unlink_created) noexcept {
    bool cleanup_succeeded = true;
    if (region.mapping != nullptr && region.mapping != MAP_FAILED) {
        if (::munmap(region.mapping, sizeof(SharedRegion)) != 0) {
            cleanup_succeeded = false;
        }
        region.mapping = nullptr;
    }
    if (region.file_descriptor >= 0) {
        if (::close(region.file_descriptor) != 0) {
            cleanup_succeeded = false;
        }
        region.file_descriptor = -1;
    }
    if (unlink_created && region.created && !region.name.empty()) {
        if (::shm_unlink(region.name.c_str()) != 0 && errno != ENOENT) {
            cleanup_succeeded = false;
        }
    }
    return cleanup_succeeded;
}

[[nodiscard]] bool lock_file(int file_descriptor) noexcept {
    while (::flock(file_descriptor, LOCK_EX) != 0) {
        if (errno != EINTR) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool unlock_file(int file_descriptor) noexcept {
    return ::flock(file_descriptor, LOCK_UN) == 0;
}

[[nodiscard]] bool initialize_region(SharedRegion& region, std::string_view tenant_id) noexcept {
    std::memset(&region, 0, sizeof(region));
    region.magic = kRegionMagic;
    region.version = kRegionVersion;
    region.header_size = offsetof(SharedRegion, mutex);
    region.region_size = sizeof(SharedRegion);
    region.state = kRegionInitializing;
    region.tenant_length = static_cast<std::uint32_t>(tenant_id.size());
    region.generation = 1;
    region.max_devices = kMaxDevices;
    region.max_processes = kMaxProcessSlots;
    std::memcpy(region.tenant_id, tenant_id.data(), tenant_id.size());

    pthread_mutexattr_t attributes{};
    if (::pthread_mutexattr_init(&attributes) != 0) {
        return false;
    }

    const bool attributes_ready =
        ::pthread_mutexattr_setpshared(&attributes, PTHREAD_PROCESS_SHARED) == 0 &&
        ::pthread_mutexattr_setrobust(&attributes, PTHREAD_MUTEX_ROBUST) == 0;
    const bool mutex_ready =
        attributes_ready && ::pthread_mutex_init(&region.mutex, &attributes) == 0;
    const bool attributes_destroyed = ::pthread_mutexattr_destroy(&attributes) == 0;
    if (!mutex_ready || !attributes_destroyed) {
        if (mutex_ready && ::pthread_mutex_destroy(&region.mutex) != 0) {
            return false;
        }
        return false;
    }

    region.state = kRegionReady;
    return true;
}

[[nodiscard]] std::optional<OpenRegion> open_region(const SharedMemoryQuotaConfig& config,
                                                    SharedMemoryQuotaError* error) {
    if (config.tenant_id.empty() ||
        config.tenant_id.size() > SharedMemoryQuota::max_tenant_id_bytes || config.device < 0) {
        if (error != nullptr) {
            *error = config.tenant_id.size() > SharedMemoryQuota::max_tenant_id_bytes
                         ? SharedMemoryQuotaError::kNameTooLong
                         : SharedMemoryQuotaError::kInvalidConfiguration;
        }
        return std::nullopt;
    }

    OpenRegion region;
    try {
        region.name = make_region_name(config.tenant_id);
    } catch (...) {
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kUnknown;
        }
        return std::nullopt;
    }

    region.file_descriptor = ::shm_open(region.name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (region.file_descriptor >= 0) {
        region.created = true;
    } else if (errno == EEXIST) {
        region.file_descriptor = ::shm_open(region.name.c_str(), O_RDWR, 0600);
    }
    if (region.file_descriptor < 0) {
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kOpenFailed;
        }
        return std::nullopt;
    }

    if (!lock_file(region.file_descriptor)) {
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kLockFailed;
        }
        if (!close_open_region(region, true) && error != nullptr) {
            *error = SharedMemoryQuotaError::kUnknown;
        }
        return std::nullopt;
    }

    bool success = false;
    if (region.created) {
        if (::ftruncate(region.file_descriptor, static_cast<off_t>(sizeof(SharedRegion))) != 0) {
            if (error != nullptr) {
                *error = SharedMemoryQuotaError::kResizeFailed;
            }
        } else {
            region.mapping = ::mmap(nullptr, sizeof(SharedRegion), PROT_READ | PROT_WRITE,
                                    MAP_SHARED, region.file_descriptor, 0);
            if (region.mapping == MAP_FAILED) {
                if (error != nullptr) {
                    *error = SharedMemoryQuotaError::kMapFailed;
                }
            } else {
                success = initialize_region(*static_cast<SharedRegion*>(region.mapping),
                                            config.tenant_id);
                if (!success && error != nullptr) {
                    *error = SharedMemoryQuotaError::kInitializationFailed;
                }
            }
        }
    } else {
        struct stat file_status {};
        if (::fstat(region.file_descriptor, &file_status) != 0 ||
            file_status.st_size != static_cast<off_t>(sizeof(SharedRegion))) {
            if (error != nullptr) {
                *error = SharedMemoryQuotaError::kIncompatibleRegion;
            }
        } else {
            region.mapping = ::mmap(nullptr, sizeof(SharedRegion), PROT_READ | PROT_WRITE,
                                    MAP_SHARED, region.file_descriptor, 0);
            if (region.mapping == MAP_FAILED) {
                if (error != nullptr) {
                    *error = SharedMemoryQuotaError::kMapFailed;
                }
            } else if (!valid_region(*static_cast<SharedRegion*>(region.mapping),
                                     config.tenant_id)) {
                if (error != nullptr) {
                    *error = SharedMemoryQuotaError::kIncompatibleRegion;
                }
            } else {
                success = true;
            }
        }
    }

    if (!unlock_file(region.file_descriptor)) {
        success = false;
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kLockFailed;
        }
    }
    if (!success) {
        if (!close_open_region(region, true) && error != nullptr) {
            *error = SharedMemoryQuotaError::kUnknown;
        }
        return std::nullopt;
    }
    return region;
}

[[nodiscard]] std::optional<std::size_t> find_device_slot(const SharedRegion& region,
                                                          DeviceId device) noexcept {
    for (std::size_t index = 0; index < kMaxDevices; ++index) {
        if (region.devices[index].state == kDeviceActive &&
            region.devices[index].device == device) {
            return index;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::size_t> find_free_device_slot(
    const SharedRegion& region) noexcept {
    for (std::size_t index = 0; index < kMaxDevices; ++index) {
        if (region.devices[index].state == kDeviceFree) {
            return index;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::size_t> find_free_process_slot(
    const SharedRegion& region) noexcept {
    for (std::size_t index = 0; index < kMaxProcessSlots; ++index) {
        if (region.processes[index].state == kProcessFree) {
            return index;
        }
    }
    return std::nullopt;
}

void clear_process_slot(ProcessQuotaSlot& slot) noexcept {
    std::memset(&slot, 0, sizeof(slot));
}

void snapshot_processes(const SharedRegion& region,
                        std::array<ProcessIdentity, kMaxProcessSlots>* snapshot) noexcept {
    for (std::size_t index = 0; index < kMaxProcessSlots; ++index) {
        const auto& process = region.processes[index];
        if (process.state == kProcessActive || process.state == kProcessOrphaned) {
            (*snapshot)[index] = ProcessIdentity{
                .process_id = process.process_id,
                .process_start_time = process.process_start_time,
                .generation = process.generation,
            };
        }
    }
}

[[nodiscard]] bool reclaim_dead_processes(
    SharedRegion& region, const std::array<ProcessIdentity, kMaxProcessSlots>& snapshot,
    std::uint64_t now_milliseconds) noexcept {
    bool accounting_valid = true;
    for (std::size_t index = 0; index < kMaxProcessSlots; ++index) {
        const ProcessIdentity& identity = snapshot[index];
        if (identity.process_id <= 0 || identity.process_start_time == 0 ||
            identity.generation == 0) {
            continue;
        }

        auto& process = region.processes[index];
        if ((process.state != kProcessActive && process.state != kProcessOrphaned) ||
            process.process_id != identity.process_id ||
            process.process_start_time != identity.process_start_time ||
            process.generation != identity.generation) {
            continue;
        }

        bool process_counters_valid = true;
        for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
            auto& device = region.devices[device_index];
            const core::MemoryBytes reserved = process.reserved_bytes[device_index];
            const core::MemoryBytes committed = process.committed_bytes[device_index];
            if (device.state == kDeviceActive) {
                if (device.reserved_bytes < reserved || device.committed_bytes < committed) {
                    process_counters_valid = false;
                    break;
                }
            } else if (reserved != 0 || committed != 0) {
                process_counters_valid = false;
            }
        }
        if (!process_counters_valid) {
            accounting_valid = false;
            continue;
        }

        if (process.orphaned_since_milliseconds == 0) {
            process.state = kProcessOrphaned;
            process.orphaned_since_milliseconds = now_milliseconds;
            for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
                auto& device = region.devices[device_index];
                if (device.state != kDeviceActive) {
                    continue;
                }
                device.reserved_bytes -= process.reserved_bytes[device_index];
                process.reserved_bytes[device_index] = 0;
            }
            bool has_committed_bytes = false;
            for (const core::MemoryBytes bytes : process.committed_bytes) {
                has_committed_bytes = has_committed_bytes || bytes != 0;
            }
            if (!has_committed_bytes) {
                clear_process_slot(process);
            }
            continue;
        }

        if (now_milliseconds < process.orphaned_since_milliseconds ||
            now_milliseconds - process.orphaned_since_milliseconds <
                kCommittedRecoveryGraceMilliseconds) {
            continue;
        }

        for (std::size_t device_index = 0; device_index < kMaxDevices; ++device_index) {
            auto& device = region.devices[device_index];
            if (device.state != kDeviceActive) {
                continue;
            }
            device.reserved_bytes -= process.reserved_bytes[device_index];
            device.committed_bytes -= process.committed_bytes[device_index];
        }
        clear_process_slot(process);
    }
    return accounting_valid;
}

[[nodiscard]] bool recover_dead_processes(SharedRegion& region) noexcept {
    if (!lock_region(region)) {
        return false;
    }

    std::array<ProcessIdentity, kMaxProcessSlots> process_snapshot{};
    snapshot_processes(region, &process_snapshot);
    if (!unlock_region(region)) {
        return false;
    }

    std::array<ProcessIdentity, kMaxProcessSlots> dead_processes{};
    for (std::size_t index = 0; index < kMaxProcessSlots; ++index) {
        if (process_is_dead(process_snapshot[index])) {
            dead_processes[index] = process_snapshot[index];
        }
    }

    if (!lock_region(region)) {
        return false;
    }
    const bool accounting_valid =
        reclaim_dead_processes(region, dead_processes, monotonic_milliseconds());
    const bool unlocked = unlock_region(region);
    return accounting_valid && unlocked;
}

}  // namespace

struct SharedQuotaState {
    int file_descriptor = -1;
    void* mapping = nullptr;
    SharedRegion* region = nullptr;
    std::string region_name;
    std::uint32_t process_slot = 0;
    std::uint64_t process_generation = 0;
    pid_t process_id = 0;
    std::uint64_t process_start_time = 0;
    SharedMemoryQuotaConfig config;
    std::atomic<std::uint64_t> next_reservation_id{1};
    mutable std::atomic<std::uint64_t> last_recovery_milliseconds{0};
    // This mutex is process-local. Replace the inherited instance after a
    // fork before touching it; a child must never wait on a lock held by a
    // thread that no longer exists.
    std::unique_ptr<std::mutex> reservation_mutex = std::make_unique<std::mutex>();
    std::unordered_set<std::uint64_t> active_reservation_ids;
    pid_t local_process_id = 0;
    core::MemoryBytes default_limit_bytes = 0;
    mutable std::atomic<bool> is_healthy_{true};

    ~SharedQuotaState();

    [[nodiscard]] bool register_process(const SharedMemoryQuotaConfig& config,
                                        SharedMemoryQuotaError* error) noexcept;
    [[nodiscard]] std::optional<ReservationToken> reserve(DeviceId device,
                                                          core::MemoryBytes memory_bytes) noexcept;
    [[nodiscard]] bool commit(const ReservationToken& token) noexcept;
    [[nodiscard]] bool cancel(const ReservationToken& token) noexcept;
    [[nodiscard]] bool release(DeviceId device, core::MemoryBytes memory_bytes) noexcept;
    [[nodiscard]] core::QuotaUsage usage(DeviceId device) const noexcept;
    [[nodiscard]] bool is_healthy() const noexcept;
    void mark_unhealthy() const noexcept;
    [[nodiscard]] bool is_owner_locked() const noexcept;

   private:
    [[nodiscard]] bool recover_dead_processes_if_due() const noexcept;
    [[nodiscard]] bool ensure_process_identity() noexcept;
    [[nodiscard]] bool track_reservation(std::uint64_t reservation_id) noexcept;
    [[nodiscard]] std::optional<ReservationToken> reserve_impl(DeviceId device,
                                                               core::MemoryBytes memory_bytes,
                                                               bool recovery_attempted) noexcept;
    [[nodiscard]] bool commit_counters(const ReservationToken& token) noexcept;
    [[nodiscard]] bool cancel_counters(const ReservationToken& token) noexcept;
};

SharedQuotaState::~SharedQuotaState() {
    if (process_id != ::getpid()) {
        // A child that never touched the quota must not destroy a mutex object
        // inherited while it may have been held by a vanished parent thread.
        const bool inherited_mutex_present = reservation_mutex.release() != nullptr;
        if (!inherited_mutex_present) {
            is_healthy_.store(false, std::memory_order_relaxed);
        }
    }
    if (process_id == ::getpid() && region != nullptr && lock_region(*region)) {
        if (process_slot < kMaxProcessSlots) {
            auto& slot = region->processes[process_slot];
            if (slot.state == kProcessActive && slot.generation == process_generation &&
                slot.process_id == process_id && slot.process_start_time == process_start_time) {
                bool has_usage = false;
                for (std::size_t index = 0; index < kMaxDevices; ++index) {
                    has_usage = has_usage || slot.reserved_bytes[index] != 0 ||
                                slot.committed_bytes[index] != 0;
                }
                if (has_usage) {
                    slot.state = kProcessOrphaned;
                    slot.orphaned_since_milliseconds = 0;
                } else {
                    clear_process_slot(slot);
                }
            }
        }
        if (!unlock_region(*region)) {
            is_healthy_.store(false, std::memory_order_relaxed);
        }
    }
    if (mapping != nullptr && mapping != MAP_FAILED) {
        if (::munmap(mapping, sizeof(SharedRegion)) != 0) {
            is_healthy_.store(false, std::memory_order_relaxed);
        }
    }
    if (file_descriptor >= 0) {
        if (::close(file_descriptor) != 0) {
            is_healthy_.store(false, std::memory_order_relaxed);
        }
    }
}

bool SharedQuotaState::register_process(const SharedMemoryQuotaConfig& config,
                                        SharedMemoryQuotaError* error) noexcept {
    process_id = ::getpid();
    local_process_id = process_id;
    default_limit_bytes = config.limit_bytes;
    if (!read_process_start_time(process_id, &process_start_time)) {
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kProcessIdentityFailed;
        }
        return false;
    }
    if (region == nullptr || !recover_dead_processes(*region) || !lock_region(*region)) {
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kLockFailed;
        }
        return false;
    }
    const auto existing_device = find_device_slot(*region, config.device);
    std::size_t device_slot = 0;
    bool created_device = false;
    if (existing_device.has_value()) {
        device_slot = *existing_device;
        if (region->devices[device_slot].limit_bytes != config.limit_bytes) {
            const bool unlocked = unlock_region(*region);
            if (!unlocked) {
                mark_unhealthy();
            }
            if (error != nullptr) {
                *error = unlocked ? SharedMemoryQuotaError::kDeviceLimitMismatch
                                  : SharedMemoryQuotaError::kLockFailed;
            }
            return false;
        }
    } else {
        const auto free_device = find_free_device_slot(*region);
        if (!free_device.has_value()) {
            const bool unlocked = unlock_region(*region);
            if (!unlocked) {
                mark_unhealthy();
            }
            if (error != nullptr) {
                *error = unlocked ? SharedMemoryQuotaError::kDeviceTableFull
                                  : SharedMemoryQuotaError::kLockFailed;
            }
            return false;
        }
        device_slot = *free_device;
        region->devices[device_slot] = DeviceQuotaRecord{
            .state = kDeviceActive,
            .device = config.device,
            .limit_bytes = config.limit_bytes,
            .reserved_bytes = 0,
            .committed_bytes = 0,
        };
        created_device = true;
    }

    const auto free_process = find_free_process_slot(*region);
    if (!free_process.has_value()) {
        if (created_device) {
            region->devices[device_slot] = DeviceQuotaRecord{};
        }
        const bool unlocked = unlock_region(*region);
        if (!unlocked) {
            mark_unhealthy();
        }
        if (error != nullptr) {
            *error = unlocked ? SharedMemoryQuotaError::kProcessTableFull
                              : SharedMemoryQuotaError::kLockFailed;
        }
        return false;
    }

    ++region->generation;
    if (region->generation == 0) {
        region->generation = 1;
    }
    process_slot = static_cast<std::uint32_t>(*free_process);
    process_generation = region->generation;
    auto& slot = region->processes[process_slot];
    clear_process_slot(slot);
    slot.state = kProcessActive;
    slot.process_id = static_cast<std::int32_t>(process_id);
    slot.process_start_time = process_start_time;
    slot.generation = process_generation;
    slot.heartbeat = monotonic_ticks();
    (void)device_slot;
    const bool unlocked = unlock_region(*region);
    if (!unlocked && error != nullptr) {
        *error = SharedMemoryQuotaError::kLockFailed;
    }
    return unlocked;
}

bool SharedQuotaState::ensure_process_identity() noexcept {
    const pid_t current_process_id = ::getpid();
    if (process_id == current_process_id) {
        return true;
    }

    // The child inherits the bytes of the parent, including any process-local
    // mutex that may have been held by a vanished thread. Leak the inherited
    // mutex object in the child and install a fresh one before registering a
    // new shared-memory process slot.
    const bool inherited_mutex_present = reservation_mutex.release() != nullptr;
    if (!inherited_mutex_present) {
        mark_unhealthy();
    }
    reservation_mutex.reset(new (std::nothrow) std::mutex());
    if (reservation_mutex == nullptr) {
        mark_unhealthy();
        return false;
    }
    active_reservation_ids.clear();
    local_process_id = current_process_id;
    return register_process(config, nullptr);
}

bool SharedQuotaState::recover_dead_processes_if_due() const noexcept {
    if (region == nullptr) {
        return false;
    }

    constexpr std::uint64_t recovery_interval_milliseconds = 1000;
    const std::uint64_t now = monotonic_milliseconds();
    std::uint64_t last = last_recovery_milliseconds.load(std::memory_order_relaxed);
    if (last != 0 && now >= last && now - last < recovery_interval_milliseconds) {
        return true;
    }
    if (!last_recovery_milliseconds.compare_exchange_strong(last, now, std::memory_order_relaxed,
                                                            std::memory_order_relaxed)) {
        return true;
    }
    return recover_dead_processes(*region);
}

bool SharedQuotaState::track_reservation(std::uint64_t reservation_id) noexcept {
    if (reservation_mutex == nullptr || local_process_id != ::getpid()) {
        return false;
    }
    try {
        std::scoped_lock lock(*reservation_mutex);
        return active_reservation_ids.insert(reservation_id).second;
    } catch (...) {
        return false;
    }
}

bool SharedQuotaState::is_owner_locked() const noexcept {
    if (region == nullptr || process_slot >= kMaxProcessSlots) {
        return false;
    }
    const auto& slot = region->processes[process_slot];
    return slot.state == kProcessActive && slot.process_id == process_id &&
           slot.process_start_time == process_start_time && slot.generation == process_generation;
}

// The quota API deliberately pairs a device identifier with a byte count.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)

std::optional<ReservationToken> SharedQuotaState::reserve(DeviceId device,
                                                          core::MemoryBytes memory_bytes) noexcept {
    if (device < 0) {
        return std::nullopt;
    }
    if (!is_healthy()) {
        return std::nullopt;
    }
    if (!ensure_process_identity()) {
        mark_unhealthy();
        return std::nullopt;
    }
    if (!recover_dead_processes_if_due()) {
        mark_unhealthy();
        return std::nullopt;
    }
    return reserve_impl(device, memory_bytes, false);
}

std::optional<ReservationToken> SharedQuotaState::reserve_impl(DeviceId device,
                                                               core::MemoryBytes memory_bytes,
                                                               bool recovery_attempted) noexcept {
    if (region == nullptr || !lock_region(*region)) {
        mark_unhealthy();
        return std::nullopt;
    }
    auto device_slot = find_device_slot(*region, device);
    const bool owner_valid = is_owner_locked();
    if (!owner_valid) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        mark_unhealthy();
        return std::nullopt;
    }

    if (!device_slot.has_value()) {
        const auto free_device = find_free_device_slot(*region);
        if (!free_device.has_value()) {
            if (!unlock_region(*region)) {
                mark_unhealthy();
            }
            return std::nullopt;
        }
        region->devices[*free_device] = DeviceQuotaRecord{
            .state = kDeviceActive,
            .device = device,
            .limit_bytes = default_limit_bytes,
            .reserved_bytes = 0,
            .committed_bytes = 0,
        };
        device_slot = free_device;
    }

    auto& device_record = region->devices[*device_slot];
    auto& process = region->processes[process_slot];
    core::MemoryBytes used_bytes = 0;
    core::MemoryBytes requested_used = 0;
    core::MemoryBytes new_device_reserved = 0;
    core::MemoryBytes new_process_reserved = 0;
    const bool sums_valid =
        checked_add(device_record.reserved_bytes, device_record.committed_bytes, &used_bytes) &&
        checked_add(used_bytes, memory_bytes, &requested_used);
    const bool quota_rejected = sums_valid && requested_used > device_record.limit_bytes;
    if (!sums_valid || quota_rejected ||
        !checked_add(device_record.reserved_bytes, memory_bytes, &new_device_reserved) ||
        !checked_add(process.reserved_bytes[*device_slot], memory_bytes, &new_process_reserved)) {
        const bool unlocked = unlock_region(*region);
        if (!unlocked) {
            mark_unhealthy();
            return std::nullopt;
        }
        if (quota_rejected && !recovery_attempted) {
            if (!recover_dead_processes(*region)) {
                mark_unhealthy();
                return std::nullopt;
            }
            return reserve_impl(device, memory_bytes, true);
        }
        return std::nullopt;
    }
    device_record.reserved_bytes = new_device_reserved;
    process.reserved_bytes[*device_slot] = new_process_reserved;
    process.heartbeat = monotonic_ticks();
    std::uint64_t reservation_id = next_reservation_id.fetch_add(1, std::memory_order_relaxed);
    if (reservation_id == 0) {
        reservation_id = next_reservation_id.fetch_add(1, std::memory_order_relaxed);
        if (reservation_id == 0) {
            reservation_id = 1;
        }
    }
    const ReservationToken token{
        .process_slot = process_slot,
        .device_slot = static_cast<std::uint32_t>(*device_slot),
        .process_generation = process_generation,
        .reservation_id = reservation_id,
        .memory_bytes = memory_bytes,
    };
    if (!unlock_region(*region)) {
        // No reservation token was published because the shared mutex did not
        // release successfully. Restore both counters while this thread still
        // owns the lock (or while the region is already unusable), then fail
        // closed. A later process-death recovery can reclaim the slot if the
        // mutex itself cannot be recovered.
        device_record.reserved_bytes -= memory_bytes;
        process.reserved_bytes[*device_slot] -= memory_bytes;
        mark_unhealthy();
        return std::nullopt;
    }
    if (!track_reservation(token.reservation_id)) {
        if (!cancel_counters(token)) {
            mark_unhealthy();
        }
        return std::nullopt;
    }
    return token;
}

bool SharedQuotaState::commit(const ReservationToken& token) noexcept {
    if (!ensure_process_identity()) {
        mark_unhealthy();
        return false;
    }
    if (reservation_mutex == nullptr || local_process_id != ::getpid()) {
        mark_unhealthy();
        return false;
    }
    try {
        std::scoped_lock lock(*reservation_mutex);
        if (!active_reservation_ids.contains(token.reservation_id)) {
            mark_unhealthy();
            return false;
        }
        if (!commit_counters(token)) {
            mark_unhealthy();
            return false;
        }
        active_reservation_ids.erase(token.reservation_id);
        return true;
    } catch (...) {
        mark_unhealthy();
        return false;
    }
}

bool SharedQuotaState::cancel(const ReservationToken& token) noexcept {
    if (!ensure_process_identity()) {
        mark_unhealthy();
        return false;
    }
    if (reservation_mutex == nullptr || local_process_id != ::getpid()) {
        mark_unhealthy();
        return false;
    }
    try {
        std::scoped_lock lock(*reservation_mutex);
        if (!active_reservation_ids.contains(token.reservation_id)) {
            return true;
        }
        if (!cancel_counters(token)) {
            mark_unhealthy();
            return false;
        }
        active_reservation_ids.erase(token.reservation_id);
        return true;
    } catch (...) {
        mark_unhealthy();
        return false;
    }
}

bool SharedQuotaState::commit_counters(const ReservationToken& token) noexcept {
    if (region == nullptr || !lock_region(*region)) {
        return false;
    }
    const bool token_valid = token.process_slot == process_slot &&
                             token.device_slot < kMaxDevices &&
                             token.process_generation == process_generation &&
                             token.reservation_id != 0 && is_owner_locked();
    if (!token_valid) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        return false;
    }
    auto& device = region->devices[token.device_slot];
    auto& process = region->processes[process_slot];
    core::MemoryBytes new_device_committed = 0;
    core::MemoryBytes new_process_committed = 0;
    if (device.state != kDeviceActive ||
        process.reserved_bytes[token.device_slot] < token.memory_bytes ||
        device.reserved_bytes < token.memory_bytes ||
        !checked_add(device.committed_bytes, token.memory_bytes, &new_device_committed) ||
        !checked_add(process.committed_bytes[token.device_slot], token.memory_bytes,
                     &new_process_committed)) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        return false;
    }
    device.committed_bytes = new_device_committed;
    process.committed_bytes[token.device_slot] = new_process_committed;
    device.reserved_bytes -= token.memory_bytes;
    process.reserved_bytes[token.device_slot] -= token.memory_bytes;
    process.heartbeat = monotonic_ticks();
    return unlock_region(*region);
}

bool SharedQuotaState::cancel_counters(const ReservationToken& token) noexcept {
    if (region == nullptr || !lock_region(*region)) {
        return false;
    }
    const bool token_valid = token.process_slot == process_slot &&
                             token.device_slot < kMaxDevices &&
                             token.process_generation == process_generation &&
                             token.reservation_id != 0 && is_owner_locked();
    if (!token_valid) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        return false;
    }
    auto& device = region->devices[token.device_slot];
    auto& process = region->processes[process_slot];
    if (device.state != kDeviceActive ||
        process.reserved_bytes[token.device_slot] < token.memory_bytes ||
        device.reserved_bytes < token.memory_bytes) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        return false;
    }
    device.reserved_bytes -= token.memory_bytes;
    process.reserved_bytes[token.device_slot] -= token.memory_bytes;
    process.heartbeat = monotonic_ticks();
    return unlock_region(*region);
}

bool SharedQuotaState::release(DeviceId device, core::MemoryBytes memory_bytes) noexcept {
    if (device < 0) {
        return false;
    }
    if (!ensure_process_identity()) {
        mark_unhealthy();
        return false;
    }
    if (!recover_dead_processes_if_due()) {
        mark_unhealthy();
        return false;
    }
    if (region == nullptr || !lock_region(*region)) {
        mark_unhealthy();
        return false;
    }
    const auto device_slot = find_device_slot(*region, device);
    if (!device_slot.has_value() || !is_owner_locked()) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        return false;
    }
    auto& device_record = region->devices[*device_slot];
    auto& process = region->processes[process_slot];
    if (device_record.committed_bytes < memory_bytes ||
        process.committed_bytes[*device_slot] < memory_bytes) {
        if (!unlock_region(*region)) {
            mark_unhealthy();
        }
        return false;
    }
    device_record.committed_bytes -= memory_bytes;
    process.committed_bytes[*device_slot] -= memory_bytes;
    process.heartbeat = monotonic_ticks();
    const bool unlocked = unlock_region(*region);
    if (!unlocked) {
        mark_unhealthy();
    }
    return unlocked;
}

core::QuotaUsage SharedQuotaState::usage(DeviceId device) const noexcept {
    if (device < 0) {
        return {};
    }
    if (!const_cast<SharedQuotaState*>(this)->ensure_process_identity()) {
        is_healthy_.store(false, std::memory_order_relaxed);
        return {};
    }
    if (!recover_dead_processes_if_due()) {
        is_healthy_.store(false, std::memory_order_relaxed);
        return {};
    }
    if (region == nullptr || !lock_region(*const_cast<SharedRegion*>(region))) {
        is_healthy_.store(false, std::memory_order_relaxed);
        return {};
    }
    const auto device_slot = find_device_slot(*region, device);
    if (!device_slot.has_value()) {
        if (!unlock_region(*const_cast<SharedRegion*>(region))) {
            is_healthy_.store(false, std::memory_order_relaxed);
            return {};
        }
        return core::QuotaUsage{
            .limit_bytes = default_limit_bytes,
            .reserved_bytes = 0,
            .allocated_bytes = 0,
        };
    }
    const auto& device_record = region->devices[*device_slot];
    const core::QuotaUsage result{
        .limit_bytes = device_record.limit_bytes,
        .reserved_bytes = device_record.reserved_bytes,
        .allocated_bytes = device_record.committed_bytes,
    };
    if (!unlock_region(*const_cast<SharedRegion*>(region))) {
        is_healthy_.store(false, std::memory_order_relaxed);
        return {};
    }
    return result;
}

bool SharedQuotaState::is_healthy() const noexcept {
    return is_healthy_.load(std::memory_order_relaxed);
}

void SharedQuotaState::mark_unhealthy() const noexcept {
    is_healthy_.store(false, std::memory_order_relaxed);
}

}  // namespace glimmer::control::detail

namespace glimmer::control {

namespace {

class SharedMemoryReservationState final : public detail::MemoryReservationState {
   public:
    SharedMemoryReservationState(std::shared_ptr<detail::SharedQuotaState> state,
                                 detail::ReservationToken token)
        : state_(std::move(state)), token_(token) {}

    [[nodiscard]] bool commit() noexcept override {
        if (!is_active_) {
            return false;
        }
        try {
            if (!state_->commit(token_)) {
                return false;
            }
            is_active_ = false;
            state_.reset();
            return true;
        } catch (...) {
            return false;
        }
    }

    void cancel() noexcept override {
        if (is_active_) {
            if (state_->cancel(token_)) {
                is_active_ = false;
            }
        }
    }

    void abandon() noexcept override {
        is_active_ = false;
        state_.reset();
    }

    [[nodiscard]] bool is_active() const noexcept override {
        return is_active_;
    }

    [[nodiscard]] core::MemoryBytes memory_bytes() const noexcept override {
        return is_active_ ? token_.memory_bytes : 0;
    }

   private:
    std::shared_ptr<detail::SharedQuotaState> state_;
    detail::ReservationToken token_;
    bool is_active_ = true;
};

}  // namespace

SharedMemoryQuota::SharedMemoryQuota(std::shared_ptr<detail::SharedQuotaState> state) noexcept
    : state_(std::move(state)) {}

SharedMemoryQuota::SharedMemoryQuota(SharedMemoryQuota&& other) noexcept
    : state_(std::move(other.state_)) {}

SharedMemoryQuota& SharedMemoryQuota::operator=(SharedMemoryQuota&& other) noexcept {
    if (this != &other) {
        state_ = std::move(other.state_);
    }
    return *this;
}

SharedMemoryQuota::~SharedMemoryQuota() = default;

std::unique_ptr<SharedMemoryQuota> SharedMemoryQuota::open(const SharedMemoryQuotaConfig& config,
                                                           SharedMemoryQuotaError* error) noexcept {
    if (error != nullptr) {
        *error = SharedMemoryQuotaError::kNone;
    }
    std::optional<detail::OpenRegion> open_result;
    try {
        open_result = detail::open_region(config, error);
        if (!open_result.has_value()) {
            return nullptr;
        }

        auto state = std::make_shared<detail::SharedQuotaState>();
        state->file_descriptor = open_result->file_descriptor;
        state->mapping = open_result->mapping;
        state->region = static_cast<detail::SharedRegion*>(open_result->mapping);
        state->region_name = open_result->name;
        state->config = config;
        open_result->file_descriptor = -1;
        open_result->mapping = nullptr;
        if (!state->register_process(config, error)) {
            return nullptr;
        }
        return std::unique_ptr<SharedMemoryQuota>(new SharedMemoryQuota(std::move(state)));
    } catch (...) {
        if (open_result.has_value()) {
            if (!detail::close_open_region(*open_result, true) && error != nullptr) {
                *error = SharedMemoryQuotaError::kUnknown;
            }
        }
        if (error != nullptr) {
            *error = SharedMemoryQuotaError::kUnknown;
        }
        return nullptr;
    }
}

bool SharedMemoryQuota::remove_region(std::string_view tenant_id) noexcept {
    if (tenant_id.empty() || tenant_id.size() > SharedMemoryQuota::max_tenant_id_bytes) {
        return false;
    }
    try {
        detail::OpenRegion region;
        region.name = detail::make_region_name(tenant_id);
        region.file_descriptor = ::shm_open(region.name.c_str(), O_RDWR, 0600);
        if (region.file_descriptor < 0) {
            return errno == ENOENT;
        }
        if (!detail::lock_file(region.file_descriptor)) {
            const bool closed = detail::close_open_region(region, false);
            if (!closed) {
                return false;
            }
            return false;
        }

        bool can_remove = true;
        struct stat file_status {};
        if (::fstat(region.file_descriptor, &file_status) != 0) {
            can_remove = false;
        } else if (file_status.st_size == static_cast<off_t>(sizeof(detail::SharedRegion))) {
            region.mapping = ::mmap(nullptr, sizeof(detail::SharedRegion), PROT_READ | PROT_WRITE,
                                    MAP_SHARED, region.file_descriptor, 0);
            if (region.mapping == MAP_FAILED) {
                region.mapping = nullptr;
                can_remove = false;
            } else {
                auto& shared_region = *static_cast<detail::SharedRegion*>(region.mapping);
                if (detail::valid_region(shared_region, tenant_id)) {
                    if (!detail::lock_region(shared_region)) {
                        can_remove = false;
                    } else {
                        for (const auto& process : shared_region.processes) {
                            if (process.state == detail::kProcessFree) {
                                continue;
                            }
                            constexpr std::size_t device_count =
                                sizeof(process.reserved_bytes) / sizeof(process.reserved_bytes[0]);
                            for (std::size_t device_index = 0; device_index < device_count;
                                 ++device_index) {
                                if (process.reserved_bytes[device_index] != 0 ||
                                    process.committed_bytes[device_index] != 0) {
                                    can_remove = false;
                                    break;
                                }
                            }
                            if (!can_remove) {
                                break;
                            }
                        }
                        if (can_remove) {
                            // Mark the inode unusable before unlinking it. A process that
                            // opened the old inode before this call then fails validation
                            // instead of registering into a split-brain region.
                            shared_region.state = detail::kRegionRemoving;
                        }
                        if (!detail::unlock_region(shared_region)) {
                            can_remove = false;
                        }
                    }
                }
            }
        }

        if (!can_remove) {
            const bool unlocked = detail::unlock_file(region.file_descriptor);
            const bool closed = detail::close_open_region(region, false);
            if (!unlocked || !closed) {
                return false;
            }
            return false;
        }

        const bool unlinked = ::shm_unlink(region.name.c_str()) == 0 || errno == ENOENT;
        const bool unlocked = detail::unlock_file(region.file_descriptor);
        const bool closed = detail::close_open_region(region, false);
        return unlinked && unlocked && closed;
    } catch (...) {
        return false;
    }
}

std::optional<MemoryReservation> SharedMemoryQuota::try_reserve(DeviceId device,
                                                                core::MemoryBytes memory_bytes) {
    if (state_ == nullptr) {
        return std::nullopt;
    }
    const auto token = state_->reserve(device, memory_bytes);
    if (!token.has_value()) {
        return std::nullopt;
    }
    try {
        return MemoryReservation(std::make_unique<SharedMemoryReservationState>(state_, *token));
    } catch (...) {
        if (!state_->cancel(*token)) {
            state_->mark_unhealthy();
        }
        return std::nullopt;
    }
}

bool SharedMemoryQuota::release(DeviceId device, core::MemoryBytes memory_bytes) {
    return state_ != nullptr && state_->release(device, memory_bytes);
}

MemoryInfo SharedMemoryQuota::get_memory_info(DeviceId device,
                                              core::MemoryBytes physical_total_bytes,
                                              core::MemoryBytes physical_free_bytes) const {
    const core::QuotaUsage current_usage = usage(device);
    const core::MemoryBytes visible_total_bytes =
        std::min(current_usage.limit_bytes, physical_total_bytes);
    const core::MemoryBytes visible_free_bytes =
        std::min(current_usage.available_bytes(), physical_free_bytes);
    return MemoryInfo{
        .total_bytes = visible_total_bytes,
        .free_bytes = std::min(visible_total_bytes, visible_free_bytes),
    };
}

// NOLINTEND(bugprone-easily-swappable-parameters)

core::QuotaUsage SharedMemoryQuota::usage(DeviceId device) const {
    return state_ == nullptr ? core::QuotaUsage{} : state_->usage(device);
}

bool SharedMemoryQuota::is_healthy() const noexcept {
    return state_ != nullptr && state_->is_healthy();
}

}  // namespace glimmer::control
