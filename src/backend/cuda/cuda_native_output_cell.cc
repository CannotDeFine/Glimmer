#include "internal/cuda_native_output_cell.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
constexpr std::uint64_t kPagePresent = UINT64_C(1) << 63;
constexpr std::uint64_t kPageSwapped = UINT64_C(1) << 62;
constexpr std::uint64_t kPageFile = UINT64_C(1) << 61;
constexpr int kImmutableSeals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL;

CUresult system_failure(int error) noexcept {
    if (error == ENOMEM || error == EMFILE || error == ENFILE) return CUDA_ERROR_OUT_OF_MEMORY;
    if (error == EACCES || error == EPERM || error == ENOSYS || error == EOPNOTSUPP)
        return CUDA_ERROR_NOT_SUPPORTED;
    return CUDA_ERROR_UNKNOWN;
}

void report_cleanup_failure(const char* operation, int error) noexcept {
    try {
        spdlog::critical("Native output cell cleanup failed: operation={}, errno={}", operation,
                         error);
    } catch (...) {
        // No exception may escape a resource destructor, including logging.
    }
}
void close_descriptor(int& descriptor) noexcept {
    const auto owned = std::exchange(descriptor, -1);
    // Linux close consumes the descriptor even on EINTR. Never retry and risk
    // closing another thread's newly reused descriptor number.
    if (owned >= 0 && ::close(owned) != 0) report_cleanup_failure("close", errno);
}
}  // namespace

struct CudaNativeOutputCell::Implementation {
    pid_t process{::getpid()};
    int file{-1}, pagemap{-1};
    void* mapping{MAP_FAILED};
    std::size_t page_bytes{}, cell_bytes{};
    off_t pagemap_offset{};

    ~Implementation() {
        // MADV_DONTFORK deliberately omits this mapping from the child. Its
        // numeric address may already identify unrelated child storage. Do not
        // inspect/unmap it or use inherited cleanup state in that process.
        if (::getpid() != process) return;
        if (mapping != MAP_FAILED && ::munmap(mapping, page_bytes) != 0)
            report_cleanup_failure("munmap", errno);
        close_descriptor(pagemap);
        close_descriptor(file);
    }

    CudaNativeOutputObservation observe() const noexcept {
        std::uint64_t entry{};
        ssize_t received{};
        do {
            received = ::pread(pagemap, &entry, sizeof(entry), pagemap_offset);
        } while (received < 0 && errno == EINTR);
        if (received < 0) return {system_failure(errno), false};
        if (received != static_cast<ssize_t>(sizeof(entry)) || !(entry & kPagePresent) ||
            (entry & kPageSwapped))
            return {CUDA_ERROR_UNKNOWN, false};
        return {CUDA_SUCCESS, !(entry & kPageFile)};
    }

    CUresult initialize(std::span<const std::byte> initial) noexcept {
        const auto system_page = ::sysconf(_SC_PAGESIZE);
        if (system_page <= 0) return CUDA_ERROR_NOT_SUPPORTED;
        const auto page = static_cast<std::uintmax_t>(system_page);
        if (page > std::numeric_limits<std::size_t>::max() ||
            page > static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max()))
            return CUDA_ERROR_NOT_SUPPORTED;
        page_bytes = static_cast<std::size_t>(page);
        cell_bytes = initial.size();
        if (!cell_bytes || !initial.data() || cell_bytes > page_bytes)
            return CUDA_ERROR_INVALID_VALUE;
        file = ::memfd_create("glimmer-native-output", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (file < 0) return system_failure(errno);
        if (::ftruncate(file, static_cast<off_t>(page_bytes)) != 0) return system_failure(errno);
        std::size_t written{};
        while (written != cell_bytes) {
            const auto count = ::pwrite(file, initial.data() + written, cell_bytes - written,
                                        static_cast<off_t>(written));
            if (count < 0) {
                if (errno == EINTR) continue;
                return system_failure(errno);
            }
            if (!count || static_cast<std::uintmax_t>(count) > cell_bytes - written)
                return CUDA_ERROR_UNKNOWN;
            written += static_cast<std::size_t>(count);
        }
        if (::fcntl(file, F_ADD_SEALS, kImmutableSeals) != 0) return system_failure(errno);
        const auto seals = ::fcntl(file, F_GET_SEALS);
        if (seals < 0) return system_failure(errno);
        if ((seals & kImmutableSeals) != kImmutableSeals) return CUDA_ERROR_UNKNOWN;
        mapping = ::mmap(nullptr, page_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, file, 0);
        if (mapping == MAP_FAILED) return system_failure(errno);
        if (::madvise(mapping, page_bytes, MADV_DONTFORK) != 0 ||
            ::madvise(mapping, page_bytes, MADV_NOHUGEPAGE) != 0)
            return system_failure(errno);
        pagemap = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
        if (pagemap < 0) return system_failure(errno);
        const auto page_index = reinterpret_cast<std::uintptr_t>(mapping) / page_bytes;
        const auto maximum_offset = static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max());
        if (page_index > maximum_offset / sizeof(std::uint64_t)) return CUDA_ERROR_NOT_SUPPORTED;
        pagemap_offset = static_cast<off_t>(page_index * sizeof(std::uint64_t));
        // A read fault makes the immutable file page present without triggering
        // COW. Reject a platform which cannot prove this initial provenance.
        const auto first = *static_cast<const volatile unsigned char*>(mapping);
        (void)first;
        const auto observed = observe();
        if (observed.status != CUDA_SUCCESS) return observed.status;
        return observed.was_written ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
    }
};

CudaNativeOutputCell::CudaNativeOutputCell(std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation)) {}
CudaNativeOutputCell::~CudaNativeOutputCell() = default;

CudaNativeOutputCellCreation CudaNativeOutputCell::create(
    std::span<const std::byte> initial) noexcept {
    try {
        auto implementation = std::make_unique<Implementation>();
        const auto status = implementation->initialize(initial);
        if (status != CUDA_SUCCESS) return {status, {}};
        return {CUDA_SUCCESS, std::shared_ptr<CudaNativeOutputCell>(
                                  new CudaNativeOutputCell(std::move(implementation)))};
    } catch (const std::bad_alloc&) {
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    } catch (...) {
        return {CUDA_ERROR_UNKNOWN, {}};
    }
}
void* CudaNativeOutputCell::data() const noexcept {
    return ::getpid() == implementation_->process ? implementation_->mapping : nullptr;
}
CudaNativeOutputObservation CudaNativeOutputCell::snapshot(
    std::span<std::byte> output) const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process) return {CUDA_ERROR_INVALID_CONTEXT, false};
    if (!output.data() || output.size() != state.cell_bytes)
        return {CUDA_ERROR_INVALID_VALUE, false};
    const auto destination = reinterpret_cast<std::uintptr_t>(output.data());
    const auto source = reinterpret_cast<std::uintptr_t>(state.mapping);
    if (output.size() > std::numeric_limits<std::uintptr_t>::max() - destination ||
        (destination >= source ? destination - source < state.page_bytes
                               : source - destination < output.size()))
        return {CUDA_ERROR_INVALID_VALUE, false};
    std::memcpy(output.data(), state.mapping, output.size());
    return state.observe();
}

}  // namespace glimmer::backend::cuda
