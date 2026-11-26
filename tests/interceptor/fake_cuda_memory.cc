#include "../../src/interceptor/internal/driver_copy_wrappers.h"

#include <array>
#include <atomic>
#include <new>
#include <string_view>

// Native lookup returns local implementations, not application-interposed
// aliases. This models the real Driver\'s privately bound dispatch table.
// NOLINTBEGIN(bugprone-easily-swappable-parameters): preserve CUDA argument order.
namespace {
std::array<std::atomic<unsigned int>, 44> calls{};
std::atomic<void (*)()> stream_metadata_callback{};
}  // namespace

// Test-local instrumentation of original invocation; no application routing.
extern "C" void glimmer_test_set_stream_metadata_callback(void (*callback)()) {
    stream_metadata_callback.store(callback);
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpyhtodasync_v2(
    CUdeviceptr destination, const void* host, std::size_t memory_bytes, CUstream stream) {
    (void)stream;
    (void)host;
    (void)destination;
    ++calls[0];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpyhtodasync_v2_ptsz(
    CUdeviceptr destination, const void* host, std::size_t memory_bytes, CUstream stream) {
    (void)stream;
    (void)host;
    (void)destination;
    ++calls[1];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpyhtod_v2(CUdeviceptr destination, const void* host, std::size_t memory_bytes) {
    (void)host;
    (void)destination;
    ++calls[2];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpyhtod_v2_ptds(CUdeviceptr destination, const void* host, std::size_t memory_bytes) {
    (void)host;
    (void)destination;
    ++calls[3];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpydtohasync_v2(
    void* host, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    (void)stream;
    (void)source;
    (void)host;
    ++calls[4];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpydtohasync_v2_ptsz(
    void* host, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    (void)stream;
    (void)source;
    (void)host;
    ++calls[5];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpydtoh_v2(void* host, CUdeviceptr source, std::size_t memory_bytes) {
    (void)source;
    (void)host;
    ++calls[6];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpydtoh_v2_ptds(void* host, CUdeviceptr source, std::size_t memory_bytes) {
    (void)source;
    (void)host;
    ++calls[7];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpydtodasync_v2(
    CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    (void)stream;
    (void)source;
    (void)destination;
    ++calls[8];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpydtodasync_v2_ptsz(
    CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    (void)stream;
    (void)source;
    (void)destination;
    ++calls[9];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpydtod_v2(CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes) {
    (void)source;
    (void)destination;
    ++calls[10];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpydtod_v2_ptds(CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes) {
    (void)source;
    (void)destination;
    ++calls[11];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemhostalloc(void** pointer, std::size_t memory_bytes, unsigned int flags) {
    (void)flags;
    ++calls[12];
    if (!pointer || !memory_bytes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer = new (std::nothrow) std::byte[memory_bytes];
    return *pointer ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemallochost_v2(void** pointer, std::size_t memory_bytes) {
    ++calls[13];
    if (!pointer || !memory_bytes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer = new (std::nothrow) std::byte[memory_bytes];
    return *pointer ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemfreehost(void* pointer) {
    ++calls[14];
    delete[] static_cast<std::byte*>(pointer);
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemhostregister_v2(void* pointer, std::size_t memory_bytes, unsigned int flags) {
    (void)flags;
    (void)memory_bytes;
    (void)pointer;
    ++calls[15];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemhostunregister(void* pointer) {
    (void)pointer;
    ++calls[16];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemhostgetdevicepointer_v2(CUdeviceptr* device_pointer, void* pointer,
                                    unsigned int flags) {
    (void)flags;
    ++calls[17];
    if (!device_pointer) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = reinterpret_cast<CUdeviceptr>(pointer);
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreamcreate(CUstream* stream, unsigned int flags) {
    (void)flags;
    ++calls[18];
    if (!stream) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *stream = reinterpret_cast<CUstream>(0x5500);
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreamcreatewithpriority(CUstream* stream, unsigned int flags, int priority) {
    (void)priority;
    (void)flags;
    ++calls[19];
    if (!stream) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *stream = reinterpret_cast<CUstream>(0x5500);
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreamgetflags(CUstream stream, unsigned int* flags) {
    (void)stream;
    ++calls[20];
    if (const auto callback = stream_metadata_callback.load()) {
        callback();
    }
    if (!flags) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_STREAM_NON_BLOCKING;
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreamgetpriority(CUstream stream, int* priority) {
    (void)stream;
    ++calls[21];
    if (!priority) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *priority = -1;
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemsetd8_v2(CUdeviceptr pointer, unsigned char value, std::size_t count) {
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[22];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemsetd8_v2_ptds(CUdeviceptr pointer, unsigned char value, std::size_t count) {
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[23];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemsetd8async(
    CUdeviceptr pointer, unsigned char value, std::size_t count, CUstream stream) {
    (void)stream;
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[24];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemsetd8async_ptsz(
    CUdeviceptr pointer, unsigned char value, std::size_t count, CUstream stream) {
    (void)stream;
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[25];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemsetd16_v2(CUdeviceptr pointer, unsigned short value, std::size_t count) {
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[26];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemsetd16_v2_ptds(CUdeviceptr pointer, unsigned short value, std::size_t count) {
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[27];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemsetd16async(
    CUdeviceptr pointer, unsigned short value, std::size_t count, CUstream stream) {
    (void)stream;
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[28];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemsetd16async_ptsz(
    CUdeviceptr pointer, unsigned short value, std::size_t count, CUstream stream) {
    (void)stream;
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[29];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemsetd32_v2(CUdeviceptr pointer, unsigned int value, std::size_t count) {
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[30];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemsetd32_v2_ptds(CUdeviceptr pointer, unsigned int value, std::size_t count) {
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[31];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemsetd32async(
    CUdeviceptr pointer, unsigned int value, std::size_t count, CUstream stream) {
    (void)stream;
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[32];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemsetd32async_ptsz(
    CUdeviceptr pointer, unsigned int value, std::size_t count, CUstream stream) {
    (void)stream;
    (void)count;
    (void)value;
    (void)pointer;
    ++calls[33];
    return CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreambegincapture_v2(CUstream stream, CUstreamCaptureMode mode) {
    ++calls[34];
    if (stream == reinterpret_cast<CUstream>(0xdead)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return mode == CU_STREAM_CAPTURE_MODE_RELAXED ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreambegincapture_v2_ptsz(CUstream stream, CUstreamCaptureMode mode) {
    ++calls[35];
    if (stream == reinterpret_cast<CUstream>(0xdead)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    return mode == CU_STREAM_CAPTURE_MODE_RELAXED ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreambegincapture(CUstream stream) {
    ++calls[42];
    return stream == reinterpret_cast<CUstream>(0xdead) ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}
extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreambegincapture_ptsz(CUstream stream) {
    ++calls[43];
    return stream == reinterpret_cast<CUstream>(0xdead) ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreamendcapture(CUstream stream, CUgraph* graph) {
    (void)stream;
    ++calls[36];
    if (graph) {
        *graph = nullptr;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_custreamendcapture_ptsz(CUstream stream, CUgraph* graph) {
    (void)stream;
    ++calls[37];
    if (graph) {
        *graph = nullptr;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpy(CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes) {
    (void)destination;
    (void)source;

    ++calls[38];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy(CUdeviceptr destination, CUdeviceptr source,
                                     std::size_t memory_bytes) {
    return native_cumemcpy(destination, source, memory_bytes);
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI
native_cumemcpy_ptds(CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes) {
    (void)destination;
    (void)source;

    ++calls[39];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t memory_bytes) {
    return native_cumemcpy_ptds(destination, source, memory_bytes);
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpyasync(
    CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    (void)destination;
    (void)source;
    (void)stream;
    ++calls[40];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpyasync(destination, source, memory_bytes, stream);
}

extern "C" __attribute__((visibility("hidden"))) CUresult CUDAAPI native_cumemcpyasync_ptsz(
    CUdeviceptr destination, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    (void)destination;
    (void)source;
    (void)stream;
    ++calls[41];
    return memory_bytes == 13 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr destination, CUdeviceptr source,
                                               std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpyasync_ptsz(destination, source, memory_bytes, stream);
}

namespace {
struct Symbol {
    std::string_view name;
    void* function;
    std::size_t index;
};
const auto kSymbols = std::to_array<Symbol>({
    {"cuMemcpyHtoDAsync_v2", reinterpret_cast<void*>(&native_cumemcpyhtodasync_v2), 0},
    {"cuMemcpyHtoDAsync", reinterpret_cast<void*>(&native_cumemcpyhtodasync_v2), 0},
    {"cuMemcpyHtoDAsync_v2_ptsz", reinterpret_cast<void*>(&native_cumemcpyhtodasync_v2_ptsz), 1},
    {"cuMemcpyHtoDAsync_ptsz", reinterpret_cast<void*>(&native_cumemcpyhtodasync_v2_ptsz), 1},
    {"cuMemcpyHtoD_v2", reinterpret_cast<void*>(&native_cumemcpyhtod_v2), 2},
    {"cuMemcpyHtoD", reinterpret_cast<void*>(&native_cumemcpyhtod_v2), 2},
    {"cuMemcpyHtoD_v2_ptds", reinterpret_cast<void*>(&native_cumemcpyhtod_v2_ptds), 3},
    {"cuMemcpyHtoD_ptds", reinterpret_cast<void*>(&native_cumemcpyhtod_v2_ptds), 3},
    {"cuMemcpyDtoHAsync_v2", reinterpret_cast<void*>(&native_cumemcpydtohasync_v2), 4},
    {"cuMemcpyDtoHAsync", reinterpret_cast<void*>(&native_cumemcpydtohasync_v2), 4},
    {"cuMemcpyDtoHAsync_v2_ptsz", reinterpret_cast<void*>(&native_cumemcpydtohasync_v2_ptsz), 5},
    {"cuMemcpyDtoHAsync_ptsz", reinterpret_cast<void*>(&native_cumemcpydtohasync_v2_ptsz), 5},
    {"cuMemcpyDtoH_v2", reinterpret_cast<void*>(&native_cumemcpydtoh_v2), 6},
    {"cuMemcpyDtoH", reinterpret_cast<void*>(&native_cumemcpydtoh_v2), 6},
    {"cuMemcpyDtoH_v2_ptds", reinterpret_cast<void*>(&native_cumemcpydtoh_v2_ptds), 7},
    {"cuMemcpyDtoH_ptds", reinterpret_cast<void*>(&native_cumemcpydtoh_v2_ptds), 7},
    {"cuMemcpyDtoDAsync_v2", reinterpret_cast<void*>(&native_cumemcpydtodasync_v2), 8},
    {"cuMemcpyDtoDAsync", reinterpret_cast<void*>(&native_cumemcpydtodasync_v2), 8},
    {"cuMemcpyDtoDAsync_v2_ptsz", reinterpret_cast<void*>(&native_cumemcpydtodasync_v2_ptsz), 9},
    {"cuMemcpyDtoDAsync_ptsz", reinterpret_cast<void*>(&native_cumemcpydtodasync_v2_ptsz), 9},
    {"cuMemcpyDtoD_v2", reinterpret_cast<void*>(&native_cumemcpydtod_v2), 10},
    {"cuMemcpyDtoD", reinterpret_cast<void*>(&native_cumemcpydtod_v2), 10},
    {"cuMemcpyDtoD_v2_ptds", reinterpret_cast<void*>(&native_cumemcpydtod_v2_ptds), 11},
    {"cuMemcpyDtoD_ptds", reinterpret_cast<void*>(&native_cumemcpydtod_v2_ptds), 11},
    {"cuMemHostAlloc", reinterpret_cast<void*>(&native_cumemhostalloc), 12},
    {"cuMemAllocHost_v2", reinterpret_cast<void*>(&native_cumemallochost_v2), 13},
    {"cuMemAllocHost", reinterpret_cast<void*>(&native_cumemallochost_v2), 13},
    {"cuMemFreeHost", reinterpret_cast<void*>(&native_cumemfreehost), 14},
    {"cuMemHostRegister_v2", reinterpret_cast<void*>(&native_cumemhostregister_v2), 15},
    {"cuMemHostRegister", reinterpret_cast<void*>(&native_cumemhostregister_v2), 15},
    {"cuMemHostUnregister", reinterpret_cast<void*>(&native_cumemhostunregister), 16},
    {"cuMemHostGetDevicePointer_v2", reinterpret_cast<void*>(&native_cumemhostgetdevicepointer_v2),
     17},
    {"cuMemHostGetDevicePointer", reinterpret_cast<void*>(&native_cumemhostgetdevicepointer_v2),
     17},
    {"cuStreamCreate", reinterpret_cast<void*>(&native_custreamcreate), 18},
    {"cuStreamCreateWithPriority", reinterpret_cast<void*>(&native_custreamcreatewithpriority), 19},
    {"cuStreamGetFlags", reinterpret_cast<void*>(&native_custreamgetflags), 20},
    {"cuStreamGetPriority", reinterpret_cast<void*>(&native_custreamgetpriority), 21},
    {"cuMemsetD8_v2", reinterpret_cast<void*>(&native_cumemsetd8_v2), 22},
    {"cuMemsetD8", reinterpret_cast<void*>(&native_cumemsetd8_v2), 22},
    {"cuMemsetD8_v2_ptds", reinterpret_cast<void*>(&native_cumemsetd8_v2_ptds), 23},
    {"cuMemsetD8_ptds", reinterpret_cast<void*>(&native_cumemsetd8_v2_ptds), 23},
    {"cuMemsetD8Async", reinterpret_cast<void*>(&native_cumemsetd8async), 24},
    {"cuMemsetD8Async_ptsz", reinterpret_cast<void*>(&native_cumemsetd8async_ptsz), 25},
    {"cuMemsetD16_v2", reinterpret_cast<void*>(&native_cumemsetd16_v2), 26},
    {"cuMemsetD16", reinterpret_cast<void*>(&native_cumemsetd16_v2), 26},
    {"cuMemsetD16_v2_ptds", reinterpret_cast<void*>(&native_cumemsetd16_v2_ptds), 27},
    {"cuMemsetD16_ptds", reinterpret_cast<void*>(&native_cumemsetd16_v2_ptds), 27},
    {"cuMemsetD16Async", reinterpret_cast<void*>(&native_cumemsetd16async), 28},
    {"cuMemsetD16Async_ptsz", reinterpret_cast<void*>(&native_cumemsetd16async_ptsz), 29},
    {"cuMemsetD32_v2", reinterpret_cast<void*>(&native_cumemsetd32_v2), 30},
    {"cuMemsetD32", reinterpret_cast<void*>(&native_cumemsetd32_v2), 30},
    {"cuMemsetD32_v2_ptds", reinterpret_cast<void*>(&native_cumemsetd32_v2_ptds), 31},
    {"cuMemsetD32_ptds", reinterpret_cast<void*>(&native_cumemsetd32_v2_ptds), 31},
    {"cuMemsetD32Async", reinterpret_cast<void*>(&native_cumemsetd32async), 32},
    {"cuMemsetD32Async_ptsz", reinterpret_cast<void*>(&native_cumemsetd32async_ptsz), 33},
    {"cuStreamBeginCapture_v2", reinterpret_cast<void*>(&native_custreambegincapture_v2), 34},
    {"cuStreamBeginCapture", reinterpret_cast<void*>(&native_custreambegincapture), 42},
    {"cuStreamBeginCapture_v2_ptsz", reinterpret_cast<void*>(&native_custreambegincapture_v2_ptsz),
     35},
    {"cuStreamBeginCapture_ptsz", reinterpret_cast<void*>(&native_custreambegincapture_ptsz), 43},
    {"cuStreamEndCapture", reinterpret_cast<void*>(&native_custreamendcapture), 36},
    {"cuStreamEndCapture_ptsz", reinterpret_cast<void*>(&native_custreamendcapture_ptsz), 37},
    {"cuMemcpy", reinterpret_cast<void*>(&native_cumemcpy), 38},
    {"cuMemcpy_ptds", reinterpret_cast<void*>(&native_cumemcpy_ptds), 39},
    {"cuMemcpyAsync", reinterpret_cast<void*>(&native_cumemcpyasync), 40},
    {"cuMemcpyAsync_ptsz", reinterpret_cast<void*>(&native_cumemcpyasync_ptsz), 41},
});
}  // namespace
extern "C" void* glimmer_test_memory_symbol(const char* name, cuuint64_t flags) {
    const std::string_view requested{name};
    if ((flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0) {
        for (const auto& symbol : kSymbols) {
            if (symbol.name.size() == requested.size() + 5 && symbol.name.starts_with(requested) &&
                (symbol.name.ends_with("_ptsz") || symbol.name.ends_with("_ptds"))) {
                return symbol.function;
            }
        }
    }
    for (const auto& symbol : kSymbols) {
        if (symbol.name == requested) {
            return symbol.function;
        }
    }
    return nullptr;
}
extern "C" unsigned int glimmer_test_native_memory_calls(const char* name) {
    for (const auto& symbol : kSymbols) {
        if (symbol.name == name) {
            return calls[symbol.index].load();
        }
    }
    return 0;
}

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void* host,
                                                 std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpyhtodasync_v2(destination, host, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr destination, const void* host,
                                                      std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpyhtodasync_v2_ptsz(destination, host, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination, const void* host,
                                            std::size_t memory_bytes) {
    return native_cumemcpyhtod_v2(destination, host, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr destination, const void* host,
                                                 std::size_t memory_bytes) {
    return native_cumemcpyhtod_v2_ptds(destination, host, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* host, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpydtohasync_v2(host, source, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void* host, CUdeviceptr source,
                                                      std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpydtohasync_v2_ptsz(host, source, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2(void* host, CUdeviceptr source,
                                            std::size_t memory_bytes) {
    return native_cumemcpydtoh_v2(host, source, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void* host, CUdeviceptr source,
                                                 std::size_t memory_bytes) {
    return native_cumemcpydtoh_v2_ptds(host, source, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpydtodasync_v2(destination, source, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr destination, CUdeviceptr source,
                                                      std::size_t memory_bytes, CUstream stream) {
    return native_cumemcpydtodasync_v2_ptsz(destination, source, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr destination, CUdeviceptr source,
                                            std::size_t memory_bytes) {
    return native_cumemcpydtod_v2(destination, source, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes) {
    return native_cumemcpydtod_v2_ptds(destination, source, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemHostAlloc(void** pointer, std::size_t memory_bytes,
                                           unsigned int flags) {
    return native_cumemhostalloc(pointer, memory_bytes, flags);
}

extern "C" CUresult CUDAAPI cuMemAllocHost_v2(void** pointer, std::size_t memory_bytes) {
    return native_cumemallochost_v2(pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemFreeHost(void* pointer) {
    return native_cumemfreehost(pointer);
}

extern "C" CUresult CUDAAPI cuMemHostRegister_v2(void* pointer, std::size_t memory_bytes,
                                                 unsigned int flags) {
    return native_cumemhostregister_v2(pointer, memory_bytes, flags);
}

extern "C" CUresult CUDAAPI cuMemHostUnregister(void* pointer) {
    return native_cumemhostunregister(pointer);
}

extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* device_pointer, void* pointer,
                                                         unsigned int flags) {
    return native_cumemhostgetdevicepointer_v2(device_pointer, pointer, flags);
}

extern "C" CUresult CUDAAPI cuStreamCreate(CUstream* stream, unsigned int flags) {
    return native_custreamcreate(stream, flags);
}

extern "C" CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* stream, unsigned int flags,
                                                       int priority) {
    return native_custreamcreatewithpriority(stream, flags, priority);
}

extern "C" CUresult CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int* flags) {
    return native_custreamgetflags(stream, flags);
}

extern "C" CUresult CUDAAPI cuStreamGetPriority(CUstream stream, int* priority) {
    return native_custreamgetpriority(stream, priority);
}

extern "C" CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr pointer, unsigned char value,
                                          std::size_t count) {
    return native_cumemsetd8_v2(pointer, value, count);
}

extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr pointer, unsigned char value,
                                               std::size_t count) {
    return native_cumemsetd8_v2_ptds(pointer, value, count);
}

extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr pointer, unsigned char value,
                                            std::size_t count, CUstream stream) {
    return native_cumemsetd8async(pointer, value, count, stream);
}

extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr pointer, unsigned char value,
                                                 std::size_t count, CUstream stream) {
    return native_cumemsetd8async_ptsz(pointer, value, count, stream);
}

extern "C" CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr pointer, unsigned short value,
                                           std::size_t count) {
    return native_cumemsetd16_v2(pointer, value, count);
}

extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr pointer, unsigned short value,
                                                std::size_t count) {
    return native_cumemsetd16_v2_ptds(pointer, value, count);
}

extern "C" CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr pointer, unsigned short value,
                                             std::size_t count, CUstream stream) {
    return native_cumemsetd16async(pointer, value, count, stream);
}

extern "C" CUresult CUDAAPI cuMemsetD16Async_ptsz(CUdeviceptr pointer, unsigned short value,
                                                  std::size_t count, CUstream stream) {
    return native_cumemsetd16async_ptsz(pointer, value, count, stream);
}

extern "C" CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr pointer, unsigned int value,
                                           std::size_t count) {
    return native_cumemsetd32_v2(pointer, value, count);
}

extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr pointer, unsigned int value,
                                                std::size_t count) {
    return native_cumemsetd32_v2_ptds(pointer, value, count);
}

extern "C" CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr pointer, unsigned int value,
                                             std::size_t count, CUstream stream) {
    return native_cumemsetd32async(pointer, value, count, stream);
}

extern "C" CUresult CUDAAPI cuMemsetD32Async_ptsz(CUdeviceptr pointer, unsigned int value,
                                                  std::size_t count, CUstream stream) {
    return native_cumemsetd32async_ptsz(pointer, value, count, stream);
}

extern "C" CUresult CUDAAPI cuStreamBeginCapture(CUstream stream) {
    return native_custreambegincapture(stream);
}
extern "C" CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream stream) {
    return native_custreambegincapture_ptsz(stream);
}
extern "C" CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode) {
    return native_custreambegincapture_v2(stream, mode);
}

extern "C" CUresult CUDAAPI cuStreamBeginCapture_v2_ptsz(CUstream stream,
                                                         CUstreamCaptureMode mode) {
    return native_custreambegincapture_v2_ptsz(stream, mode);
}

extern "C" CUresult CUDAAPI cuStreamEndCapture(CUstream stream, CUgraph* graph) {
    return native_custreamendcapture(stream, graph);
}

extern "C" CUresult CUDAAPI cuStreamEndCapture_ptsz(CUstream stream, CUgraph* graph) {
    return native_custreamendcapture_ptsz(stream, graph);
}

// NOLINTEND(bugprone-easily-swappable-parameters)
