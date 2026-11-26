#include <cuda.h>
#include <dlfcn.h>

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string_view>

extern "C" CUresult CUDAAPI direct_old_capture(CUstream) asm("cuStreamBeginCapture");
extern "C" CUresult CUDAAPI direct_old_capture_ptsz(CUstream) asm("cuStreamBeginCapture_ptsz");

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
template <typename Function>
Function lookup(const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
    require(function != nullptr, "ordinary dynamic symbol must resolve");
    return function;
}
using Counter = unsigned int (*)(const char*);
using NativeSymbol = void* (*)(const char*, cuuint64_t);

void verify_pointer(const char* base, std::string_view exported, cuuint64_t flags) {
    void* function = nullptr;
    CUdriverProcAddressQueryResult status{};
    const auto result = cuGetProcAddress(base, &function, 12600, flags, &status);
    if (result != CUDA_SUCCESS || status != CU_GET_PROC_ADDRESS_SUCCESS ||
        function != dlsym(RTLD_DEFAULT, exported.data())) {
        throw std::runtime_error(std::string{"modern proc-address ABI mismatch: "} +
                                 std::string{exported} + " result=" + std::to_string(result) +
                                 " status=" + std::to_string(status));
    }
}

void verify_capture_begin(bool should_reject) {
    using Old = CUresult(CUDAAPI*)(CUstream);
    using Modern = CUresult(CUDAAPI*)(CUstream, CUstreamCaptureMode);
    using OldQuery = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
    using ModernQuery =
        CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    using Query = std::function<CUresult(const char*, void**, int, cuuint64_t)>;
    const auto old_query = lookup<OldQuery>("cuGetProcAddress");
    const auto modern_query = lookup<ModernQuery>("cuGetProcAddress_v2");
    const std::array<Query, 2> queries{
        old_query, [modern_query](const char* name, void** output, int version, cuuint64_t flags) {
            return modern_query(name, output, version, flags, nullptr);
        }};
    const auto counter = lookup<Counter>("glimmer_test_native_memory_calls");
    const std::array<Old, 2> direct{&direct_old_capture, &direct_old_capture_ptsz};
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    require(driver, "explicit native Driver handle available");
    for (const bool is_ptds : {false, true}) {
        const char* old_name = is_ptds ? "cuStreamBeginCapture_ptsz" : "cuStreamBeginCapture";
        const char* modern_name =
            is_ptds ? "cuStreamBeginCapture_v2_ptsz" : "cuStreamBeginCapture_v2";
        const auto old = lookup<Old>(old_name);
        const auto modern = lookup<Modern>(modern_name);
        require(reinterpret_cast<void*>(old) != reinterpret_cast<void*>(modern) &&
                    reinterpret_cast<void*>(old) == reinterpret_cast<void*>(direct[is_ptds]) &&
                    dlsym(driver, old_name) == reinterpret_cast<void*>(old),
                "old unary capture must not alias the modern binary ABI through any lookup");
        const auto stream = reinterpret_cast<CUstream>(0x2200);
        const auto invalid_stream = reinterpret_cast<CUstream>(0xdead);
        const auto invoke_old = [&](Old function) {
            const auto before = counter(old_name);
            require(function(stream) == (should_reject ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS) &&
                        function(invalid_stream) == (should_reject ? CUDA_ERROR_NOT_SUPPORTED
                                                                   : CUDA_ERROR_INVALID_HANDLE) &&
                        counter(old_name) == before + (should_reject ? 0U : 2U),
                    "unary capture preserves exact native call/error or rejects before issue");
        };
        const auto invoke_modern = [&](Modern function) {
            const auto before = counter(modern_name);
            require(
                function(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
                        (should_reject ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS) &&
                    function(stream, CU_STREAM_CAPTURE_MODE_RELAXED) ==
                        (should_reject ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE) &&
                    function(invalid_stream, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
                        (should_reject ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_HANDLE) &&
                    counter(modern_name) == before + (should_reject ? 0U : 3U),
                "binary capture preserves mode, exact error/call count and preview rejection");
        };
        invoke_old(direct[is_ptds]);
        invoke_old(old);
        invoke_modern(modern);
        for (const auto& query : queries) {
            for (const int version : {10000, 10010, 12600}) {
                void* address{};
                const auto flags = is_ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                           : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
                require(query("cuStreamBeginCapture", &address, version, flags) == CUDA_SUCCESS &&
                            address == (version < 10010 ? reinterpret_cast<void*>(old)
                                                        : reinterpret_cast<void*>(modern)),
                        "both resolvers select exact requested capture ABI and stream flags");
                if (version < 10010) {
                    invoke_old(reinterpret_cast<Old>(address));
                } else {
                    invoke_modern(reinterpret_cast<Modern>(address));
                }
                require(query(old_name, &address, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM) ==
                                CUDA_SUCCESS &&
                            address == (version < 10010 ? reinterpret_cast<void*>(old)
                                                        : reinterpret_cast<void*>(modern)),
                        "explicit PTDS base name preserves its versioned ABI");
            }
        }
    }
    require(dlclose(driver) == 0, "release explicit Driver lookup reference");
}

void forwarding() {
    require(cuInit(0) == CUDA_SUCCESS, "fake Driver must initialize");
    const auto counter = lookup<Counter>("glimmer_test_native_memory_calls");
    const auto native_symbol = lookup<NativeSymbol>("glimmer_test_memory_symbol");
    std::array<std::byte, 8> host{};
    for (bool is_ptds : {false, true}) {
        const auto flags = is_ptds ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                   : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
        const char* upload = is_ptds ? "cuMemcpyHtoDAsync_v2_ptsz" : "cuMemcpyHtoDAsync_v2";
        const char* download = is_ptds ? "cuMemcpyDtoHAsync_v2_ptsz" : "cuMemcpyDtoHAsync_v2";
        const char* device_copy = is_ptds ? "cuMemcpyDtoDAsync_v2_ptsz" : "cuMemcpyDtoDAsync_v2";
        verify_pointer("cuMemcpyHtoDAsync", upload, flags);
        verify_pointer("cuMemcpyDtoHAsync", download, flags);
        verify_pointer("cuMemcpyDtoDAsync", device_copy, flags);
        auto h2d = lookup<decltype(&cuMemcpyHtoDAsync)>(upload);
        auto d2h = lookup<decltype(&cuMemcpyDtoHAsync)>(download);
        auto d2d = lookup<decltype(&cuMemcpyDtoDAsync)>(device_copy);
        const auto stream = reinterpret_cast<CUstream>(0x2200);
        const char* uva_sync = is_ptds ? "cuMemcpy_ptds" : "cuMemcpy";
        const char* uva_async = is_ptds ? "cuMemcpyAsync_ptsz" : "cuMemcpyAsync";
        verify_pointer("cuMemcpy", uva_sync, flags);
        verify_pointer("cuMemcpyAsync", uva_async, flags);
        require(lookup<decltype(&cuMemcpy)>(uva_sync)(0x3000, 0x2000, 8) == CUDA_SUCCESS &&
                    lookup<decltype(&cuMemcpy)>(uva_sync)(0x3000, 0x2000, 13) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    lookup<decltype(&cuMemcpyAsync)>(uva_async)(0x3000, 0x2000, 8, stream) ==
                        CUDA_SUCCESS &&
                    lookup<decltype(&cuMemcpyAsync)>(uva_async)(0x3000, 0x2000, 13, stream) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    counter(uva_sync) == 2 && counter(uva_async) == 2,
                "UVA copy signatures/PTDS aliases must preserve original errors and exact native "
                "calls");
        require(h2d(0x2000, host.data(), 8, stream) == CUDA_SUCCESS &&
                    h2d(0x2000, host.data(), 13, stream) == CUDA_ERROR_INVALID_VALUE &&
                    d2h(host.data(), 0x2000, 8, stream) == CUDA_SUCCESS &&
                    d2d(0x3000, 0x2000, 8, stream) == CUDA_SUCCESS && counter(upload) == 2 &&
                    counter(download) == 1 && counter(device_copy) == 1,
                "async copies must preserve original errors and forward once through exact PTDS");
        const char* sync_upload = is_ptds ? "cuMemcpyHtoD_v2_ptds" : "cuMemcpyHtoD_v2";
        const char* sync_download = is_ptds ? "cuMemcpyDtoH_v2_ptds" : "cuMemcpyDtoH_v2";
        const char* sync_device = is_ptds ? "cuMemcpyDtoD_v2_ptds" : "cuMemcpyDtoD_v2";
        verify_pointer("cuMemcpyHtoD", sync_upload, flags);
        verify_pointer("cuMemcpyDtoH", sync_download, flags);
        verify_pointer("cuMemcpyDtoD", sync_device, flags);
        require(
            lookup<decltype(&cuMemcpyHtoD)>(sync_upload)(0x2000, host.data(), 8) == CUDA_SUCCESS &&
                lookup<decltype(&cuMemcpyDtoH)>(sync_download)(host.data(), 0x2000, 8) ==
                    CUDA_SUCCESS &&
                lookup<decltype(&cuMemcpyDtoD)>(sync_device)(0x3000, 0x2000, 8) == CUDA_SUCCESS &&
                counter(sync_upload) == 1 && counter(sync_download) == 1 &&
                counter(sync_device) == 1,
            "synchronous PTDS aliases must not accidentally select legacy stream semantics");
    }
    void* pinned = nullptr;
    CUdeviceptr mapped{};
    require(cuMemHostAlloc(&pinned, 8, CU_MEMHOSTALLOC_DEVICEMAP) == CUDA_SUCCESS &&
                cuMemHostGetDevicePointer(&mapped, pinned, 0) == CUDA_SUCCESS &&
                mapped == reinterpret_cast<CUdeviceptr>(pinned) &&
                cuMemFreeHost(pinned) == CUDA_SUCCESS,
            "covered pinned allocation and checked free must forward unchanged");
    require(cuMemHostRegister(host.data(), 8, 0) == CUDA_SUCCESS &&
                cuMemHostUnregister(host.data()) == CUDA_SUCCESS &&
                counter("cuMemHostRegister_v2") == 1 && counter("cuMemHostUnregister") == 1,
            "host registration boundaries must forward once");
    CUstream stream{};
    int priority{};
    unsigned int stream_flags{};
    require(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, -1) == CUDA_SUCCESS &&
                cuStreamGetPriority(stream, &priority) == CUDA_SUCCESS && priority == -1 &&
                cuStreamGetFlags(stream, &stream_flags) == CUDA_SUCCESS &&
                stream_flags == CU_STREAM_NON_BLOCKING,
            "native stream priority metadata must stay intact");
    require(cuMemsetD8(0x2000, 0, 8) == CUDA_SUCCESS &&
                cuMemsetD16Async(0x2000, 0, 8, stream) == CUDA_SUCCESS &&
                cuMemsetD32(0x2000, 0, 8) == CUDA_SUCCESS,
            "native memory execution boundaries must forward");
    require(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS,
            "disabled cooperative preview must preserve native capture");
    CUgraph graph{};
    require(cuStreamEndCapture(stream, &graph) == CUDA_ERROR_INVALID_VALUE,
            "original capture error must remain unchanged");
    for (const auto& [name, version] : {std::pair{"cuMemcpyHtoD", 2000},
                                        {"cuMemcpy", 3000},
                                        {"cuMemcpyAsync", 3000},
                                        {"cuMemHostRegister", 4000}}) {
        void* address = nullptr;
        CUdriverProcAddressQueryResult status{};
        require(cuGetProcAddress(name, &address, version, CU_GET_PROC_ADDRESS_LEGACY_STREAM,
                                 &status) == CUDA_SUCCESS &&
                    address == native_symbol(name, CU_GET_PROC_ADDRESS_LEGACY_STREAM),
                "historical resolver ABI must not be substituted by a modern wrapper");
    }
    verify_capture_begin(false);
}

void rejection() {
    require(cuInit(0) == CUDA_SUCCESS, "original initialization stays available");
    const auto counter = lookup<Counter>("glimmer_test_native_memory_calls");
    std::array<std::byte, 8> host{};
    require(cuMemcpyHtoDAsync(0x2000, host.data(), 8, nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
                counter("cuMemcpyHtoDAsync_v2") == 0,
            "missing cooperative proof dispatch must reject before native copy issue");
    require(cuMemcpyAsync(0x3000, 0x2000, 8, nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
                counter("cuMemcpyAsync") == 0,
            "missing cooperative dispatch must also reject UVA copy before native issue");
    require(
        cuStreamBeginCapture(nullptr, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_ERROR_NOT_SUPPORTED &&
            counter("cuStreamBeginCapture_v2") == 0,
        "unimplemented capture must reject before native capture construction");
    verify_capture_begin(true);
}

void recursive_lookup_rejection() {
    require(cuInit(0) == CUDA_SUCCESS, "Driver must initialize before lookup rejection");
    const auto counter = lookup<Counter>("glimmer_test_native_memory_calls");
    std::array<std::byte, 8> host{};
    require(cuMemcpyHtoDAsync(0x2000, host.data(), 8, nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
                counter("cuMemcpyHtoDAsync_v2") == 0,
            "a native resolver returning this wrapper must reject rather than recurse");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view{argv[1]} == "--reject") {
            rejection();
        } else if (argc == 2 && std::string_view{argv[1]} == "--self") {
            recursive_lookup_rejection();
        } else if (argc == 1) {
            forwarding();
        } else {
            return 1;
        }
        std::cout << "driver_copy_routing status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
