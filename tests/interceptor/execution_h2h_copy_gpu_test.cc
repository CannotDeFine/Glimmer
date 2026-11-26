#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <dlfcn.h>

// One unchanged SDK input checks native and owned H2H bytes and GPU progress.
namespace {
using Clock = std::chrono::steady_clock;
using Image = std::array<unsigned char, 256>;
constexpr const char* kGate = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry h2h_release_gate(.param .u64 state) {
 .reg .pred %p;
 .reg .b32 %r;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [state];
 add.u64 %rd2, %rd1, 64;
 add.u64 %rd3, %rd1, 128;
 mov.u64 %rd4, %globaltimer;
 st.release.sys.global.u32 [%rd1], 1;
WAIT:
 ld.acquire.sys.global.u32 %r, [%rd2];
 setp.eq.u32 %p, %r, 1;
 @%p bra DONE;
 mov.u64 %rd5, %globaltimer;
 sub.u64 %rd6, %rd5, %rd4;
 setp.lt.u64 %p, %rd6, 5000000000;
 @%p bra WAIT;
 st.release.sys.global.u32 [%rd3], 2;
 ret;
DONE:
 st.release.sys.global.u32 [%rd3], 1;
 ret;
}
)ptx";
constexpr std::uint32_t kGuard = 0xc35a91e7;
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free);
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct SourceCallback {
    std::mutex mutex;
    std::condition_variable changed;
    bool may_return{};
    std::atomic<bool> entered{}, completed{}, failed{};
    std::atomic<unsigned int> calls{};
    unsigned char* source{};
    unsigned char value{};
    void release() {
        {
            std::lock_guard lock(mutex);
            may_return = true;
        }
        changed.notify_all();
    }
    static void CUDA_CB invoke(void* opaque) noexcept {
        auto& state = *static_cast<SourceCallback*>(opaque);
        try {
            state.entered.store(true, std::memory_order_release);
            {
                std::unique_lock lock(state.mutex);
                if (!state.changed.wait_for(lock, std::chrono::seconds(7),
                                            [&] { return state.may_return; }))
                    state.failed = true;
            }
            std::fill_n(state.source + 17, 128, state.value);
        } catch (...) {
            state.failed = true;
        }
        ++state.calls;
        state.completed.store(true, std::memory_order_release);
    }
};
std::string hex(const void* data, std::size_t count) {
    constexpr char digits[] = "0123456789abcdef";
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::string result;
    result.reserve(count * 2);
    for (std::size_t index = 0; index != count; ++index) {
        result += digits[bytes[index] >> 4];
        result += digits[bytes[index] & 15];
    }
    return result;
}
struct Experiment {
    CUcontext context{};
    CUmodule module{};
    CUfunction gate{};
    std::array<CUstream, 2> streams{};
    std::array<void*, 2> pinned{};
    std::uint32_t* control{};
    CUdeviceptr device_control{};
    CUdeviceptr upload_destination{};
    CUevent tail{};
    std::array<Image, 4> images{}, expected{};
    std::array<std::uint32_t, 64> pageable_word{};
    decltype(&cuMemcpy) copy{&cuMemcpy};
    decltype(&cuMemcpyAsync) asynchronous{&cuMemcpyAsync};
    decltype(&cudaLaunchHostFunc) host_function{&cudaLaunchHostFunc};
    decltype(&cuMemcpyHtoD) upload{&cuMemcpyHtoD};
    decltype(&cuMemcpyHtoDAsync) upload_async{&cuMemcpyHtoDAsync};
    using RuntimeCopy = cudaError_t(CUDARTAPI*)(void*, const void*, std::size_t, cudaMemcpyKind);
    using RuntimeAsync = cudaError_t(CUDARTAPI*)(void*, const void*, std::size_t, cudaMemcpyKind,
                                                 cudaStream_t);
    RuntimeCopy runtime_copy{&cudaMemcpy};
    RuntimeAsync runtime_async{&cudaMemcpyAsync};
    bool cached{}, uncertain{}, finished_watchdog{};
    std::atomic<bool> watchdog_released{};
    std::mutex mutex;
    std::condition_variable changed;
    std::thread watchdog;
    SourceCallback source_callback;
    std::ostringstream cases, progress;
    std::string uuid;
    int version{};
    unsigned int count{}, progress_count{};

    void check(CUresult status, const char* operation) {
        if (status != CUDA_SUCCESS) {
            uncertain = true;
            throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
        }
    }
    static CUdeviceptr address(const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer);
    }
    void start(bool selected_cached, bool needs_upload = false) {
        cached = selected_cached;
        check(cuInit(0), "initialize Driver");
        if (cached) {
            const auto resolve = [&](const char* name, int abi = 4000) {
                void* pointer{};
                CUdriverProcAddressQueryResult query{};
                check(cuGetProcAddress_v2(name, &pointer, abi, CU_GET_PROC_ADDRESS_LEGACY_STREAM,
                                          &query),
                      "resolve generic Driver entry");
                expect(pointer && query == CU_GET_PROC_ADDRESS_SUCCESS,
                       "exact requested Driver ABI");
                return pointer;
            };
            copy = reinterpret_cast<decltype(copy)>(resolve("cuMemcpy"));
            asynchronous = reinterpret_cast<decltype(asynchronous)>(resolve("cuMemcpyAsync"));
            host_function = reinterpret_cast<decltype(host_function)>(
                ::dlsym(RTLD_DEFAULT, "cudaLaunchHostFunc"));
            expect(host_function != nullptr, "resolve public cached Runtime host-function entry");
            if (needs_upload) {
                upload = reinterpret_cast<decltype(upload)>(resolve("cuMemcpyHtoD", 3020));
                upload_async =
                    reinterpret_cast<decltype(upload_async)>(resolve("cuMemcpyHtoDAsync", 3020));
                runtime_copy = reinterpret_cast<RuntimeCopy>(::dlsym(RTLD_DEFAULT, "cudaMemcpy"));
                runtime_async =
                    reinterpret_cast<RuntimeAsync>(::dlsym(RTLD_DEFAULT, "cudaMemcpyAsync"));
                expect(runtime_copy && runtime_async, "resolve public cached Runtime copy entries");
            }
        }
        check(cuDriverGetVersion(&version), "read Driver version");
        CUdevice device{};
        check(cuDeviceGet(&device, 0), "select device");
        check(cuCtxCreate(&context, CU_CTX_MAP_HOST, device), "create mapped context");
        if (needs_upload)
            check(cuMemAlloc(&upload_destination, 256), "allocate complete guarded upload image");
        CUuuid identity{};
        check(cuDeviceGetUuid(&identity, device), "identify actual GPU");
        const auto raw = hex(identity.bytes, sizeof(identity.bytes));
        uuid = "GPU-" + raw.substr(0, 8) + '-' + raw.substr(8, 4) + '-' + raw.substr(12, 4) + '-' +
               raw.substr(16, 4) + '-' + raw.substr(20);
        check(cuStreamCreate(&streams[0], CU_STREAM_DEFAULT), "create blocking stream");
        check(cuStreamCreate(&streams[1], CU_STREAM_NON_BLOCKING), "create independent stream");
        for (auto& pointer : pinned)
            check(
                cuMemHostAlloc(&pointer, 256, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
                "create public mapped HST image");
        void* mapping{};
        check(cuMemHostAlloc(&mapping, 256, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
              "create mapped GPU control");
        control = static_cast<std::uint32_t*>(mapping);
        check(cuMemHostGetDevicePointer(&device_control, mapping, 0), "query public mapped alias");
        check(cuModuleLoadData(&module, kGate), "load ordinary gate PTX");
        check(cuModuleGetFunction(&gate, module, "h2h_release_gate"), "resolve ordinary gate");
        check(cuEventCreate(&tail, CU_EVENT_DISABLE_TIMING), "create gate completion event");
    }
    void stop_watchdog() {
        {
            std::lock_guard lock(mutex);
            finished_watchdog = true;
        }
        changed.notify_all();
        if (watchdog.joinable()) watchdog.join();
    }
    void self_release(bool async, bool pinned_source) {
        std::fill_n(control, 64, kGuard);
        control[0] = control[16] = control[32] = 0;
        pageable_word.fill(kGuard);
        pageable_word[16] = 1;
        std::memcpy(pinned[0], pageable_word.data(), 256);
        const auto source_before = hex(pinned_source ? pinned[0] : pageable_word.data(), 256);
        void* arguments[]{&device_control};
        check(cuLaunchKernel(gate, 1, 1, 1, 1, 1, 1, 0, streams[1], arguments, nullptr),
              "launch independently held GPU gate");
        check(cuEventRecord(tail, streams[1]), "record exact gate completion");
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        while (std::atomic_ref{control[0]}.load(std::memory_order_acquire) != 1) {
            expect(Clock::now() < deadline, "GPU did not publish gate readiness");
            std::this_thread::yield();
        }
        expect(cuEventQuery(tail) == CUDA_ERROR_NOT_READY && control[16] == 0 && control[32] == 0,
               "real predecessor is independently held before the copy");
        finished_watchdog = false;
        watchdog_released = false;
        watchdog = std::thread([&] {
            std::unique_lock lock(mutex);
            if (!changed.wait_for(lock, std::chrono::seconds(2),
                                  [&] { return finished_watchdog; })) {
                watchdog_released = true;
                std::atomic_ref{control[16]}.store(1, std::memory_order_release);
            }
        });
        const auto source = address(pinned_source ? pinned[0] : pageable_word.data()) + 64;
        const auto status = async ? asynchronous(address(control + 16), source, 4, nullptr)
                                  : copy(address(control + 16), source, 4);
        check(status, "original H2H self-release copy");
        check(cuEventSynchronize(tail), "complete self-released GPU gate");
        check(cuStreamSynchronize(nullptr), "complete actual H2H default stream");
        stop_watchdog();
        bool guards = true;
        for (std::size_t index = 0; index != 64; ++index)
            guards = guards &&
                     control[index] == (index == 0 || index == 16 || index == 32 ? 1U : kGuard);
        const auto source_after = hex(pinned_source ? pinned[0] : pageable_word.data(), 256);
        expect(guards && source_before == source_after, "complete release-word and source images");
        // Watchdog recovery is deliberately reported as failure, not progress.
        expect(!watchdog_released, "H2H progress required CPU watchdog recovery");
        if (progress_count) progress << ',';
        progress << "{\"ordinal\":" << progress_count << ",\"async\":" << (async ? "true" : "false")
                 << ",\"pinned_source\":" << (pinned_source ? "true" : "false")
                 << ",\"status\":0,\"initial_event_status\":600,\"physical_event_status\":0,"
                    "\"watchdog_released\":false,\"control_image\":\""
                 << hex(control, 256) << "\",\"source_image\":\"" << source_after << "\"}";
        ++progress_count;
    }
    void matrix(std::size_t bytes, unsigned int mode, unsigned int route) {
        for (std::size_t image = 0; image != images.size(); ++image)
            for (std::size_t index = 0; index != 256; ++index)
                images[image][index] = static_cast<unsigned char>(
                    index < 16 || index >= 240
                        ? 0x80 + image
                        : (index * 17 + image * 31 + static_cast<std::size_t>(count) * 7) & 255);
        expected = images;
        for (std::size_t index = 0; index != pinned.size(); ++index)
            std::memcpy(pinned[index], images[index + 2].data(), 256);
        const std::size_t source = route & 1U ? 2 : 0;
        const std::size_t destination = route & 2U ? 3 : 1;
        const auto endpoint = [&](std::size_t index) {
            return address(index < 2 ? images[index].data() : pinned[index - 2]);
        };
        std::copy_n(expected[source].begin() + 17, bytes, expected[destination].begin() + 43);
        const auto stream = mode >= 2 ? streams[mode - 2] : nullptr;
        const auto status =
            mode ? asynchronous(endpoint(destination) + 43, endpoint(source) + 17, bytes, stream)
                 : copy(endpoint(destination) + 43, endpoint(source) + 17, bytes);
        check(status, "original settled H2H matrix copy");
        check(cuStreamSynchronize(stream), "complete exact H2H matrix stream");
        for (std::size_t index = 0; index != pinned.size(); ++index)
            std::memcpy(images[index + 2].data(), pinned[index], 256);
        expect(images == expected, "complete four-image H2H byte oracle");
        if (count) cases << ',';
        cases << "{\"ordinal\":" << count << ",\"bytes\":" << bytes << ",\"mode\":" << mode
              << ",\"route\":" << route << ",\"status\":0,\"images\":[";
        for (std::size_t index = 0; index != images.size(); ++index) {
            if (index) cases << ',';
            cases << '"' << hex(images[index].data(), 256) << '"';
        }
        cases << "]}";
        ++count;
    }
    void source_order(bool async, unsigned int stream_kind, unsigned int route,
                      unsigned int api = 3) {
        const bool is_upload = api < 3;
        for (std::size_t image = 0; image != images.size(); ++image)
            images[image].fill(static_cast<unsigned char>(0x80 + image));
        const std::size_t source = route & 1U ? 2 : 0;
        const std::size_t destination = !is_upload && (route & 2U) ? 3 : 1;
        std::fill_n(images[source].begin() + 17, 128, 0x11);
        for (std::size_t index = 0; index != pinned.size(); ++index)
            std::memcpy(pinned[index], images[index + 2].data(), 256);
        const auto pointer = [&](std::size_t index) {
            return index < 2 ? images[index].data()
                             : static_cast<unsigned char*>(pinned[index - 2]);
        };
        auto& callback = source_callback;
        if (is_upload) {
            check(copy(upload_destination, address(images[1].data()), 256),
                  "initialize full device guards");
            check(cuCtxSynchronize(), "complete upload setup before callback");
        }
        callback.may_return = false;
        callback.entered = callback.completed = callback.failed = false;
        callback.calls = 0;
        callback.source = pointer(source);
        callback.value = 0x5d;
        const auto stream = stream_kind == 3 ? CU_STREAM_PER_THREAD
                            : stream_kind    ? streams[stream_kind - 1]
                                             : nullptr;
        const auto enqueued = host_function(stream, SourceCallback::invoke, &callback);
        if (enqueued != cudaSuccess) {
            uncertain = true;
            throw std::runtime_error("enqueue actual Runtime source callback: " +
                                     std::to_string(enqueued));
        }
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        while (!callback.entered.load(std::memory_order_acquire)) {
            expect(Clock::now() < deadline, "source callback did not publish entry");
            std::this_thread::yield();
        }
        expect(!callback.completed && !callback.calls, "source producer is independently held");
        finished_watchdog = false;
        watchdog_released = false;
        watchdog = std::thread([&] {
            std::unique_lock lock(mutex);
            if (!changed.wait_for(lock, std::chrono::seconds(2),
                                  [&] { return finished_watchdog; })) {
                watchdog_released = true;
                callback.release();
            }
        });
        const auto target =
            is_upload ? upload_destination + 43 : address(pointer(destination) + 43);
        CUresult status{};
        if (api == 2) {
            auto* device_pointer = std::bit_cast<void*>(target);
            const auto result = async ? runtime_async(device_pointer, pointer(source) + 17, 128,
                                                      cudaMemcpyHostToDevice, stream)
                                      : runtime_copy(device_pointer, pointer(source) + 17, 128,
                                                     cudaMemcpyHostToDevice);
            if (result != cudaSuccess) {
                uncertain = true;
                throw std::runtime_error("original Runtime upload: " + std::to_string(result));
            }
        } else if (api == 1) {
            status = async ? upload_async(target, pointer(source) + 17, 128, stream)
                           : upload(target, pointer(source) + 17, 128);
        } else {
            status = async ? asynchronous(target, address(pointer(source) + 17), 128, stream)
                           : copy(target, address(pointer(source) + 17), 128);
        }
        const bool callback_complete_at_return = callback.completed.load(std::memory_order_acquire);
        const bool released_before_return = watchdog_released.load();
        callback.release();
        stop_watchdog();
        check(status, "original callback-dependent H2H copy");
        check(cuCtxSynchronize(), "join real copy and source callback");
        if (is_upload) {
            check(copy(address(images[1].data()), upload_destination, 256),
                  "read back complete device image");
            check(cuCtxSynchronize(), "complete full device readback");
        }
        expect(callback.completed && callback.calls == 1 && !callback.failed,
               "source callback returns exactly once before image access");
        for (std::size_t index = 0; index != pinned.size(); ++index)
            std::memcpy(images[index + 2].data(), pinned[index], 256);
        auto before = expected = images;
        // Build both legal whole-image alternatives independently of the
        // observed destination. Partial/guard/source writes match neither.
        for (std::size_t image = 0; image != before.size(); ++image) {
            before[image].fill(static_cast<unsigned char>(0x80 + image));
            expected[image] = before[image];
        }
        std::fill_n(before[source].begin() + 17, 128, 0x5d);
        expected = before;
        std::fill_n(before[destination].begin() + 43, 128, 0x11);
        std::fill_n(expected[destination].begin() + 43, 128, 0x5d);
        const bool copied_before = images == before;
        expect(copied_before || images == expected, "complete source-order image and guard oracle");
        if (count) cases << ',';
        cases << "{\"ordinal\":" << count << ",\"async\":" << (async ? "true" : "false")
              << ",\"stream_kind\":" << stream_kind;
        if (is_upload)
            cases << ",\"api\":" << api << ",\"pinned_source\":" << (route ? "true" : "false");
        else
            cases << ",\"route\":" << route;
        cases << ",\"status\":0,\"callback_calls\":1,\"callback_complete_at_return\":"
              << (callback_complete_at_return ? "true" : "false")
              << ",\"watchdog_before_return\":" << (released_before_return ? "true" : "false")
              << ",\"copied_before_callback\":" << (copied_before ? "true" : "false")
              << ",\"images\":[";
        for (std::size_t index = 0; index != images.size(); ++index) {
            if (index) cases << ',';
            cases << '"' << hex(images[index].data(), 256) << '"';
        }
        cases << "]}";
        ++count;
    }
    void close() {
        expect(!uncertain, "uncertain CUDA work cannot qualify cleanup");
        check(cuCtxSynchronize(), "final exact context completion");
        if (upload_destination) {
            check(cuMemFree(upload_destination), "release proven upload image");
            upload_destination = 0;
        }
        check(cuEventDestroy(tail), "release proven gate event");
        check(cuModuleUnload(module), "release proven gate module");
        for (auto& pointer : pinned) {
            check(cuMemFreeHost(pointer), "release mapped HST image");
            pointer = nullptr;
        }
        check(cuMemFreeHost(control), "release proven mapped GPU control");
        control = nullptr;
        for (auto stream : streams) check(cuStreamDestroy(stream), "release proven stream");
        check(cuCtxDestroy(context), "release proven application context");
        context = nullptr;
    }
};
}  // namespace

int main(int argc, char** argv) {
    std::cout << std::boolalpha;
    auto experiment = std::make_unique<Experiment>();
    try {
        bool cached{}, source_order{}, upload_order{}, per_thread_order{};
        for (int index = 1; index != argc; ++index) {
            if (std::string_view(argv[index]) == "--cached")
                cached = true;
            else if (std::string_view(argv[index]) == "--source-order")
                source_order = true;
            else if (std::string_view(argv[index]) == "--upload-order")
                upload_order = true;
            else if (std::string_view(argv[index]) == "--per-thread-order")
                per_thread_order = true;
            else
                throw std::runtime_error(
                    "usage: h2h-copy-gpu-test [--cached] "
                    "[--source-order|--upload-order|--per-thread-order]");
        }
        expect(static_cast<unsigned int>(source_order) + static_cast<unsigned int>(upload_order) +
                       static_cast<unsigned int>(per_thread_order) <=
                   1,
               "select only one source-order matrix");
        experiment->start(cached, upload_order);
        if (source_order || upload_order || per_thread_order) {
            for (unsigned int api = upload_order ? 0U : 3U; api != (upload_order ? 3U : 4U); ++api)
                for (const bool async : {false, true})
                    for (unsigned int stream = per_thread_order ? 3U : 0U;
                         stream != (per_thread_order ? 4U : 3U); ++stream)
                        for (unsigned int route = 0; route != (upload_order ? 2U : 4U); ++route)
                            experiment->source_order(async, stream, route, api);
            experiment->close();
            std::cout
                << "{\"record\":\""
                << (per_thread_order ? "h2h_per_thread_order_result"
                    : upload_order   ? "h2d_order_result"
                                     : "h2h_order_result")
                << "\",\"schema\":1,\"cached\":" << cached << ",\"gpu_uuid\":\"" << experiment->uuid
                << "\",\"driver_version\":" << experiment->version
                << ",\"case_count\":" << experiment->count
                << ",\"normal_cleanup\":true,\"full_device_qualified\":false,"
                   "\"priority_qualified\":false,\"formal_comparison_qualified\":false,\"cases\":["
                << experiment->cases.str() << "]}\n";
            return EXIT_SUCCESS;
        }
        for (const bool async : {false, true})
            for (const bool pinned_source : {false, true})
                experiment->self_release(async, pinned_source);
        for (const auto bytes : {std::size_t{32}, std::size_t{65}})
            for (unsigned int mode = 0; mode != 4; ++mode)
                for (unsigned int route = 0; route != 4; ++route)
                    experiment->matrix(bytes, mode, route);
        experiment->close();
        std::cout << "{\"record\":\"h2h_copy_result\",\"schema\":1,\"cached\":" << cached
                  << ",\"gpu_uuid\":\"" << experiment->uuid
                  << "\",\"driver_version\":" << experiment->version
                  << ",\"case_count\":" << experiment->count
                  << ",\"progress_count\":" << experiment->progress_count
                  << ",\"original_copy_calls\":36,\"normal_cleanup\":true,"
                     "\"full_device_qualified\":false,\"priority_qualified\":false,"
                     "\"formal_comparison_qualified\":false,\"progress\":["
                  << experiment->progress.str() << "],\"cases\":[" << experiment->cases.str()
                  << "]}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        experiment->source_callback.release();
        if (experiment->control)
            std::atomic_ref{experiment->control[16]}.store(1, std::memory_order_release);
        experiment->stop_watchdog();
        std::cerr << "H2H SDK control failed: " << error.what() << '\n';
        // Keep all buffers, mappings and handles until authenticated process
        // containment after any failed native/physical/byte observation.
        (void)experiment.release();
        std::_Exit(EXIT_FAILURE);
    }
}
