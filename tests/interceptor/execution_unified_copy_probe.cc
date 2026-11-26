#include <cuda.h>

#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <poll.h>
#include <unistd.h>

// Native-only observations, not framework compatibility or timing assertions.
// A positive witness requires a returned copy and a still-closed GPU gate.
// Safety watchdog expiration never proves that the copy depends on that gate.
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kBytes = 256, kOffset = 64, kPayload = 128, kWords = 64;
constexpr std::size_t kReady = 16, kRelease = 32, kDone = 48;
constexpr std::uint32_t kGuard = 0xc35a91e7U;
using Image = std::array<unsigned char, kBytes>;
using Copy = decltype(&cuMemcpy);
using CopyAsync = decltype(&cuMemcpyAsync);
static_assert(std::atomic_ref<std::uint32_t>::is_always_lock_free);
constexpr const char* kKernel = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry unified_copy_gate(.param .u64 state) {
 .reg .pred %p;
 .reg .b32 %r<3>;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [state];
 add.u64 %rd2, %rd1, 64;
 add.u64 %rd3, %rd1, 128;
 add.u64 %rd4, %rd1, 192;
 mov.u64 %rd5, %globaltimer;
 st.release.sys.global.u32 [%rd2], 1;
WAIT:
 ld.acquire.sys.global.u32 %r1, [%rd3];
 setp.eq.u32 %p, %r1, 1;
 @%p bra RELEASED;
 mov.u64 %rd6, %globaltimer;
 sub.u64 %rd7, %rd6, %rd5;
 setp.lt.u64 %p, %rd7, 5000000000;
 @%p bra WAIT;
 st.release.sys.global.u32 [%rd4], 2;
 ret;
RELEASED:
 st.release.sys.global.u32 [%rd4], 1;
 ret;
}
)ptx";

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value, const char* message) {
    if (value != CUDA_SUCCESS)
        throw std::runtime_error(std::string(message) + ": CUDA " + std::to_string(value));
}
std::string quote(std::string_view value) {
    std::string result = "\"";
    constexpr char digits[] = "0123456789abcdef";
    for (unsigned char character : value) {
        if (character == '\\' || character == '"') result += '\\';
        if (character < 32) {
            result += "\\u00";
            result += digits[character >> 4];
            result += digits[character & 15];
        } else {
            result += static_cast<char>(character);
        }
    }
    return result + '"';
}
void end_record() {
    std::cout << '\n' << std::flush;
    expect(std::cout.good(), "record publication failed");
}
std::string hex(const void* pointer, std::size_t bytes) {
    constexpr char digits[] = "0123456789abcdef";
    const auto* values = static_cast<const unsigned char*>(pointer);
    std::string result;
    result.reserve(bytes * 2);
    for (std::size_t index = 0; index != bytes; ++index) {
        result += digits[values[index] >> 4];
        result += digits[values[index] & 15];
    }
    return result;
}
std::string uuid_text(const CUuuid& value) {
    const auto raw = hex(value.bytes, sizeof(value.bytes));
    return "GPU-" + raw.substr(0, 8) + '-' + raw.substr(8, 4) + '-' + raw.substr(12, 4) + '-' +
           raw.substr(16, 4) + '-' + raw.substr(20);
}
Image initial_source() {
    Image result{};
    result.fill(0xd3);
    for (std::size_t index = 0; index != kPayload; ++index)
        result[index + kOffset] = static_cast<unsigned char>((index * 17 + 3) & 255);
    return result;
}
Image initial_destination() {
    Image result{};
    result.fill(0xa5);
    return result;
}
void await_run() {
    const auto deadline = Clock::now() + std::chrono::seconds(15);
    std::string input;
    while (input.size() < 4) {
        const auto remaining = deadline - Clock::now();
        expect(remaining > Clock::duration::zero(), "READY gate deadline");
        pollfd descriptor{STDIN_FILENO, POLLIN, 0};
        const auto milliseconds = std::chrono::ceil<std::chrono::milliseconds>(remaining).count();
        const auto ready = ::poll(&descriptor, 1, static_cast<int>(milliseconds));
        if (ready < 0 && errno == EINTR) continue;
        expect(ready > 0 && (descriptor.revents & (POLLIN | POLLHUP)) != 0 &&
                   (descriptor.revents & (POLLERR | POLLNVAL)) == 0,
               "READY gate unavailable");
        std::array<char, 5> buffer{};
        const auto count = ::read(STDIN_FILENO, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        expect(count > 0, "READY gate closed");
        input.append(buffer.data(), static_cast<std::size_t>(count));
        expect(input.size() <= 4 && std::string_view("run\n").starts_with(input),
               "READY gate requires exactly run newline");
    }
}

struct Case {
    std::string id;
    std::string_view route;
    bool is_async{}, is_per_thread{}, is_source_pinned{}, is_destination_pinned{};
    unsigned int stream_kind{};  // 0 legacy, 1 blocking, 2 nonblocking, 3 per-thread.
    bool is_boundary{}, has_current{true}, is_zero{};
};
std::vector<Case> inventory(std::string_view mode) {
    const std::array<Case, 7> routes{{
        {"", "sync_legacy", false, false, false, false, 0},
        {"", "sync_ptds", false, true, false, false, 3},
        {"", "async_legacy", true, false, false, false, 0},
        {"", "async_blocking", true, false, false, false, 1},
        {"", "async_nonblocking", true, false, false, false, 2},
        {"", "async_ptsz", true, true, false, false, 3},
        {"", "async_per_thread_handle", true, false, false, false, 3},
    }};
    std::vector<Case> result;
    if (mode != "boundaries") {
        for (const auto& route : routes) {
            for (unsigned int memory = 0; memory != 4; ++memory) {
                auto test = route;
                test.is_source_pinned = (memory & 1U) != 0;
                test.is_destination_pinned = (memory & 2U) != 0;
                test.id = std::string(route.route) + "_" + std::to_string(memory);
                result.push_back(test);
            }
        }
    }
    if (mode != "held") {
        for (std::size_t route : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{5}}) {
            for (unsigned int boundary = 0; boundary != 4; ++boundary) {
                auto test = routes[route];
                test.is_boundary = true;
                test.has_current = (boundary & 1U) == 0;
                test.is_zero = (boundary & 2U) != 0;
                test.id = std::string(test.route) + "_boundary_" + std::to_string(boundary);
                result.push_back(test);
            }
        }
    }
    return result;
}
void descriptor(const Case& test) {
    std::cout << "\"id\":" << quote(test.id) << ",\"route\":" << quote(test.route)
              << ",\"async\":" << test.is_async << ",\"per_thread_entry\":" << test.is_per_thread
              << ",\"source_pinned\":" << test.is_source_pinned
              << ",\"destination_pinned\":" << test.is_destination_pinned
              << ",\"stream_kind\":" << test.stream_kind << ",\"boundary\":" << test.is_boundary
              << ",\"current_context\":" << test.has_current
              << ",\"null_pointers\":" << test.is_zero
              << ",\"copy_bytes\":" << (test.is_zero ? 0 : kPayload)
              << ",\"storage_bytes\":" << kBytes << ",\"offset\":" << kOffset;
}
struct Buffer {
    Image pageable{};
    unsigned char* data{pageable.data()};
    bool is_pinned{};
    void allocate(bool pinned, const Image& initial) {
        is_pinned = pinned;
        if (pinned) {
            void* pointer{};
            check(cuMemHostAlloc(&pointer, kBytes, 0), "allocate pinned copy buffer");
            data = static_cast<unsigned char*>(pointer);
        }
        std::memcpy(data, initial.data(), kBytes);
    }
    void close() {
        if (is_pinned) {
            check(cuMemFreeHost(data), "free pinned copy buffer");
            is_pinned = false;
            data = pageable.data();
        }
    }
};
struct Trial {
    Buffer source, destination;
    std::uint32_t* control{};
    CUdeviceptr device_control{};
    CUstream stream{};
    CUevent tail{};
    std::thread worker, watchdog;
    std::mutex mutex;
    std::condition_variable changed;
    bool is_gate_ready{}, is_authorized{}, should_stop_watchdog{}, is_worker_done{};
    std::atomic<bool> has_returned{}, was_watchdog_released{};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::string error;

    std::atomic_ref<std::uint32_t> word(std::size_t index) const {
        return std::atomic_ref<std::uint32_t>(control[index]);
    }
    void release() const {
        if (control) word(kRelease).store(1, std::memory_order_release);
    }
    void stop_watchdog() {
        {
            std::lock_guard lock(mutex);
            should_stop_watchdog = true;
        }
        changed.notify_all();
    }
    void finish_threads() {
        release();
        stop_watchdog();
        {
            std::unique_lock lock(mutex);
            is_authorized = true;
            changed.notify_all();
            if (worker.joinable())
                expect(changed.wait_until(lock, Clock::now() + std::chrono::seconds(7),
                                          [&] { return is_worker_done; }),
                       "worker failed to finish after independent release");
        }
        if (worker.joinable()) worker.join();
        if (watchdog.joinable()) watchdog.join();
    }
};
struct Runner {
    CUcontext context{};
    CUmodule module{};
    CUfunction gate{};
    Copy copy_ptds{};
    CopyAsync copy_ptsz{};
    std::unique_ptr<Trial> trial;
    std::string gpu_uuid;
    int driver_version{};
    std::size_t completed{}, positive{};

    void initialize() {
        check(cuInit(0), "initialize Driver");
        CUdevice device{};
        check(cuDeviceGet(&device, 0), "select device");
        int unified{}, mapped{}, major{};
        check(cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device),
              "UVA");
        check(cuDeviceGetAttribute(&mapped, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device),
              "mapped host capability");
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
              "SM version");
        expect(unified != 0 && mapped != 0 && major >= 7, "requires UVA mapped host and SM70");
        CUuuid uuid{};
        check(cuDeviceGetUuid(&uuid, device), "GPU UUID");
        gpu_uuid = uuid_text(uuid);
        check(cuDriverGetVersion(&driver_version), "Driver version");
        check(cuCtxCreate(&context, CU_CTX_MAP_HOST, device), "create native context");
        check(cuModuleLoadData(&module, kKernel), "load native gate PTX");
        check(cuModuleGetFunction(&gate, module, "unified_copy_gate"), "resolve native gate");
        const auto resolve = [](const char* name) {
            void* pointer{};
            CUdriverProcAddressQueryResult status{};
            check(cuGetProcAddress_v2(name, &pointer, 12000,
                                      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status),
                  "resolve per-thread native entry");
            expect(pointer != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS,
                   "per-thread native entry absent");
            return pointer;
        };
        copy_ptds = reinterpret_cast<Copy>(resolve("cuMemcpy"));
        copy_ptsz = reinterpret_cast<CopyAsync>(resolve("cuMemcpyAsync"));
        check(cuCtxSynchronize(), "complete native READY setup");
    }
    void identity() const {
        std::cout << "\"pid\":" << ::getpid() << ",\"gpu_uuid\":" << quote(gpu_uuid)
                  << ",\"driver_version\":" << driver_version;
    }
    CUresult invoke(const Case& test, CUstream stream) const {
        const auto destination =
            test.is_zero ? CUdeviceptr{}
                         : std::bit_cast<CUdeviceptr>(trial->destination.data + kOffset);
        const auto source =
            test.is_zero ? CUdeviceptr{} : std::bit_cast<CUdeviceptr>(trial->source.data + kOffset);
        const auto bytes = test.is_zero ? 0 : kPayload;
        if (!test.is_async)
            return (test.is_per_thread ? copy_ptds : &cuMemcpy)(destination, source, bytes);
        return (test.is_per_thread ? copy_ptsz : &cuMemcpyAsync)(
            destination, source, bytes, test.is_per_thread ? nullptr : stream);
    }
    bool verify_images(CUresult status, bool is_zero) const {
        const auto source = initial_source();
        auto destination = initial_destination();
        if (status == CUDA_SUCCESS && !is_zero)
            std::memcpy(destination.data() + kOffset, source.data() + kOffset, kPayload);
        return std::memcmp(source.data(), trial->source.data, kBytes) == 0 &&
               std::memcmp(destination.data(), trial->destination.data, kBytes) == 0;
    }
    void images() const {
        std::cout << ",\"source_image\":" << quote(hex(trial->source.data, kBytes))
                  << ",\"destination_image\":" << quote(hex(trial->destination.data, kBytes));
    }
    void prepare(const Case& test) {
        trial = std::make_unique<Trial>();
        trial->source.allocate(test.is_source_pinned, initial_source());
        trial->destination.allocate(test.is_destination_pinned, initial_destination());
        if (test.is_boundary) return;
        void* pointer{};
        check(cuMemHostAlloc(&pointer, kWords * sizeof(std::uint32_t), CU_MEMHOSTALLOC_DEVICEMAP),
              "allocate independent mapped gate");
        trial->control = static_cast<std::uint32_t*>(pointer);
        expect(reinterpret_cast<std::uintptr_t>(pointer) %
                       std::atomic_ref<std::uint32_t>::required_alignment ==
                   0,
               "mapped gate alignment");
        for (std::size_t index = 0; index != kWords; ++index) trial->control[index] = kGuard;
        trial->control[kReady] = trial->control[kRelease] = trial->control[kDone] = 0;
        check(cuMemHostGetDevicePointer(&trial->device_control, pointer, 0),
              "map gate device address");
        check(cuEventCreate(&trial->tail, CU_EVENT_DISABLE_TIMING), "create independent tail");
        if (test.stream_kind == 1 || test.stream_kind == 2)
            check(cuStreamCreate(&trial->stream, test.stream_kind == 1 ? CU_STREAM_DEFAULT
                                                                       : CU_STREAM_NON_BLOCKING),
                  "create original application stream");
    }
    void original_return(const Case& test) const {
        std::cout << "{\"record\":\"uva_copy_return\",\"schema\":1,";
        descriptor(test);
        std::cout << ",\"pid\":" << ::getpid() << ",\"status\":" << static_cast<int>(trial->status)
                  << ",\"complete\":false}";
        end_record();
    }
    void close_trial() {
        if (trial->tail) {
            check(cuEventDestroy(trial->tail), "destroy proven tail");
            trial->tail = nullptr;
        }
        if (trial->stream) {
            check(cuStreamDestroy(trial->stream), "destroy proven stream");
            trial->stream = nullptr;
        }
        if (trial->control) {
            check(cuMemFreeHost(trial->control), "free proven gate");
            trial->control = nullptr;
        }
        trial->source.close();
        trial->destination.close();
    }
    void held(const Case& test) {
        prepare(test);
        auto& state = *trial;
        const auto stream = test.stream_kind == 3 ? CU_STREAM_PER_THREAD : state.stream;
        state.worker = std::thread([&, stream] {
            try {
                check(cuCtxSetCurrent(context), "worker current context");
                void* arguments[]{&state.device_control};
                check(cuLaunchKernel(gate, 1, 1, 1, 1, 1, 1, 0, stream, arguments, nullptr),
                      "launch one-thread original gate");
                check(cuEventRecord(state.tail, stream), "record exact gate tail");
                const auto ready_deadline = Clock::now() + std::chrono::seconds(2);
                while (state.word(kReady).load(std::memory_order_acquire) != 1) {
                    expect(Clock::now() < ready_deadline, "GPU gate did not publish readiness");
                    std::this_thread::yield();
                }
                {
                    std::unique_lock lock(state.mutex);
                    state.is_gate_ready = true;
                    state.changed.notify_all();
                    expect(state.changed.wait_until(lock, Clock::now() + std::chrono::seconds(2),
                                                    [&] { return state.is_authorized; }),
                           "controller did not authorize original call");
                }
                state.status = invoke(test, stream);
                {
                    // Pair notification with the wait mutex; an atomic flag
                    // alone would permit a lost condition-variable wakeup.
                    std::lock_guard lock(state.mutex);
                    state.has_returned.store(true, std::memory_order_release);
                }
                state.changed.notify_all();
                check(cuCtxSetCurrent(nullptr), "detach worker context");
            } catch (const std::exception& error) {
                std::lock_guard lock(state.mutex);
                state.error = error.what();
            }
            {
                std::lock_guard lock(state.mutex);
                state.is_worker_done = true;
            }
            state.changed.notify_all();
        });
        {
            std::unique_lock lock(state.mutex);
            expect(state.changed.wait_until(
                       lock, Clock::now() + std::chrono::seconds(3),
                       [&] { return state.is_gate_ready || state.is_worker_done; }),
                   "gate setup worker deadline");
            expect(state.error.empty() && state.is_gate_ready, "gate setup failed");
        }
        const auto initial_query = cuEventQuery(state.tail);
        expect(initial_query == CUDA_ERROR_NOT_READY && state.word(kRelease).load() == 0 &&
                   state.word(kDone).load() == 0,
               "GPU predecessor was not independently held");
        state.watchdog = std::thread([&] {
            std::unique_lock lock(state.mutex);
            if (!state.changed.wait_until(lock, Clock::now() + std::chrono::seconds(2),
                                          [&] { return state.should_stop_watchdog; })) {
                state.was_watchdog_released.store(true, std::memory_order_release);
                state.release();
            }
        });
        {
            std::lock_guard lock(state.mutex);
            state.is_authorized = true;
        }
        state.changed.notify_all();
        {
            std::unique_lock lock(state.mutex);
            expect(state.changed.wait_until(lock, Clock::now() + std::chrono::seconds(7),
                                            [&] {
                                                return state.has_returned.load(
                                                           std::memory_order_acquire) ||
                                                       state.is_worker_done;
                                            }),
                   "original copy did not return after watchdog release");
            expect(state.error.empty() && state.has_returned.load(std::memory_order_acquire),
                   "original worker failed");
        }
        original_return(test);
        const bool are_bytes_correct = verify_images(state.status, false);
        const auto release_before = state.word(kRelease).load(std::memory_order_acquire);
        const auto return_query = cuEventQuery(state.tail);
        const auto done_at_return = state.word(kDone).load(std::memory_order_acquire);
        const auto release_after = state.word(kRelease).load(std::memory_order_acquire);
        expect(return_query == CUDA_SUCCESS || return_query == CUDA_ERROR_NOT_READY,
               "tail query failed during original observation");
        const bool is_positive = state.status == CUDA_SUCCESS && are_bytes_correct &&
                                 release_before == 0 && release_after == 0 && done_at_return == 0 &&
                                 return_query == CUDA_ERROR_NOT_READY;
        state.finish_threads();
        expect(state.error.empty(), "worker cleanup failed");
        check(cuEventSynchronize(state.tail), "independent physical gate completion");
        for (std::size_t index = 0; index != kWords; ++index)
            expect(state.control[index] ==
                       (index == kReady || index == kRelease || index == kDone ? 1U : kGuard),
                   "gate guard corruption or GPU hard watchdog expiration");
        expect(are_bytes_correct && verify_images(state.status, false), "full H2H byte oracle");
        const auto control_image = hex(state.control, kWords * sizeof(std::uint32_t));
        const auto source_image = hex(state.source.data, kBytes);
        const auto destination_image = hex(state.destination.data, kBytes);
        const auto status = state.status;
        const bool was_watchdog_released = state.was_watchdog_released.load();
        close_trial();
        std::cout << "{\"record\":\"uva_copy_observation\",\"schema\":1,";
        descriptor(test);
        std::cout
            << ",\"pid\":" << ::getpid() << ",\"status\":" << static_cast<int>(status)
            << ",\"initial_event_status\":" << static_cast<int>(initial_query)
            << ",\"return_event_status\":" << static_cast<int>(return_query)
            << ",\"release_before_query\":" << release_before
            << ",\"release_after_query\":" << release_after
            << ",\"done_at_query\":" << done_at_return
            << ",\"watchdog_released\":" << was_watchdog_released
            << ",\"returned_while_held\":" << is_positive << ",\"observation\":"
            << quote(is_positive ? "positive_early_return" : "no_early_witness")
            << ",\"source_image\":" << quote(source_image)
            << ",\"destination_image\":" << quote(destination_image)
            << ",\"control_image\":" << quote(control_image)
            << ",\"physical_event_status\":0,\"full_bytes_verified\":true,\"normal_cleanup\":true}";
        end_record();
        positive += is_positive ? 1 : 0;
        ++completed;
        trial.reset();
    }
    void boundary(const Case& test) {
        prepare(test);
        check(cuCtxSynchronize(), "drain before boundary observation");
        if (!test.has_current) check(cuCtxSetCurrent(nullptr), "clear boundary current context");
        CUcontext before{}, after{};
        check(cuCtxGetCurrent(&before), "observe boundary input context");
        expect(before == (test.has_current ? context : nullptr), "wrong boundary input context");
        trial->status = invoke(test, nullptr);
        original_return(test);
        check(cuCtxGetCurrent(&after), "observe boundary output context");
        expect(after == before, "generic Driver copy changed the current context");
        check(cuCtxSetCurrent(context), "restore context after boundary observation");
        check(cuCtxSynchronize(), "physical boundary completion");
        expect(verify_images(trial->status, test.is_zero), "full boundary byte oracle");
        std::cout << "{\"record\":\"uva_copy_boundary\",\"schema\":1,";
        descriptor(test);
        std::cout << ",\"pid\":" << ::getpid() << ",\"status\":" << static_cast<int>(trial->status)
                  << ",\"context_unchanged\":true";
        images();
        close_trial();
        std::cout << ",\"full_bytes_verified\":true,\"normal_cleanup\":true}";
        end_record();
        ++completed;
        trial.reset();
    }
    void close() {
        check(cuCtxSynchronize(), "final native physical completion");
        check(cuModuleUnload(module), "unload gate module");
        module = nullptr;
        check(cuCtxDestroy(context), "destroy native context");
        context = nullptr;
    }
};
}  // namespace

int main(int argc, char** argv) {
    std::cout << std::boolalpha;
    auto runner = std::make_unique<Runner>();
    try {
        std::string_view mode = "all";
        bool should_list{};
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--list")
                should_list = true;
            else if (argument == "--mode" && index + 1 < argc)
                mode = argv[++index];
            else if (argument == "--help") {
                std::cout << "Native UVA control: [--mode all|held|boundaries] [--list]\n"
                             "Execution requires authenticated READY followed by run newline.\n";
                return 0;
            } else
                throw std::runtime_error("unknown or incomplete argument");
        }
        expect(mode == "all" || mode == "held" || mode == "boundaries", "unknown mode");
        const auto cases = inventory(mode);
        if (should_list) {
            std::cout << "{\"record\":\"uva_copy_inventory\",\"schema\":1,\"mode\":" << quote(mode)
                      << ",\"cases\":[";
            for (std::size_t index = 0; index != cases.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << '{';
                descriptor(cases[index]);
                std::cout << '}';
            }
            std::cout << "]}";
            end_record();
            return 0;
        }
        expect(std::getenv("LD_PRELOAD") == nullptr && std::getenv("LD_AUDIT") == nullptr,
               "native probe cannot inherit loader instrumentation");
        runner->initialize();
        std::cout << "{\"record\":\"uva_copy_ready\",\"schema\":1,";
        runner->identity();
        std::cout << ",\"mode\":" << quote(mode) << ",\"case_count\":" << cases.size()
                  << ",\"step\":\"awaiting_run\"}";
        end_record();
        await_run();
        for (const auto& test : cases) {
            if (test.is_boundary)
                runner->boundary(test);
            else
                runner->held(test);
        }
        runner->close();
        std::cout << "{\"record\":\"uva_copy_result\",\"schema\":1,";
        runner->identity();
        std::cout << ",\"mode\":" << quote(mode) << ",\"case_count\":" << runner->completed
                  << ",\"positive_early_returns\":" << runner->positive
                  << ",\"normal_cleanup\":true,\"full_device_qualified\":false"
                     ",\"priority_qualified\":false,\"formal_comparison_qualified\":false}";
        end_record();
        return 0;
    } catch (const std::exception& error) {
        std::string cleanup_error;
        if (runner->trial) {
            try {
                runner->trial->finish_threads();
            } catch (const std::exception& cleanup) {
                cleanup_error = cleanup.what();
            }
        }
        std::cout << "{\"record\":\"uva_copy_failure\",\"schema\":1,\"pid\":" << ::getpid()
                  << ",\"error\":" << quote(error.what())
                  << ",\"cleanup_error\":" << quote(cleanup_error) << ",\"normal_cleanup\":false}"
                  << '\n'
                  << std::flush;
        // Uncertain native work retains all storage until authenticated process
        // exit. Never unwind a live thread, module, mapped word or copy buffer.
        std::_Exit(1);
    }
}
