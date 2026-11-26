#include "glimmer/control/execution_blas_protocol.h"
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cublas_v2.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_memory_native_calls();
extern "C" unsigned int glimmer_blas_native_calls();
extern "C" unsigned int glimmer_geam_native_calls();

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kHostileCases = 21;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned Driver request");
}
void check(cublasStatus_t value) {
    expect(value == CUBLAS_STATUS_SUCCESS, "owned BLAS request");
}
constexpr CUdeviceptr address(std::size_t index) {
    return 0x100000 + index * 0x1000;
}
// NOLINTBEGIN(performance-no-int-to-ptr): opaque fake GPU addresses, never host memory.
template <typename T>
T* pointer(CUdeviceptr value) {
    return reinterpret_cast<T*>(value);
}
// NOLINTEND(performance-no-int-to-ptr)
template <typename T>
T symbol(void* provider, const char* name) {
    auto* result = ::dlsym(provider, name);
    expect(result != nullptr, "qualified SDK export lookup");
    return reinterpret_cast<T>(result);
}
struct Libraries {
    void* driver{};
    void* blas{};
    void* geam{};
    decltype(&cublasSgeam) single{};
    decltype(&cublasDgeam) wide{};
    explicit Libraries(char** paths) {
        driver = ::dlopen(paths[1], RTLD_NOW | RTLD_LOCAL);
        blas = ::dlopen(paths[2], RTLD_NOW | RTLD_LOCAL);
        geam = ::dlopen(paths[3], RTLD_NOW | RTLD_LOCAL);
        expect(driver && blas && geam, "load independent caller-native trap providers");
        single = symbol<decltype(single)>(geam, "cublasSgeam");
        wide = symbol<decltype(wide)>(geam, "cublasDgeam");
        expect(single == &cublasSgeam && wide == &cublasDgeam &&
                   reinterpret_cast<void*>(single) == ::dlsym(RTLD_DEFAULT, "cublasSgeam") &&
                   reinterpret_cast<void*>(wide) == ::dlsym(RTLD_DEFAULT, "cublasDgeam"),
               "PLT, cached, global and qualified dlsym all select the owned GEAM ABI");
    }
    ~Libraries() {
        if (geam) static_cast<void>(::dlclose(geam));
        if (blas) static_cast<void>(::dlclose(blas));
        if (driver) static_cast<void>(::dlclose(driver));
    }
};
struct Case {
    bool is_double{}, is_device{}, unreadable_scalars{}, alpha_null{};
    std::uint32_t ta{}, tb{1};
    int rows{3}, columns{4}, lda{6}, ldb{7}, ldc{5};
    std::array<CUdeviceptr, 3> matrices{address(0) + 16, address(1) + 32, address(2) + 48};
    cublasStatus_t status{CUBLAS_STATUS_SUCCESS};
    bool zero_alpha{};
};
std::vector<Case> ordinary_cases() {
    std::vector<Case> result(10);
    result[0].matrices[0] = address(5) + 16;  // Distinct allocation held during the delayed reply.
    result[1].is_double = true;
    result[1].ta = 2;
    result[1].tb = 0;
    result[1].status = CUBLAS_STATUS_EXECUTION_FAILED;
    result[2].is_device = true;
    result[3].rows = 0;
    result[3].unreadable_scalars = true;
    result[3].matrices = {1, 2, 3};
    result[4].rows = -1;
    result[4].unreadable_scalars = true;
    result[4].status = CUBLAS_STATUS_INVALID_VALUE;
    result[5].alpha_null = true;
    result[6].zero_alpha = true;
    result[6].matrices[0] = 1;
    result[7].zero_alpha = true;
    result[7].matrices[0] = address(0) + 512;
    result[8].tb = 0;
    result[8].ldb = result[8].ldc = result[8].lda;
    result[8].matrices.fill(address(0) + 16);
    result[9].ta = 3;
    result[9].unreadable_scalars = true;
    result[9].status = CUBLAS_STATUS_INVALID_VALUE;
    return result;
}

// A separate ABI ledger; it neither decodes the actual request nor asks frontend helpers.
ExecutionGeam expected_request(const Case& test) {
    ExecutionGeam expected;
    expected.type = test.is_double ? ExecutionGeamType::kDouble : ExecutionGeamType::kFloat;
    expected.transpose_a = test.ta;
    expected.transpose_b = test.tb;
    expected.rows = test.rows;
    expected.columns = test.columns;
    expected.leading_a = test.lda;
    expected.leading_b = test.ldb;
    expected.leading_c = test.ldc;
    expected.handle = 70;
    expected.pointer_mode = test.is_device ? 1 : 0;
    expected.presence = test.alpha_null ? 30 : 31;
    for (std::size_t slot = 0; slot != 3; ++slot)
        for (std::size_t allocation = 0; allocation != 6; ++allocation)
            if (test.matrices[slot] >= address(allocation) &&
                test.matrices[slot] <= address(allocation) + 512)
                expected.buffers[slot] = {11 + allocation,
                                          test.matrices[slot] - address(allocation)};
    if (test.is_device) {
        expected.buffers[3] = {14, 4};
        expected.buffers[4] = {14, 16};
    } else if (!test.unreadable_scalars) {
        if (test.is_double) {
            const double alpha = 0.5, beta = -0.25;
            if (!test.alpha_null && !test.zero_alpha)
                std::memcpy(expected.alpha.data(), &alpha, sizeof(alpha));
            std::memcpy(expected.beta.data(), &beta, sizeof(beta));
        } else {
            const float alpha = 0.5F, beta = -0.25F;
            if (!test.alpha_null && !test.zero_alpha)
                std::memcpy(expected.alpha.data(), &alpha, sizeof(alpha));
            std::memcpy(expected.beta.data(), &beta, sizeof(beta));
        }
    }
    return expected;
}
void same_request(const ExecutionGeam& actual, const ExecutionGeam& expected) {
    expect(actual.type == expected.type && actual.transpose_a == expected.transpose_a &&
               actual.transpose_b == expected.transpose_b && actual.rows == expected.rows &&
               actual.columns == expected.columns && actual.leading_a == expected.leading_a &&
               actual.leading_b == expected.leading_b && actual.leading_c == expected.leading_c &&
               actual.handle == expected.handle && actual.pointer_mode == expected.pointer_mode &&
               actual.presence == expected.presence && actual.alpha == expected.alpha &&
               actual.beta == expected.beta,
           "complete independent GEAM scalar/shape/presence ABI ledger");
    for (std::size_t index = 0; index != 5; ++index)
        expect(actual.buffers[index].allocation == expected.buffers[index].allocation &&
                   actual.buffers[index].offset == expected.buffers[index].offset,
               "all five ordered symbolic ranges, aliases and unresolved pointers are exact");
}

struct Latches {
    std::atomic<bool> saw_command{}, release_command{};
    std::atomic<bool> setter_started{}, free_started{};
};
cublasStatus_t invoke(const Libraries& sdk, cublasHandle_t handle, const Case& test,
                      const void* protected_page) {
    alignas(double) std::array<std::byte, 8> alpha{}, beta{};
    if (test.is_double) {
        const double av = test.zero_alpha ? 0 : 0.5, bv = -0.25;
        std::memcpy(alpha.data(), &av, sizeof(av));
        std::memcpy(beta.data(), &bv, sizeof(bv));
    } else {
        const float av = test.zero_alpha ? 0 : 0.5F, bv = -0.25F;
        std::memcpy(alpha.data(), &av, sizeof(av));
        std::memcpy(beta.data(), &bv, sizeof(bv));
    }
    const void* a = test.is_device            ? pointer<void>(address(3) + 4)
                    : test.unreadable_scalars ? protected_page
                                              : alpha.data();
    const void* b = test.is_device            ? pointer<void>(address(3) + 16)
                    : test.unreadable_scalars ? protected_page
                                              : beta.data();
    if (test.alpha_null) a = nullptr;
    if (test.is_double)
        return sdk.wide(handle, static_cast<cublasOperation_t>(test.ta),
                        static_cast<cublasOperation_t>(test.tb), test.rows, test.columns,
                        static_cast<const double*>(a), pointer<double>(test.matrices[0]), test.lda,
                        static_cast<const double*>(b), pointer<double>(test.matrices[1]), test.ldb,
                        pointer<double>(test.matrices[2]), test.ldc);
    return sdk.single(handle, static_cast<cublasOperation_t>(test.ta),
                      static_cast<cublasOperation_t>(test.tb), test.rows, test.columns,
                      static_cast<const float*>(a), pointer<float>(test.matrices[0]), test.lda,
                      static_cast<const float*>(b), pointer<float>(test.matrices[1]), test.ldb,
                      pointer<float>(test.matrices[2]), test.ldc);
}

struct Hostile {
    ExecutionWireReply reply;
    cublasStatus_t expected{CUBLAS_STATUS_INTERNAL_ERROR};
    bool healthy{};
};
Hostile hostile(unsigned int which) {
    Hostile test;
    test.reply = {.disposition = ExecutionWireDisposition::kComplete,
                  .status = 0,
                  .original_status = 0,
                  .id = 1,
                  .arguments = {},
                  .data = {}};
    auto& reply = test.reply;
    switch (which) {
        case 0:
            reply.id = 0;
            break;
        case 1:
            reply.original_status.reset();
            break;
        case 2:
            reply.status = 13;
            break;
        case 3:
            reply.status = -1;
            reply.original_status = -1;
            break;
        case 4:
            reply.status = 32;
            reply.original_status = 32;
            break;
        case 5:
            reply.address = 1;
            break;
        case 6:
            reply.data.push_back(std::byte{1});
            break;
        case 7:
            reply.parameter_bytes = 2;
            break;
        case 8:
            reply.parameter_bytes = 1;
            reply.arguments.push_back({0, 1});
            break;
        case 9:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            break;
        case 10:
        case 11:
        case 12:
        case 13:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = which == 10 ? 0 : which == 11 ? -1 : which == 12 ? 1024 : 1;
            if (which != 13) reply.original_status.reset();
            break;
        case 14:
            reply.parameter_bytes = 1;
            test.expected = CUBLAS_STATUS_SUCCESS;
            break;
        case 15:
            reply.parameter_bytes = 1;
            reply.status = 13;
            reply.original_status = 13;
            test.expected = CUBLAS_STATUS_EXECUTION_FAILED;
            break;
        case 16:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.id = 0;
            reply.status = CUDA_ERROR_INVALID_VALUE;
            test.expected = CUBLAS_STATUS_INVALID_VALUE;
            test.healthy = true;
            break;
        case 17:
            reply.status = 13;
            reply.original_status = 13;
            test.expected = CUBLAS_STATUS_EXECUTION_FAILED;
            test.healthy = true;
            break;
        case 18:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            test.expected = CUBLAS_STATUS_NOT_SUPPORTED;
            test.healthy = true;
            break;
        case 19:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = 1;
            break;
        case 20:
            reply.status = 31;
            reply.original_status = 31;
            test.expected = static_cast<cublasStatus_t>(31);
            test.healthy = true;
            break;
        default:
            throw std::runtime_error("unknown isolated reply fixture");
    }
    return test;
}

void application(char** paths, std::optional<unsigned int> malformed, Latches& latches) {
    Libraries sdk(paths);
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cublasHandle_t handle{};
    check(cublasCreate(&handle));
    std::array<CUdeviceptr, 6> allocations{};
    for (std::size_t index = 0; index != allocations.size(); ++index) {
        check(cuMemAlloc(&allocations[index], 512));
        expect(allocations[index] == address(index), "independent symbolic address ledger");
    }
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    expect(page_size > 0, "guard page size");
    auto* page = ::mmap(nullptr, static_cast<std::size_t>(page_size), PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(page != MAP_FAILED, "unreadable scalar guard page");
    if (malformed) {
        const auto test = hostile(*malformed);
        const Case call;
        expect(invoke(sdk, handle, call, page) == test.expected,
               "strict GEAM receipt status versus original-status and uncertainty domains");
        expect(invoke(sdk, handle, call, page) ==
                   (test.healthy ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED),
               "canonical errors remain usable; hostile or uncertain receipts prevent any replay");
        if (test.healthy) {
            check(cublasDestroy(handle));
            for (const auto allocation : allocations) check(cuMemFree(allocation));
            check(cuCtxDestroy(context));
        }
    } else {
        const auto cases = ordinary_cases();
        auto pending = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(context));
            return invoke(sdk, handle, cases.front(), page);
        });
        const auto deadline = Clock::now() + std::chrono::seconds{3};
        while (!latches.saw_command && Clock::now() < deadline) std::this_thread::yield();
        expect(latches.saw_command, "exact GEAM request reached the fake service");
        auto setter = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(context));
            latches.setter_started = true;
            return cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        });
        auto freeing = std::async(std::launch::async, [&] {
            check(cuCtxSetCurrent(context));
            latches.free_started = true;
            return cuMemFree(allocations[5]);
        });
        while ((!latches.setter_started || !latches.free_started) && Clock::now() < deadline)
            std::this_thread::yield();
        const bool blocked =
            latches.setter_started && latches.free_started &&
            setter.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
            pending.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
        latches.release_command = true;  // Always release before assertions/joining.
        check(pending.get());
        check(setter.get());
        check(freeing.get());
        expect(blocked, "GEAM handle command lease excludes setters during its RPC");
        for (std::size_t index = 1; index != cases.size(); ++index) {
            check(cublasSetPointerMode(handle, cases[index].is_device ? CUBLAS_POINTER_MODE_DEVICE
                                                                      : CUBLAS_POINTER_MODE_HOST));
            expect(invoke(sdk, handle, cases[index], page) == cases[index].status,
                   "original typed GEAM status and exact scalar/range serialization");
        }
        Case invalid;
        invalid.unreadable_scalars = true;
        expect(invoke(sdk, handle, invalid, page) == CUBLAS_STATUS_INVALID_VALUE,
               "active unreadable HOST scalars fail locally without reading protected memory");
        invalid = {};
        invalid.matrices[0] = allocations[5];
        expect(invoke(sdk, handle, invalid, page) == CUBLAS_STATUS_INVALID_VALUE,
               "known retired device identity cannot become an unresolved native pointer");
        expect(invoke(sdk, nullptr, {}, page) == CUBLAS_STATUS_NOT_INITIALIZED,
               "invalid owned handle cannot fall through to caller-native GEAM");
        check(cublasDestroy(handle));
        for (std::size_t index = 0; index != 5; ++index) check(cuMemFree(allocations[index]));
        check(cuCtxDestroy(context));
    }
    expect(::munmap(page, static_cast<std::size_t>(page_size)) == 0, "release scalar guard page");
    expect(glimmer_geam_native_calls() == 0 && glimmer_blas_native_calls() == 0 &&
               glimmer_memory_native_calls() == 0,
           "neither caller-native GEAM nor caller Driver/BLAS consumes remote resources");
}

void respond(ExecutionSocketServer& server, const ExecutionReceivedRequest& request,
             const ExecutionWireReply& reply) {
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = request.connection;
    header.incarnation = {191, 193};
    auto payload = encode_execution_reply(reply);
    expect(payload.has_value(), "hostile fixture is structurally valid transport");
    const auto decoded = decode_execution_reply(*payload);
    expect(decoded.has_value(), "hostile capsule reaches route-specific receipt validation");
    expect(server.reply(request.connection, {header, std::move(*payload)}), "authenticated reply");
}
void run(char** paths, const std::string& socket, std::optional<unsigned int> malformed) {
    ExecutionSocketConfig config;
    config.socket_path = socket;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated CPU-only fake GEAM service");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "real peer process incarnation");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", socket.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "bind service identity before frontend first use");
    Latches latches;
    auto client = std::async(std::launch::async, [&] { application(paths, malformed, latches); });
    struct Guard {
        ExecutionSocketServer& server;
        Latches& latches;
        ~Guard() {
            latches.release_command = true;
            server.stop();
        }
    } guard{server, latches};
    std::optional<ExecutionReceivedRequest> pending;
    const auto cases = ordinary_cases();
    unsigned int hello{}, creates{}, allocations{}, calls{}, releases{}, destroys{}, closes{};
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline, "bounded fake-peer progress");
        if (pending && latches.release_command) {
            ExecutionPayloadReader held_input(pending->message.payload);
            ExecutionGeam held;
            expect(read_execution_geam(held_input, held) && held_input.empty(),
                   "held command remains an immutable complete request");
            same_request(held, expected_request(cases.front()));
            respond(server, *pending,
                    {.disposition = ExecutionWireDisposition::kComplete,
                     .status = 0,
                     .original_status = 0,
                     .id = 1,
                     .arguments = {},
                     .data = {}});
            pending.reset();
        }
        auto request = server.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply reply{.disposition = ExecutionWireDisposition::kComplete,
                                 .status = 0,
                                 .original_status = 0,
                                 .arguments = {},
                                 .data = {}};
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kHello) {
            expect(decode_execution_context_bootstrap(request->message.payload) ==
                       ExecutionContextBootstrap{ExecutionContextPurpose::kExplicitUser, 0, 1},
                   "ordinary typed context bootstrap, never metadata role");
            ++hello;
        } else if (operation == ExecutionOpcode::kCreateBlas) {
            expect(input.empty(), "create contains no caller-native handle");
            reply.id = 70;
            ++creates;
        } else if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(input.read_u64(bytes) && bytes == 512 && input.empty() && allocations < 6,
                   "exact six bounded fake allocations");
            reply.id = 11 + allocations;
            reply.address = address(allocations++);
        } else if (operation == ExecutionOpcode::kSetBlasAttribute) {
            std::uint64_t handle{}, value{};
            std::uint32_t kind{};
            expect(input.read_u64(handle) && handle == 70 && input.read_u32(kind) &&
                       input.read_u64(value) && input.empty() &&
                       ((kind == 1 && value <= 1) || (kind == 2 && value == CUBLAS_DEFAULT_MATH)),
                   "exact owned mode mutation");
            expect(!pending, "no mutating SDK request overtakes the held GEAM RPC");
        } else if (operation == ExecutionOpcode::kGeam) {
            ExecutionGeam actual;
            expect(request->message.payload.size() == 160 && read_execution_geam(input, actual) &&
                       input.empty(),
                   "exact fixed-width S/D GEAM request");
            const auto expected =
                malformed ? expected_request({}) : expected_request(cases.at(calls));
            same_request(actual, expected);
            reply.id = ++calls;
            if (malformed) {
                if (calls == 1) reply = hostile(*malformed).reply;
            } else {
                reply.status = cases.at(calls - 1).status;
                reply.original_status = reply.status;
                if (calls == 1) {
                    pending = std::move(*request);
                    latches.saw_command = true;
                    continue;
                }
            }
        } else if (operation == ExecutionOpcode::kRelease) {
            std::uint64_t id{};
            expect(input.read_u64(id) && id >= 11 && id <= 16 && input.empty(),
                   "release uses the unchanged authenticated allocation identity");
            // Local range pins are not a promise that generic free waits before
            // server admission. The backend owns fresh-ID validation/physical
            // leases; this IPC fixture proves only immutable submitted IDs.
            ++releases;
        } else if (operation == ExecutionOpcode::kDestroyBlas) {
            std::uint64_t id{};
            expect(input.read_u64(id) && id == 70 && input.empty(),
                   "exact BLAS destruction identity");
            ++destroys;
        } else if (operation == ExecutionOpcode::kClose) {
            expect(input.empty(), "close ordinary owner");
            ++closes;
        } else
            throw std::runtime_error("unexpected RPC, native fallback or reply replay");
        respond(server, *request, reply);
    }
    client.get();
    const bool healthy = !malformed || hostile(*malformed).healthy;
    expect(hello == 1 && creates == 1 && allocations == 6 &&
               calls == (malformed ? healthy ? 2U : 1U : 10U) && releases == (healthy ? 6U : 0U) &&
               destroys == (healthy ? 1U : 0U) && closes == (healthy ? 1U : 0U),
           "all admitted operations occur once; poisoned/local-invalid work produces no extra RPC");
    server.stop();
}
int one(char** paths, std::optional<unsigned int> malformed) {
    char pattern[] = "/tmp/glimmer-geam-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return 1;
    int status{};
    try {
        run(paths, std::string(directory) + "/service.sock", malformed);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    if (::rmdir(directory) != 0) status = 1;
    return status;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 && std::string_view(argv[4]) == "--receipt-case") {
        unsigned int index{};
        const std::string_view text(argv[5]);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), index);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
            index >= kHostileCases)
            return 1;
        return one(argv, index);
    }
    if (argc != 4 && argc != 5) return 1;
    if (argc == 4) return one(argv, std::nullopt);
    if (std::string_view(argv[4]) != "--malformed-replies") return 1;
    for (unsigned int index = 0; index != kHostileCases; ++index) {
        const auto child = ::fork();
        if (child < 0) return 1;
        if (!child) {
            auto number = std::to_string(index);
            std::array<char*, 7> arguments{
                argv[0],       argv[1], argv[2], argv[3], const_cast<char*>("--receipt-case"),
                number.data(), nullptr};
            ::execv(argv[0], arguments.data());
            ::_exit(127);
        }
        int status{};
        if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status)) {
            std::cerr << "isolated GEAM receipt case " << index << '\n';
            return 1;
        }
    }
    std::cout << "21 GEAM receipt cases; caller_native_calls=0; GPU_qualified=false\n";
    return 0;
}
