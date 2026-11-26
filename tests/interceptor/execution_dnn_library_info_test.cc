#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_library_info_fixture.h"

#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_library_info_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_library_info;
using Clock = std::chrono::steady_clock;
using SizeQuery = decltype(&cudnnGetVersion);
constexpr unsigned int kHostileCases = 36;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Cached {
    std::array<SizeQuery, 3> sizes{};
    decltype(&cudnnGetProperty) property{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open independent four-export caller provider");
    constexpr std::array names{"cudnnGetVersion", "cudnnGetCudartVersion",
                               "cudnnGetMaxDeviceVersion", "cudnnGetProperty"};
    const std::array<void*, 4> direct{reinterpret_cast<void*>(&cudnnGetVersion),
                                      reinterpret_cast<void*>(&cudnnGetCudartVersion),
                                      reinterpret_cast<void*>(&cudnnGetMaxDeviceVersion),
                                      reinterpret_cast<void*>(&cudnnGetProperty)};
    Cached result;
    for (std::size_t index = 0; index != names.size(); ++index) {
        auto* selected = ::dlsym(library, names[index]);
        expect(selected == direct[index] && ::dlsym(RTLD_DEFAULT, names[index]) == direct[index],
               "direct/global/explicit-provider lookup selects the controlled query");
        if (index != 3)
            result.sizes[index] = reinterpret_cast<SizeQuery>(selected);
        else
            result.property = reinterpret_cast<decltype(result.property)>(selected);
    }
    expect(::dlclose(library) == 0, "cached query pointers survive explicit provider close");
    return result;
}
struct GuardedInt {
    std::array<std::uint32_t, 3> before{0x12345678, 0x90abcdef, 0x98765432};
    int value{-777};
    std::array<std::uint32_t, 3> after{0xabcdef01, 0x98765432, 0x11223344};
    void verify(int expected) const {
        expect(value == expected &&
                   before == std::array<std::uint32_t, 3>{0x12345678, 0x90abcdef, 0x98765432} &&
                   after == std::array<std::uint32_t, 3>{0xabcdef01, 0x98765432, 0x11223344},
               "property publishes exact signed value without touching surrounding guards");
    }
};
void ordinary(const char* provider) {
    const auto cached = lookup(provider);
    expect(cudnnGetVersion() == 91002 && cudnnGetCudartVersion() == 12080 &&
               cudnnGetMaxDeviceVersion() == 1000,
           "direct queries return separate original values, not local SDK/Driver constants");
    for (std::size_t index = 0; index != cached.sizes.size(); ++index) {
        for (const auto value : {std::size_t{0}, (std::size_t{1} << 40) + 91,
                                 std::numeric_limits<std::size_t>::max()}) {
            fixture::control.values[index].store(value);
            expect(cached.sizes[index]() == value,
                   "direct-return query preserves every size_t bit");
            GuardedInt output;
            expect(cached.property(MAJOR_VERSION, &output.value) == CUDNN_STATUS_SUCCESS,
                   "native zero and SIZE_MAX do not poison subsequent metadata queries");
            output.verify(fixture::control.property_value.load());
        }
    }
    for (const int value : {INT32_MIN, -1, 0, INT32_MAX}) {
        fixture::control.property_value.store(value);
        for (const auto selector : {MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL}) {
            GuardedInt output;
            expect(cached.property(selector, &output.value) == CUDNN_STATUS_SUCCESS,
                   "all original property selectors complete");
            output.verify(value);
        }
    }
    GuardedInt failed;
    fixture::control.property_status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    expect(cudnnGetProperty(MINOR_VERSION, &failed.value) == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
           "native detailed error survives even when its temporary output was overwritten");
    failed.verify(-777);
    fixture::control.property_status.store(CUDNN_STATUS_SUCCESS);
    expect(cached.property(static_cast<libraryPropertyType>(3), &failed.value) ==
               CUDNN_STATUS_BAD_PARAM,
           "representable unnamed selector reaches original validation");
    failed.verify(-777);
    expect(cached.property(PATCH_LEVEL, nullptr) == CUDNN_STATUS_BAD_PARAM,
           "nullable original output is forwarded rather than replaced by temporary storage");
    const auto calls = fixture::calls();
    auto* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "create inaccessible property-output guard");
    const auto bad_output = cached.property(MAJOR_VERSION, static_cast<int*>(inaccessible));
    expect(::munmap(inaccessible, 4096) == 0, "release property-output guard");
    expect(bad_output == CUDNN_STATUS_BAD_PARAM && fixture::calls() == calls,
           "unwritable output rejects before original invocation");
    expect(cached.property(MAJOR_VERSION, &failed.value) == CUDNN_STATUS_SUCCESS,
           "ordinary native/pointer errors leave the owner usable");
    failed.verify(INT32_MAX);
    const auto child = ::fork();
    expect(child >= 0, "fork inherited library-query boundary");
    if (!child) {
        GuardedInt output;
        bool valid =
            cached.property(MAJOR_VERSION, &output.value) == CUDNN_STATUS_NOT_INITIALIZED &&
            output.value == -777;
        for (const auto query : cached.sizes) valid = valid && query() == 0;
        ::_exit(valid && !glimmer_dnn_library_info_native_calls() ? 0 : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               !glimmer_dnn_library_info_native_calls(),
           "inherited process state never contacts the service or caller SDK");
}
void missing_callback(const char* provider, unsigned int missing) {
    const auto cached = lookup(provider);
    GuardedInt output;
    for (unsigned int repetition = 0; repetition != 2; ++repetition) {
        if (missing == 4) {
            expect(cached.property(MAJOR_VERSION, &output.value) == CUDNN_STATUS_NOT_SUPPORTED,
                   "missing property callback maps clean Driver rejection to NOT_SUPPORTED");
            output.verify(-777);
            expect(cached.sizes[0]() == 91002, "missing callback does not poison direct queries");
        } else {
            expect(cached.sizes[missing - 1]() == 0,
                   "missing direct callback uses documented zero failure convention");
            expect(cached.property(MAJOR_VERSION, &output.value) == CUDNN_STATUS_SUCCESS,
                   "clean direct rejection leaves property query usable");
            output.verify(9);
        }
    }
    expect(!fixture::control.calls[missing - 1].load() && !glimmer_dnn_library_info_native_calls(),
           "missing SDK capability never falls through to caller-native execution");
}
void configure_endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated metadata-only service endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, unsigned int missing = 0) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual library-query service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-library-info-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    configure_endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch(missing));
    expect(service.start(), "start genuine metadata query service");
    client = std::async(std::launch::async, [=] {
        if (missing)
            missing_callback(provider, missing);
        else
            ordinary(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{15};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded metadata query progress");
    client.get();
    expect(service.request_stop() && service.finished(), "metadata-only service retires cleanly");
    expect(fixture::is_resource_free() && !glimmer_dnn_library_info_native_calls(),
           "library queries never initialize/query Driver, create descriptors or call caller SDK");
    if (!missing)
        expect(fixture::control.calls[0].load() == 4 && fixture::control.calls[1].load() == 4 &&
                   fixture::control.calls[2].load() == 4 &&
                   fixture::control.calls[3].load() == 25 &&
                   fixture::control.null_outputs.load() == 1,
               "each intended original query executes once with exact nullable output semantics");
}
struct HostileCase {
    ExecutionWireReply reply;
    bool is_property{}, has_output{true}, is_clean{}, loses_transport{};
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    std::size_t size{};
    int property{-777};
};
HostileCase hostile(unsigned int index) {
    HostileCase result;
    auto& reply = result.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    result.is_property = index >= 14 && index <= 29;
    if (result.is_property) reply.original_status = CUDNN_STATUS_SUCCESS;
    switch (index) {
        case 0:
            reply.original_status = CUDNN_STATUS_SUCCESS;
            break;
        case 1:
            reply.id = 7;
            break;
        case 2:
            reply.data = {std::byte{7}};
            break;
        case 3:
            reply.parameter_bytes = 1;
            break;
        case 4:
            reply.parameter_bytes = 2;
            break;
        case 5:
            reply.status = CUDA_ERROR_INVALID_VALUE;
            break;
        case 6:
            reply.disposition = ExecutionWireDisposition::kRejected;
            break;
        case 7:
        case 8:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = index == 7 ? -1 : 1024;
            break;
        case 9:
        case 10:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            if (index == 9)
                reply.address = 7;
            else
                reply.original_status = CUDNN_STATUS_SUCCESS;
            break;
        case 11:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.id = 7;
            break;
        case 12:
            result.loses_transport = true;
            break;
        case 13:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = 1;
            break;
        case 14:
            reply.original_status.reset();
            break;
        case 15:
            reply.original_status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 16:
        case 17:
            reply.status = index == 16 ? -1 : 8192;
            reply.original_status = reply.status;
            break;
        case 18:
            reply.address = std::uint64_t{1} << 32;
            break;
        case 19:
            result.has_output = false;
            reply.address = 1;
            break;
        case 20:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.address = 1;
            break;
        case 21:
            reply.parameter_bytes = 1;
            reply.address = 9;
            break;
        case 22:
            reply.parameter_bytes = 1;
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            result.status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 23:
            reply.id = 9;
            break;
        case 24:
            reply.data = {std::byte{9}};
            break;
        case 25:
            reply.parameter_bytes = 2;
            break;
        case 26:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_VALUE;
            break;
        case 27:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            break;
        case 28:
            result.has_output = false;
            result.is_clean = true;
            result.status = CUDNN_STATUS_SUCCESS;
            break;
        case 29:
            result.is_clean = true;
            reply.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
            reply.original_status = reply.status;
            result.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
            break;
        case 30:
            result.is_clean = true;
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            break;
        case 31:
            result.is_clean = true;
            break;
        case 32:
            result.is_clean = true;
            result.size = std::numeric_limits<std::size_t>::max();
            reply.address = result.size;
            break;
        case 33:
            result.is_property = result.is_clean = true;
            reply.original_status = CUDNN_STATUS_SUCCESS;
            reply.address = 0x80000000U;
            result.status = CUDNN_STATUS_SUCCESS;
            result.property = INT32_MIN;
            break;
        case 34:
            result.is_property = result.is_clean = true;
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            result.status = CUDNN_STATUS_NOT_SUPPORTED;
            break;
        case 35:
            reply.parameter_bytes = 1;
            reply.arguments = {{0, 1}};
            break;
        default:
            throw std::runtime_error("unknown bounded library-query envelope case");
    }
    return result;
}
void hostile_case(const char* provider, unsigned int index) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual hostile endpoint identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-library-hostile-" + std::to_string(::getpid()) + ".sock";
    configure_endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start real authenticated hostile transport");
    client = std::async(std::launch::async, [=] {
        const auto cached = lookup(provider);
        expect(cached.sizes[0]() == 1234, "establish one healthy metadata-only connection");
        GuardedInt output;
        if (test.is_property) {
            expect(cached.property(MINOR_VERSION, test.has_output ? &output.value : nullptr) ==
                       test.status,
                   "property validates native/policy envelope before any publication");
            output.verify(test.property);
        } else {
            expect(cached.sizes[0]() == test.size,
                   "direct native value or documented zero failure convention is exact");
        }
        if (test.is_clean) {
            expect(cached.property(MAJOR_VERSION, &output.value) == CUDNN_STATUS_SUCCESS,
                   "valid zero/native error/policy rejection keeps metadata usable");
            output.verify(91);
        } else {
            expect(cached.sizes[0]() == 0 &&
                       cached.property(MAJOR_VERSION, &output.value) == CUDNN_STATUS_INTERNAL_ERROR,
                   "quarantined descriptor-free context rejects locally without another RPC");
            output.verify(test.property);
        }
        expect(!glimmer_dnn_library_info_native_calls(), "bad transport never invokes caller SDK");
    });
    unsigned int hellos{}, queries{};
    bool disconnected{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline, "bounded hostile query progress");
        if (disconnected) {
            std::this_thread::yield();
            continue;
        }
        expect(server.poll_for(1), "poll hostile query transport");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "hostile server still authenticates real peer");
            if (profiles.respond(server, *request)) continue;
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            ExecutionPayloadReader input(request->message.payload);
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "four queries open DNM1 only, never a GPU session");
                ++hellos;
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "encode retained CPU owner");
                reply.data.assign(ownership->begin(), ownership->end());
            } else {
                ExecutionDnnLibraryInfo query;
                expect(request->message.header.operation == ExecutionOpcode::kDnnLibraryInfo &&
                           request->message.payload.size() == 12 &&
                           read_execution_dnn_library_info(input, query) && input.empty(),
                       "exact 12-byte query with no object or allocation identity");
                ++queries;
                if (queries == 1) {
                    expect(query.kind == ExecutionDnnLibraryInfoKind::kVersion && !query.property &&
                               !query.has_output,
                           "initial direct query is canonical");
                    reply.address = 1234;
                } else if (queries == 2) {
                    expect(
                        query.kind == (test.is_property ? ExecutionDnnLibraryInfoKind::kProperty
                                                        : ExecutionDnnLibraryInfoKind::kVersion) &&
                            query.property == (test.is_property ? 1U : 0U) &&
                            query.has_output == (test.is_property && test.has_output ? 1U : 0U),
                        "tested nullable query matches expected request exactly");
                    if (test.loses_transport) {
                        server.stop();
                        disconnected = true;
                        continue;
                    }
                    reply = test.reply;
                } else {
                    expect(test.is_clean && queries == 3 &&
                               query.kind == ExecutionDnnLibraryInfoKind::kProperty &&
                               !query.property && query.has_output == 1,
                           "only valid receipts permit one subsequent remote observation");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    reply.address = 91;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode deliberately hostile query envelope");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {71, 93};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver hostile envelope on actual authenticated connection");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && queries == (test.is_clean ? 3U : 2U),
           "quarantine never reopens or sends a hidden recovery query");
    server.stop();
}
void isolated_cases(const char* executable, const char* provider, bool malformed) {
    const unsigned int first = malformed ? 0 : 1, end = malformed ? kHostileCases : 5;
    for (unsigned int index = first; index != end; ++index) {
        const auto value = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork fresh query fixture process");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider,
                    malformed ? "--malformed-case" : "--missing-case", value.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("isolated library-query case failed: " + value);
    }
}
unsigned int index_argument(const char* text) {
    const std::string_view value{text};
    expect(!value.empty() && value.size() <= 2, "bounded numeric fixture index");
    unsigned int result{};
    for (const char digit : value) {
        expect(digit >= '0' && digit <= '9', "decimal fixture index");
        result = result * 10 + static_cast<unsigned int>(digit - '0');
    }
    return result;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional query test mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--missing-callback") {
            expect(argc == 3, "isolated mode has no extra arguments");
            isolated_cases(argv[0], argv[1], mode == "--malformed-replies");
        } else if (mode == "--malformed-case" || mode == "--missing-case") {
            expect(argc == 4, "isolated case requires one bounded index");
            const auto index = index_argument(argv[3]);
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known hostile query case");
                hostile_case(argv[1], index);
                ::_exit(0);  // No invented remote cleanup after deliberate transport corruption.
            }
            expect(index >= 1 && index <= 4, "known missing original callback");
            run(argv[1], ExecutionSchedulingMode::kNaive, index);
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known metadata scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Original library-info CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
