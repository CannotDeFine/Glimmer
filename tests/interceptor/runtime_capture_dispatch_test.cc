#include "internal/runtime_api_bridge.h"
#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/driver_cooperative_router.h"
#include "internal/symbol_registry.h"

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

extern "C" void glimmer_test_set_destroy_query_callbacks(
    glimmer::interceptor::GetProcAddressFunction, glimmer::interceptor::GetProcAddressV2Function);
extern "C" unsigned int glimmer_fake_capture_query_count(unsigned int);

namespace {
using namespace glimmer::interceptor;
const auto kActualStream = reinterpret_cast<CUstream>(0x2340);
const auto kActualGraph = reinterpret_cast<CUgraph>(0x3450);

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_scope() {
    auto& gate = cooperative_process_call_gate();
    RuntimeCaptureObservation observed;
    cudaError_t native_status{cudaErrorUnknown};
    unsigned int roots{};
    unsigned int leaves{};
    bool has_consumed{};
    const RuntimeCaptureDispatch dispatch = [&](CUcontext context, CUstream stream, bool is_ptds,
                                                CudaSourceCaptureKind kind, const auto& original) {
        require(is_inside_runtime_call() && !is_inside_driver_call() && context &&
                    stream == kActualStream && kind == CudaSourceCaptureKind::kEnd,
                "fake SDK callback supplies typed original handles, not Runtime input");
        (void)is_ptds;
        if (has_consumed) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        has_consumed = true;
        ++leaves;
        const NativeDriverCallScope native;
        return original();
    };
    const auto original = [&] {
        ++roots;
        return cudaSuccess;
    };
    require(invoke_runtime_capture_original(dispatch, original, observed, native_status) ==
                    CUDA_ERROR_INVALID_VALUE &&
                roots == 0 && !observed.was_runtime_invoked,
            "unadmitted permission cannot invoke fake Runtime");
    const ProcessCallGate::Call caller(gate);
    require(invoke_runtime_capture_original(dispatch, original, observed, native_status) ==
                    CUDA_ERROR_INVALID_VALUE &&
                roots == 0,
            "normal phase cannot borrow a capture root permission");
    const ProcessCallGate::Transition transition(gate);
    // CPU-only scope verification; this phase is not CUDA physical pause proof.
    for (const bool is_ptds : {false, true}) {
        has_consumed = false;
        CUgraph actual_output{};
        const auto result = invoke_runtime_capture_original(
            dispatch,
            [&] {
                ++roots;
                require(is_inside_runtime_call() && !is_inside_driver_call(),
                        "original Runtime callback must remain visible");
                RuntimeCaptureObservation nested;
                cudaError_t nested_status{cudaErrorApiFailureBase};
                require(
                    invoke_runtime_capture_original(dispatch, original, nested, nested_status) ==
                            CUDA_ERROR_INVALID_VALUE &&
                        !nested.was_runtime_invoked && nested_status == cudaErrorApiFailureBase,
                    "nested permission rejects without modifying outputs or native issue");
                const auto leaf = dispatch_runtime_capture_origin(
                    kActualStream, is_ptds, CudaSourceCaptureKind::kEnd,
                    [&] {
                        require(is_inside_driver_call(), "typed owner guards only native original");
                        actual_output = kActualGraph;
                        return CUDA_SUCCESS;
                    },
                    &actual_output);
                require(leaf && *leaf == CUDA_SUCCESS,
                        "one typed end observes successful actual graph output");
                const auto before = observed;
                require(
                    invoke_runtime_capture_original(dispatch, original, observed, native_status) ==
                            CUDA_ERROR_INVALID_VALUE &&
                        observed.original_driver_calls == before.original_driver_calls &&
                        observed.graph == before.graph && observed.context == before.context,
                    "nested output alias cannot mutate accepted outer observation");
                return cudaSuccess;
            },
            observed, native_status);
        require(result == CUDA_SUCCESS && native_status == cudaSuccess &&
                    observed.was_runtime_invoked && observed.context &&
                    observed.stream == kActualStream && observed.graph == kActualGraph &&
                    observed.is_ptds == is_ptds && observed.original_driver_calls == 1 &&
                    !is_inside_runtime_call() && !is_inside_driver_call(),
                "typed observation, exact native status and balanced legacy/PTDS scope");
        require(
            !dispatch_runtime_capture_origin(kActualStream, is_ptds, CudaSourceCaptureKind::kEnd,
                                             [] { return CUDA_SUCCESS; }),
            "capture permission cannot escape original Runtime invocation");
    }
    require(roots == 2 && leaves == 2, "exact original counts exclude every rejected call");
    const auto runtime_error = invoke_runtime_capture_original(
        dispatch, [] { return cudaErrorApiFailureBase; }, observed, native_status);
    require(runtime_error == CUDA_ERROR_UNKNOWN && native_status == cudaErrorApiFailureBase &&
                observed.was_runtime_invoked && observed.original_driver_calls == 0 &&
                !is_inside_runtime_call(),
            "Runtime error outside CUresult range is retained without enum conversion");
    const auto thrown = invoke_runtime_capture_original(
        dispatch,
        []() -> cudaError_t { throw std::runtime_error("original Runtime callback failure"); },
        observed, native_status);
    require(thrown == CUDA_ERROR_UNKNOWN && observed.was_runtime_invoked &&
                !is_inside_runtime_call() && !is_inside_driver_call(),
            "original exception restores lexical permission without retry");
}

void check_runtime_graph_contract() {
    unsigned int originals{};
    const auto executable = reinterpret_cast<cudaGraphExec_t>(0x3450);
    const auto unused = [&] {
        ++originals;
        return cudaSuccess;
    };
    require(
        execute_owned_runtime_graph(executable, nullptr, false, unused) == cudaErrorNotSupported &&
            originals == 0,
        "unadmitted caller cannot acquire private graph execution permission");
    {
        const ProcessCallGate::Call call(cooperative_process_call_gate());
        const ProcessCallGate::Transition transition(cooperative_process_call_gate());
        require(execute_owned_runtime_graph(executable, nullptr, false, unused) ==
                        cudaErrorNotSupported &&
                    originals == 0,
                "capture/source transition receives no normal graph replay permission");
        require(
            execute_owned_runtime_graph(nullptr, nullptr, false, unused) == cudaErrorInvalidValue &&
                originals == 0,
            "missing executable rejects before graph original");
    }
    const auto original = [&] {
        ++originals;
        require(is_inside_runtime_call() && is_inside_driver_call(),
                "SDK graph issue has balanced Runtime and owned Driver scopes");
        return cudaSuccess;
    };
    const RuntimeGraphExecution issue = [](const auto& native) {
        const NativeDriverCallScope scope;
        return native();
    };
    require(invoke_runtime_graph_original({}, original) == cudaErrorNotSupported &&
                invoke_runtime_graph_original(issue, {}) == cudaErrorNotSupported && originals == 0,
            "missing graph ownership or original rejects before issue");
    require(invoke_runtime_graph_original([](const auto&) { return CUDA_ERROR_NOT_READY; },
                                          original) == cudaErrorUnknown &&
                originals == 0,
            "pre-issue ownership error invokes no SDK original");
    require(invoke_runtime_graph_original([](const auto&) { return CUDA_SUCCESS; }, original) ==
                    cudaErrorUnknown &&
                originals == 0,
            "empty successful execution is not original issue or physical proof");
    require(invoke_runtime_graph_original([](const auto& native) { return native(); }, original) ==
                    cudaErrorUnknown &&
                originals == 0,
            "unowned native scope cannot issue the original");
    require(invoke_runtime_graph_original(issue, original) == cudaSuccess && originals == 1 &&
                !is_inside_driver_call() && !is_inside_runtime_call(),
            "one owned SDK issue restores original caller TLS");
    require(invoke_runtime_graph_original(
                [](const auto& native) {
                    const NativeDriverCallScope scope;
                    require(native() == CUDA_SUCCESS, "first owned SDK issue succeeds");
                    require(native() == CUDA_ERROR_INVALID_VALUE,
                            "duplicate dispatch rejects without repeating SDK issue");
                    return CUDA_SUCCESS;
                },
                original) == cudaErrorUnknown &&
                originals == 2,
            "swallowed duplicate dispatch cannot manufacture successful ownership");
    require(invoke_runtime_graph_original(
                [](const auto& native) {
                    CUresult foreign_status{CUDA_ERROR_UNKNOWN};
                    std::thread foreign([&] {
                        const NativeDriverCallScope scope;
                        foreign_status = native();
                    });
                    foreign.join();
                    require(foreign_status == CUDA_ERROR_INVALID_VALUE,
                            "borrowed graph callback cannot move to another caller");
                    return CUDA_SUCCESS;
                },
                original) == cudaErrorUnknown &&
                originals == 2,
            "foreign caller never invokes SDK or changes the owner caller TLS");
    for (const auto error : {cudaErrorInvalidValue, cudaErrorUnknown, cudaErrorApiFailureBase}) {
        for (const bool throws_after_issue : {false, true}) {
            require(invoke_runtime_graph_original(
                        [&](const auto& native) -> CUresult {
                            const NativeDriverCallScope scope;
                            require(native() == CUDA_ERROR_UNKNOWN,
                                    "Runtime error supplies only an uncertainty signal");
                            if (throws_after_issue) {
                                throw std::runtime_error("post-issue ownership failure");
                            }
                            return CUDA_SUCCESS;
                        },
                        [&] {
                            ++originals;
                            return error;
                        }) == error,
                    "original Runtime enum remains exact through owner error or exception");
        }
    }
    require(invoke_runtime_graph_original(
                [](const auto& native) {
                    const NativeDriverCallScope scope;
                    require(native() == CUDA_SUCCESS, "successful SDK precedes owner failure");
                    return CUDA_ERROR_LAUNCH_FAILED;
                },
                original) == cudaErrorUnknown,
            "successful SDK issue does not hide failed completion ownership");
    require(invoke_runtime_graph_original(issue,
                                          []() -> cudaError_t {
                                              throw std::runtime_error("original SDK exception");
                                          }) == cudaErrorUnknown &&
                !is_inside_driver_call() && !is_inside_runtime_call(),
            "original exception restores TLS and cannot authorize replay");
}

void check_borrowed_context_completion() {
    auto& gate = cooperative_process_call_gate();
    unsigned int originals{}, quarantines{};
    const auto original = [&] {
        require(is_inside_runtime_call() && is_inside_driver_call() &&
                    gate.is_native_only_on_current_thread(),
                "borrowed completion owns real Runtime scope and excluded native phase");
        ++originals;
        return cudaSuccess;
    };
    const auto quarantine = [&] { ++quarantines; };
    const RuntimeContextCompletionExecution accepted = [&](const auto& borrowed) {
        const ProcessCallGate::Transition transition(gate);
        const NativeDriverCallScope native;
        return std::optional{borrowed()};
    };
    require(invoke_runtime_context_completion_original(accepted, original, quarantine) ==
                    cudaErrorUnknown &&
                originals == 0 && quarantines == 0,
            "unentered completion root cannot lend original effects");
    const ProcessCallGate::Call call(gate);
    require(!invoke_runtime_context_completion_original(
                [](const auto&) -> std::optional<CUresult> { return std::nullopt; }, original,
                quarantine) &&
                originals == 0 && quarantines == 0,
            "disabled ownership may decline before an SDK effect");
    require(
        invoke_runtime_context_completion_original(accepted, original, quarantine) == cudaSuccess &&
            originals == 1 && quarantines == 0 && !is_inside_driver_call() &&
            !is_inside_runtime_call() && !gate.is_native_only_on_current_thread(),
        "one accepted SDK original preserves lexical scope and healthy process phase");
    for (unsigned int mode = 0; mode < 7; ++mode) {
        const auto before = originals;
        const auto rejected = invoke_runtime_context_completion_original(
            [&](const auto& borrowed) -> std::optional<CUresult> {
                if (mode == 0) return CUDA_SUCCESS;  // missing original
                if (mode == 1) return borrowed();    // missing native/transition scopes
                const ProcessCallGate::Transition transition(gate);
                if (mode == 2) return borrowed();  // missing native scope
                const NativeDriverCallScope native;
                if (mode == 3) {
                    CUresult result{CUDA_SUCCESS};
                    std::thread foreign([&] {
                        const NativeDriverCallScope foreign_native;
                        result = borrowed();
                    });
                    foreign.join();
                    require(result == CUDA_ERROR_INVALID_VALUE,
                            "foreign completion caller rejects before actual original");
                    return CUDA_SUCCESS;
                }
                require(borrowed() == CUDA_SUCCESS, "one observed successful completion");
                if (mode == 4) {
                    require(borrowed() == CUDA_ERROR_INVALID_VALUE,
                            "duplicate borrowed original rejects before a second SDK entry");
                    return CUDA_SUCCESS;
                }
                if (mode == 5) return std::nullopt;  // decline after original effect
                throw std::runtime_error("post-completion ownership exception");
            },
            original, quarantine);
        require(rejected == cudaErrorUnknown && originals == before + (mode >= 4 ? 1U : 0U) &&
                    quarantines == mode + 1 && !is_inside_driver_call() &&
                    !is_inside_runtime_call() && !gate.is_native_only_on_current_thread(),
                "uncertain completion ownership quarantines without replay or fallback");
    }
    for (const auto status :
         {cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorApiFailureBase}) {
        const auto before = originals;
        const auto failures = quarantines;
        require(invoke_runtime_context_completion_original(
                    accepted,
                    [&] {
                        ++originals;
                        return status;
                    },
                    quarantine) == status &&
                    originals == before + 1 && quarantines == failures,
                "actual non-success SDK enum is preserved without numeric CUresult conversion");
        require(invoke_runtime_context_completion_original(
                    [&](const auto& borrowed) -> std::optional<CUresult> {
                        const ProcessCallGate::Transition transition(gate);
                        const NativeDriverCallScope native;
                        require(borrowed() == CUDA_ERROR_UNKNOWN,
                                "SDK error supplies only a failed proof signal");
                        return CUDA_SUCCESS;  // swallowed original failure
                    },
                    [status] { return status; }, quarantine) == status,
                "swallowed failed completion remains the authoritative Runtime error");
    }
    require(quarantines == 10, "each swallowed failed effect publishes one CPU quarantine");
    require(
        invoke_runtime_context_completion_original(
            accepted, []() -> cudaError_t { throw std::runtime_error("SDK completion exception"); },
            quarantine) == cudaErrorUnknown &&
            quarantines == 11,
        "exceptional original return never supplies completion proof");
    require(invoke_runtime_context_completion_original(
                [&](const auto& borrowed) -> std::optional<CUresult> {
                    const NativeDriverCallScope native;  // missing process exclusion
                    return borrowed();
                },
                original, quarantine) == cudaErrorUnknown &&
                quarantines == 12,
            "native scope alone cannot authorize an unexcluded completion proof");
}

void check_epoch_environment(std::string_view mode) {
    constexpr std::array names{
        "GLIMMER_NATIVE_EPOCH_MAX_CALLS", "GLIMMER_NATIVE_EPOCH_CALLS_PER_PREFIX",
        "GLIMMER_NATIVE_EPOCH_PENDING_PREFIXES", "GLIMMER_NATIVE_EPOCH_ACCEPTANCE_US"};
    constexpr std::array values{"8", "2", "2", "2000"};
    for (std::size_t index = 0; index < names.size(); ++index)
        require(setenv(names[index], values[index], 1) == 0, "set isolated epoch extent");
    const auto set = [](const char* name, const char* value) {
        require(setenv(name, value, 1) == 0, "set isolated startup option");
    };
    bool is_valid = mode == "valid" || mode == "default";
    if (mode == "default") {
        for (const auto* name : names) require(unsetenv(name) == 0, "clear public epoch selection");
    } else if (mode == "partial") {
        require(unsetenv(names[2]) == 0, "omit required pending-prefix extent");
    } else if (mode == "empty")
        set(names[0], "");
    else if (mode == "negative")
        set(names[0], "-1");
    else if (mode == "plus")
        set(names[0], "+8");
    else if (mode == "whitespace")
        set(names[0], " 8");
    else if (mode == "trailing")
        set(names[0], "8x");
    else if (mode == "overflow")
        set(names[0], "18446744073709551616");
    else if (mode == "zero")
        set(names[0], "0");
    else if (mode == "call-bound")
        set(names[0], "65537");
    else if (mode == "prefix-width")
        set(names[1], "9");
    else if (mode == "prefix-bound")
        set(names[2], "17");
    else if (mode == "time-bound")
        set(names[3], "1000001");
    else if (mode == "runtime-disabled")
        set("GLIMMER_COOPERATIVE_RUNTIME_PREVIEW", "0");
    else if (mode == "mode-disabled")
        set("GLIMMER_SCHEDULER_MODE", "off");
    else if (mode == "batch")
        set("GLIMMER_SCHEDULER_BATCH_SIZE", "2");
    else if (mode == "quota-missing")
        require(unsetenv("GLIMMER_MEMORY_LIMIT_BYTES") == 0, "omit quota");
    else if (mode == "quota-invalid")
        set("GLIMMER_MEMORY_LIMIT_BYTES", "bad");
    else if (mode == "quota-mode")
        set("GLIMMER_QUOTA_MODE", "invalid");
    else
        require(mode == "valid" || mode == "private-conflict", "known epoch environment mode");
    const ProcessCallGate::Call call(cooperative_process_call_gate());
    if (mode == "private-conflict")
        require(configure_cooperative_native_epochs({{8, 2, 2, 2000}, {}}) == CUDA_SUCCESS,
                "prepare trusted policy before deliberate selection collision");
    auto* router = cooperative_source_router();
    require(is_valid ? router != nullptr : router == nullptr,
            "invalid public extent rejects instead of creating an ordinary fallback router");
    require(glimmer_fake_capture_query_count(0) == 0 && glimmer_fake_capture_query_count(1) == 0,
            "startup selection invokes no original capture SDK work");
    if (router) {
        require(router->has_native_epoch_policy() == (mode == "valid"),
                "explicit versus default policy");
        set(names[0], "bad");
        require(cooperative_source_router() == router &&
                    router->has_native_epoch_policy() == (mode == "valid"),
                "later environment changes cannot mutate process policy");
        require(
            configure_cooperative_native_epochs({{8, 2, 2, 2000}, {}}) == CUDA_ERROR_NOT_PERMITTED,
            "public/default initialized policy rejects private late mutation");
    }
    std::cout << "epoch_environment mode=" << mode << " valid=" << is_valid << " sdk_effects=0\n";
}

unsigned int epoch_projection_calls{};
CUresult rejected_epoch_projection(CUevent) {
    ++epoch_projection_calls;
    return CUDA_ERROR_UNKNOWN;
}

void check_process_epoch_startup(bool is_late = false) {
    auto& gate = cooperative_process_call_gate();
    const CudaNativeEpochConfiguration configuration{{8, 2, 2, 1'000'000}, {}};
    require(configure_cooperative_native_epochs(configuration) == CUDA_ERROR_NOT_PERMITTED,
            "unentered private startup cannot configure the process owner");
    const ProcessCallGate::Call call(gate);
    if (is_late) {
        auto* router = cooperative_source_router();
        require(router && !router->has_native_epoch_policy() &&
                    configure_cooperative_native_epochs(configuration) == CUDA_ERROR_NOT_PERMITTED,
                "ordinary process startup remains disabled and rejects later policy mutation");
        return;
    }
    auto invalid = configuration;
    invalid.extent.calls_per_prefix = 0;
    require(configure_cooperative_native_epochs(invalid) == CUDA_ERROR_INVALID_VALUE,
            "invalid extent does not consume one-shot process startup");
    {
        const NativeDriverCallScope native;
        require(configure_cooperative_native_epochs(configuration) == CUDA_ERROR_NOT_PERMITTED,
                "native callback cannot configure process policy");
    }
    {
        const ProcessCallGate::Transition transition(gate);
        require(configure_cooperative_native_epochs(configuration) == CUDA_ERROR_NOT_PERMITTED,
                "nested transition rejects startup before upgrade or effects");
    }
    require(invoke_runtime_graph_original(
                [&](const auto& original) {
                    const NativeDriverCallScope native;
                    return original();
                },
                [&] {
                    require(is_inside_runtime_call() &&
                                configure_cooperative_native_epochs(configuration) ==
                                    CUDA_ERROR_NOT_PERMITTED,
                            "actual Runtime callback cannot configure process startup");
                    return cudaSuccess;
                }) == cudaSuccess,
            "startup rejection leaves original Runtime callback scopes and result intact");
    require(configure_cooperative_native_epochs(configuration) == CUDA_SUCCESS &&
                configure_cooperative_native_epochs(configuration) == CUDA_ERROR_NOT_PERMITTED,
            "normal private startup is one-shot before SDK process initialization");
    // Fork before any process-owner worker starts; this is synthetic metadata
    // qualification, not CUDA-after-fork support or leak-free child teardown.
    const auto child = fork();
    require(child >= 0, "create isolated startup metadata child");
    if (child == 0) {
        auto* fresh = cooperative_source_router();
        _exit(fresh && !fresh->has_native_epoch_policy() ? 0 : 1);
    }
    int child_status{};
    require(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                WEXITSTATUS(child_status) == 0,
            "forked fresh process metadata cannot inherit private epoch startup policy");
    auto* router = cooperative_source_router();
    require(router && router->has_native_epoch_policy() && cooperative_source_router() == router &&
                configure_cooperative_native_epochs(configuration) == CUDA_ERROR_NOT_PERMITTED,
            "actual process owner has immutable configured policy and identity after startup");
    DriverDispatch driver;
    require(driver.initialize(), "isolated fake SDK native dispatch");
    bool has_rejected_projection{};
    try {
        const DriverCooperativeRouter disabled(driver, {}, false, false, std::nullopt,
                                               {rejected_epoch_projection, nullptr, nullptr});
    } catch (const std::invalid_argument&) {
        has_rejected_projection = true;
    }
    require(has_rejected_projection && epoch_projection_calls == 0,
            "disabled policy rejects physical projection overrides without SDK effects");
    for (unsigned int alias = 0; alias < 2; ++alias) {
        void* pointer{};
        const auto flags = alias ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                 : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
        require(driver.get_proc_address("cuStreamIsCapturing", &pointer, 12600, flags) ==
                        CUDA_SUCCESS &&
                    pointer && !is_interceptor_driver_function(pointer),
                "native fake SDK capture-query aliases are not public wrappers");
        const auto count = glimmer_fake_capture_query_count(alias);
        CUstreamCaptureStatus status{};
        const NativeDriverCallScope native;
        require(reinterpret_cast<StreamCaptureFunction>(pointer)(kActualStream, &status) ==
                        CUDA_SUCCESS &&
                    status == CU_STREAM_CAPTURE_STATUS_NONE &&
                    glimmer_fake_capture_query_count(alias) == count + 1,
                "SDK legacy/PTDS proc-address selection uses its exact original alias");
    }
}

void check_private_sdk_reset_cache_binding() {
    DriverDispatch driver;
    require(driver.initialize(), "resolve native fake Driver without process-owner initialization");
    using Query = std::function<CUresult(const char*, void**, int, cuuint64_t)>;
    const std::array<Query, 2> sdk_queries{
        runtime_driver_get_proc_address,
        [](const char* name, void** address, int version, cuuint64_t flags) {
            return runtime_driver_get_proc_address_v2(name, address, version, flags, nullptr);
        }};
    for (const std::string family : {"cuDevicePrimaryCtxReset", "cuDevicePrimaryCtxRelease"}) {
        std::array<void*, 2> cached{};
        for (const auto& query : sdk_queries) {
            for (unsigned int index = 0; index < 2; ++index) {
                const int version = index == 0 ? 7000 : 11000;
                const auto name = index == 0 ? family : family + "_v2";
                void* native_address{};
                void* sdk_address{};
                void* private_address{};
                {
                    // The first lookup is trusted SDK cold initialization, before
                    // any ordinary public root configures the cooperative owner.
                    const NativeDriverCallScope native;
                    require(query(family.c_str(), &sdk_address, version, 0) == CUDA_SUCCESS &&
                                sdk_address == find_interceptor_symbol(name.c_str()) &&
                                is_inside_driver_call() && !is_inside_runtime_call(),
                            "private SDK cache keeps exact controlled ownership version and scope");
                    require(
                        driver.get_proc_address(family.c_str(), &native_address, version, 0) ==
                                CUDA_SUCCESS &&
                            native_address && native_address != sdk_address &&
                            intercept_get_proc_address(family.c_str(), &private_address, version,
                                                       0) == CUDA_SUCCESS &&
                            private_address == native_address,
                        "native table and Driver-only private resolver retain original pointers");
                    using Reset = CUresult(CUDAAPI*)(CUdevice);
                    const auto expected = index == 1 ? CUDA_ERROR_INVALID_CONTEXT
                                          : family == "cuDevicePrimaryCtxReset"
                                              ? CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
                                              : CUDA_ERROR_INVALID_DEVICE;
                    require(reinterpret_cast<Reset>(sdk_address)(0) == expected,
                            "controlled private invocation delegates the exact native version");
                }
                require(!is_inside_driver_call() && !is_inside_runtime_call(),
                        "private SDK lookup cannot leak either lexical guard");
                cached[index] = sdk_address;
            }
            for (const auto version : {6999, 10999}) {
                const auto name = version == 6999 ? family : family + "_v2";
                void* address = cached[0];
                const NativeDriverCallScope native;
                require(
                    query(name.c_str(), &address, version, 0) == CUDA_ERROR_NOT_FOUND && !address,
                    "SDK lifecycle binding cannot substitute an unavailable historical ownership "
                    "ABI");
            }
        }
        require(cached[0] && cached[1] && cached[0] != cached[1],
                "legacy and modern cached lifecycle bindings are never aliases");
    }
}

const DriverDispatch* g_destroy_query_driver{};
CUresult CUDAAPI reentrant_destroy_query(const char* name, void** address, int version,
                                         cuuint64_t flags) {
    require(g_destroy_query_driver && is_inside_driver_call(),
            "reentrant original destroy query retains its Driver guard");
    return g_destroy_query_driver->get_proc_address(name, address, version, flags);
}
CUresult CUDAAPI reentrant_destroy_query_v2(const char* name, void** address, int version,
                                            cuuint64_t flags,
                                            CUdriverProcAddressQueryResult* query) {
    require(g_destroy_query_driver && is_inside_driver_call(),
            "reentrant extended original query retains its Driver guard");
    return g_destroy_query_driver->get_proc_address_v2(name, address, version, flags, query);
}

class DestroyQueryFixture {
   public:
    DestroyQueryFixture(const DriverDispatch& driver, bool is_reentrant) {
        if (is_reentrant) {
            g_destroy_query_driver = &driver;
            glimmer_test_set_destroy_query_callbacks(reentrant_destroy_query,
                                                     reentrant_destroy_query_v2);
        }
    }
    ~DestroyQueryFixture() {
        glimmer_test_set_destroy_query_callbacks(nullptr, nullptr);
        g_destroy_query_driver = nullptr;
    }
    DestroyQueryFixture(const DestroyQueryFixture&) = delete;
    DestroyQueryFixture& operator=(const DestroyQueryFixture&) = delete;
};

void check_private_sdk_destroy_cache_binding(bool is_reentrant) {
    DriverDispatch driver;
    require(driver.initialize(), "resolve native destroy table without process-owner activation");
    const DestroyQueryFixture fixture(driver, is_reentrant);
    using Destroy = CUresult(CUDAAPI*)(CUcontext);
    for (const bool is_extended : {false, true}) {
        std::array<void*, 2> cached{};
        for (const int version : {2000, 3020, 3999, 4000, 12600}) {
            const bool is_modern = version >= 4000;
            void* sdk_address{};
            void* native_address{};
            void* private_address{};
            CUdriverProcAddressQueryResult query{CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND};
            {
                const NativeDriverCallScope native;
                const auto status =
                    is_extended
                        ? runtime_driver_get_proc_address_v2("cuCtxDestroy", &sdk_address, version,
                                                             0, &query)
                        : runtime_driver_get_proc_address("cuCtxDestroy", &sdk_address, version, 0);
                require(status == CUDA_SUCCESS &&
                            (!is_extended || query == CU_GET_PROC_ADDRESS_SUCCESS) &&
                            sdk_address == find_interceptor_symbol(is_modern ? "cuCtxDestroy_v2"
                                                                             : "cuCtxDestroy"),
                        "SDK cache retains exact destroy ownership version before owner import");
                require(driver.get_proc_address("cuCtxDestroy", &native_address, version, 0) ==
                                CUDA_SUCCESS &&
                            native_address && native_address != sdk_address &&
                            intercept_get_proc_address("cuCtxDestroy", &private_address, version,
                                                       0) == CUDA_SUCCESS &&
                            private_address == native_address,
                        "Driver-only lookup retains the exact independent native destroy entry");
                require(reinterpret_cast<Destroy>(sdk_address)(nullptr) ==
                            (is_modern ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_INVALID_HANDLE),
                        "private cached invocation preserves exact destroy native error");
                require(is_inside_driver_call() && !is_inside_runtime_call(),
                        "private destroy preserves the caller lexical guard");
            }
            require(!is_inside_driver_call() && !is_inside_runtime_call(),
                    "SDK destroy cache cannot leak either lexical guard");
            cached[is_modern ? 1 : 0] = sdk_address;
        }
        require(cached[0] && cached[1] && cached[0] != cached[1],
                "historical and modern destroy caches are not ownership aliases");
        for (const auto* name : {"cuCtxDestroy", "cuCtxDestroy_v2"}) {
            for (const int version : {1999, 2000, 3999, 4000, 12600}) {
                if (std::string_view(name) == "cuCtxDestroy" && version >= 2000) {
                    continue;
                }
                void* address = cached[0];
                CUdriverProcAddressQueryResult query{CU_GET_PROC_ADDRESS_SUCCESS};
                const NativeDriverCallScope native;
                const auto status =
                    is_extended
                        ? runtime_driver_get_proc_address_v2(name, &address, version, 0, &query)
                        : runtime_driver_get_proc_address(name, &address, version, 0);
                const auto expected = std::string_view(name) == "cuCtxDestroy"
                                          ? CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT
                                          : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
                require(is_extended ? status == CUDA_SUCCESS && !address && query == expected
                                    : status == CUDA_ERROR_NOT_FOUND && address == cached[0],
                        "unavailable destroy queries preserve each native resolver output ABI");
            }
        }
    }
}

void check_cold_sdk_capacity() {
    void* address{};
    require(
        runtime_driver_get_proc_address("cuDeviceTotalMem", &address, 3020, 0) == CUDA_SUCCESS &&
            address,
        "cold SDK retains controlled device-total cache binding");
    const auto resolver = resolve_real_dlsym();
    require(resolver, "resolve original loader for the cold capacity observer");
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    require(library, "borrow already-loaded fake Driver for its exact capacity observer");
    using Count = unsigned int (*)();
    const auto count =
        reinterpret_cast<Count>(resolver(library, "glimmer_test_context_device_queries"));
    require(dlclose(library) == 0, "balance temporary fake Driver observer reference");
    require(count && count() == 0, "cold capacity starts without an invented device query");
    using Total = CUresult(CUDAAPI*)(std::size_t*, CUdevice);
    std::size_t total{};
    require(reinterpret_cast<Total>(address)(&total, 0) == CUDA_SUCCESS && total == 8388608 &&
                count() == 0 && !is_inside_driver_call() && !is_inside_runtime_call(),
            "unbound SDK total capacity preserves quota without invalid context-device query");
}

void check_stream_context_abi() {
#if CUDA_VERSION >= 12050
    using Legacy = CUresult(CUDAAPI*)(CUstream, CUcontext*);
    using Modern = CUresult(CUDAAPI*)(CUstream, CUcontext*, CUgreenCtx*);
    using LegacyQuery = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
    using ModernQuery =
        CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
    const auto resolve = resolve_real_dlsym();
    require(resolve, "original dynamic resolver available");
    const auto legacy_query =
        reinterpret_cast<LegacyQuery>(resolve(RTLD_DEFAULT, "cuGetProcAddress"));
    const auto modern_query =
        reinterpret_cast<ModernQuery>(resolve(RTLD_DEFAULT, "cuGetProcAddress_v2"));
    require(legacy_query && modern_query, "both public proc-address resolver ABIs available");
    using Query = std::function<CUresult(const char*, void**, int, cuuint64_t)>;
    const std::array<Query, 2> queries{
        legacy_query,
        [modern_query](const char* name, void** output, int version, cuuint64_t flags) {
            return modern_query(name, output, version, flags, nullptr);
        }};
    for (const auto& query : queries) {
        for (const auto flags :
             {CU_GET_PROC_ADDRESS_LEGACY_STREAM, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM}) {
            void* legacy{};
            void* modern{};
            void* native_modern{};
            require(query("cuStreamGetCtx", &legacy, 9020, flags) == CUDA_SUCCESS && legacy,
                    "exact legacy stream-context signature remains covered");
            {
                const NativeDriverCallScope native;
                require(query("cuStreamGetCtx", &native_modern, 12050, flags) == CUDA_SUCCESS &&
                            native_modern,
                        "original newer stream-context resolver available");
            }
            require(query("cuStreamGetCtx", &modern, 12050, flags) == CUDA_SUCCESS &&
                        modern == native_modern && modern != legacy,
                    "newer resolver must not substitute the old two-argument wrapper");
            CUcontext context{};
            CUgreenCtx green = reinterpret_cast<CUgreenCtx>(0x4560);
            const auto modern_status =
                reinterpret_cast<Modern>(modern)(kActualStream, &context, &green);
            if (modern_status != CUDA_SUCCESS || !context || green) {
                std::cerr << "modern context ABI resolver=" << (&query - queries.data())
                          << " flags=" << flags << " status=" << modern_status
                          << " context=" << context << " green=" << green << '\n';
            }
            require(modern_status == CUDA_SUCCESS && context && !green,
                    "native three-argument ABI writes both context outputs");
            context = nullptr;
            require(reinterpret_cast<Legacy>(legacy)(kActualStream, &context) == CUDA_SUCCESS &&
                        context,
                    "old wrapper retains its exact two-argument ABI");
        }
    }
#endif
}

void check_source_kernel_permission() {
    auto& gate = cooperative_process_call_gate();
    const auto function = reinterpret_cast<CUfunction>(0x4560);
    RuntimeSourceKernelShape shape{{1, 2, 3}, {4, 5, 6}, 32, false};
    unsigned int roots{};
    unsigned int issues{};
    unsigned int quarantines{};
    const auto quarantine = [&] { ++quarantines; };
    require(invoke_runtime_source_kernel_original(
                shape,
                [&] {
                    ++roots;
                    return cudaSuccess;
                },
                quarantine) == cudaErrorUnknown &&
                roots == 0 && quarantines == 0,
            "unadmitted source kernel permission cannot invoke Runtime");
    const ProcessCallGate::Call caller(gate);
    const ProcessCallGate::Transition transition(gate);
    for (const bool is_ptds : {false, true}) {
        shape.is_ptds = is_ptds;
        const auto status = invoke_runtime_source_kernel_original(
            shape,
            [&] {
                ++roots;
                require(invoke_runtime_source_kernel_original(
                            shape, [] { return cudaSuccess; }, quarantine) == cudaErrorUnknown,
                        "nested source kernel permission cannot replace the current owner");
                const auto issued = dispatch_runtime_source_kernel(function, shape, [&] {
                    ++issues;
                    return CUDA_SUCCESS;
                });
                require(issued && *issued == CUDA_SUCCESS,
                        "one matching actual Driver source issue");
                return cudaSuccess;
            },
            quarantine);
        require(status == cudaSuccess && !is_inside_runtime_call() && !is_inside_driver_call(),
                "matching legacy/PTDS source dispatch preserves Runtime status and scope");
    }
    require(roots == 2 && issues == 2 && quarantines == 0 &&
                !dispatch_runtime_source_kernel(function, shape, [] { return CUDA_SUCCESS; }),
            "exact issue counts and no escaped source permission");
    constexpr unsigned int fault_modes = 11;
    for (unsigned int mode = 0; mode < fault_modes; ++mode) {
        const auto before = issues;
        const auto status = invoke_runtime_source_kernel_original(
            shape,
            [&]() -> cudaError_t {
                if (mode == 0) {
                    return cudaSuccess;
                }
                if (mode == 4) {
                    throw std::runtime_error("unknown source Runtime effects");
                }
                auto actual = shape;
                if (mode == 1) {
                    ++actual.grid[0];
                } else if (mode == 5) {
                    ++actual.block[1];
                } else if (mode == 6) {
                    ++actual.shared_memory_bytes;
                } else if (mode == 7) {
                    actual.is_ptds = !actual.is_ptds;
                }
                const auto execute = [&]() -> CUresult {
                    ++issues;
                    if (mode == 10) {
                        throw std::runtime_error("uncertain native source execution");
                    }
                    return mode == 2 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
                };
                const auto issued = mode == 9
                                        ? dispatch_runtime_source_kernel(function, actual, {})
                                        : dispatch_runtime_source_kernel(
                                              mode == 8 ? nullptr : function, actual, execute);
                require(issued.has_value(), "source callback returns an owner result");
                if (mode == 3) {
                    const auto duplicate = dispatch_runtime_source_kernel(function, shape, [&] {
                        ++issues;
                        return CUDA_SUCCESS;
                    });
                    require(duplicate && *duplicate == CUDA_ERROR_INVALID_VALUE,
                            "duplicate source callback rejects before issue");
                }
                return cudaSuccess;
            },
            quarantine);
        require(status == cudaErrorUnknown && quarantines == mode + 1 &&
                    issues == before + ((mode == 2 || mode == 3 || mode == 10) ? 1U : 0U),
                "missing/mismatch/swallowed error/duplicate/throw quarantines without replay");
    }
    require(invoke_runtime_source_kernel_original(
                shape,
                [&] {
                    const auto issued = dispatch_runtime_source_kernel(
                        function, shape, [] { return CUDA_ERROR_INVALID_VALUE; });
                    require(issued && *issued == CUDA_ERROR_INVALID_VALUE,
                            "matched native Driver error");
                    return cudaErrorApiFailureBase;
                },
                quarantine) == cudaErrorApiFailureBase &&
                quarantines == fault_modes,
            "matched original Runtime error stays separate from Driver result");
}

void check_normal_kernel_permission() {
    auto& gate = cooperative_process_call_gate();
    const auto function = reinterpret_cast<CUfunction>(0x4560);
    RuntimeCooperativeLaunch plan{function, std::nullopt, {1, 2, 3}, {4, 5, 6}};
    plan.shared_memory_bytes = 32;
    unsigned int roots{}, issues{}, executions{}, quarantines{};
    const auto quarantine = [&] { ++quarantines; };
    const RuntimeNativeKernelExecution execution = [&](CUfunction actual, CUstream stream, bool,
                                                       const auto& original) {
        require(actual == function && stream == kActualStream && is_inside_runtime_call() &&
                    !is_inside_driver_call(),
                "normal permission supplies only exact typed SDK leaf handles");
        ++executions;
        const NativeDriverCallScope native;
        return original();
    };
    const auto original = [&] {
        ++roots;
        return cudaSuccess;
    };
    require(invoke_runtime_cooperative_launch_original(plan, false, original, quarantine,
                                                       execution) == cudaErrorUnknown &&
                roots == 0,
            "unentered normal permission cannot invoke Runtime");
    const ProcessCallGate::Call call(gate);
    for (bool is_ptds : {false, true}) {
        const RuntimeSourceKernelShape shape{plan.grid, plan.block, 32, is_ptds};
        require(invoke_runtime_cooperative_launch_original(
                    plan, is_ptds,
                    [&] {
                        ++roots;
                        require(
                            invoke_runtime_cooperative_launch_original(
                                plan, is_ptds, original, quarantine, execution) == cudaErrorUnknown,
                            "nested normal permission cannot replace its owner");
                        std::optional<CUresult> foreign;
                        std::thread other([&] {
                            foreign = dispatch_runtime_native_kernel(function, kActualStream, shape,
                                                                     [] { return CUDA_SUCCESS; });
                        });
                        other.join();
                        require(!foreign, "normal permission remains caller-local");
                        const auto leaf =
                            dispatch_runtime_native_kernel(function, kActualStream, shape, [&] {
                                ++issues;
                                return CUDA_SUCCESS;
                            });
                        require(leaf && *leaf == CUDA_SUCCESS, "one exact normal native execution");
                        return cudaSuccess;
                    },
                    quarantine, execution) == cudaSuccess,
                "normal legacy/PTDS permission returns SDK success");
        require(!is_inside_runtime_call() && !is_inside_driver_call() &&
                    !runtime_cooperative_launch_is_pending(),
                "normal TLS restored after return");
    }
    require(roots == 2 && issues == 2 && executions == 2 && quarantines == 0,
            "exact successful normal permission counts");
    const RuntimeSourceKernelShape guarded_shape{plan.grid, plan.block, 32, false};
    for (unsigned int mode = 0; mode < 4; ++mode) {
        const auto before = issues;
        const RuntimeNativeKernelExecution invalid_owner = [&](CUfunction, CUstream, bool,
                                                               const auto& owned) {
            if (mode == 0) return CUDA_SUCCESS;
            if (mode == 1) return owned();  // No native effect permission.
            if (mode == 2) {
                CUresult foreign{};
                std::thread worker([&] { foreign = owned(); });
                worker.join();
                return foreign;
            }
            const NativeDriverCallScope native;
            require(owned() == CUDA_SUCCESS && owned() == CUDA_ERROR_INVALID_VALUE,
                    "borrowed original cannot authorize duplicate effects");
            return CUDA_SUCCESS;  // Deliberately swallow the rejected duplicate.
        };
        require(invoke_runtime_cooperative_launch_original(
                    plan, false,
                    [&] {
                        (void)dispatch_runtime_native_kernel(function, kActualStream, guarded_shape,
                                                             [&] {
                                                                 ++issues;
                                                                 return CUDA_SUCCESS;
                                                             });
                        return cudaSuccess;
                    },
                    quarantine, invalid_owner) == cudaErrorUnknown &&
                    issues == before + (mode == 3 ? 1U : 0U),
                "missing/unscoped/foreign/duplicate owner effects retain without replay");
    }
    require(quarantines == 4, "each invalid owner publishes exactly one quarantine");
    auto empty_plan = plan;
    empty_plan.grid[0] = 0;
    require(invoke_runtime_cooperative_launch_original(
                empty_plan, false, [] { return cudaErrorInvalidConfiguration; }, quarantine,
                execution) == cudaErrorInvalidConfiguration &&
                quarantines == 4,
            "zero-sized SDK validation error makes no epoch or quarantine");
    require(invoke_runtime_cooperative_launch_original(
                plan, false, [] { return cudaErrorInvalidValue; }, quarantine, execution) ==
                    cudaErrorInvalidValue &&
                quarantines == 5,
            "arbitrary missing-leaf error preserves enum but cannot prove no issue");
    for (unsigned int mode = 0; mode < 10; ++mode) {
        const auto before = issues;
        const auto result = invoke_runtime_cooperative_launch_original(
            plan, false,
            [&]() -> cudaError_t {
                if (mode == 0) return cudaSuccess;
                if (mode == 9) throw std::runtime_error("uncertain normal Runtime effects");
                RuntimeSourceKernelShape shape{plan.grid, plan.block, 32, false};
                if (mode == 1) ++shape.grid[0];
                if (mode == 2) ++shape.block[1];
                if (mode == 3) ++shape.shared_memory_bytes;
                if (mode == 4) shape.is_ptds = true;
                const auto leaf = dispatch_runtime_native_kernel(
                    mode == 5 ? nullptr : function, kActualStream, shape, [&]() -> CUresult {
                        ++issues;
                        if (mode == 8) throw std::runtime_error("uncertain normal leaf");
                        return mode == 6 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
                    });
                require(leaf.has_value(), "normal fault reports an owner result");
                if (mode == 7) {
                    const auto duplicate =
                        dispatch_runtime_native_kernel(function, kActualStream, shape, [&] {
                            ++issues;
                            return CUDA_SUCCESS;
                        });
                    require(duplicate && *duplicate == CUDA_ERROR_INVALID_VALUE,
                            "duplicate normal leaf never repeats original");
                }
                return cudaSuccess;
            },
            quarantine, execution);
        require(result == cudaErrorUnknown && quarantines == mode + 6 &&
                    issues == before + ((mode >= 6 && mode <= 8) ? 1U : 0U),
                "normal missing/mismatch/swallowed/duplicate/exception retains without replay");
        require(!is_inside_runtime_call() && !is_inside_driver_call() &&
                    !runtime_cooperative_launch_is_pending(),
                "normal fault restores TLS");
    }
    const RuntimeSourceKernelShape shape{plan.grid, plan.block, 32, false};
    require(invoke_runtime_cooperative_launch_original(
                plan, false,
                [&] {
                    const auto leaf = dispatch_runtime_native_kernel(
                        function, kActualStream, shape, [] { return CUDA_ERROR_INVALID_VALUE; });
                    require(leaf && *leaf == CUDA_ERROR_INVALID_VALUE,
                            "normal original Driver error");
                    return cudaErrorApiFailureBase;
                },
                quarantine, execution) == cudaErrorApiFailureBase &&
                quarantines == 15,
            "matched original Runtime error remains separate from CUresult ownership");
}

void check_private_runtime_kernel_bypass() {
    DriverDispatch driver;
    require(driver.initialize(), "load fake Driver for private kernel invocation");
    const auto resolver = resolve_real_dlsym();
    require(resolver, "resolve original loader for exact private kernel observer");
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    require(library, "borrow already-loaded fake Driver kernel observer");
    const auto set_capture =
        reinterpret_cast<void (*)(int)>(resolver(library, "glimmer_fake_set_capture_status"));
    const auto count = reinterpret_cast<unsigned int (*)(unsigned int)>(
        resolver(library, "glimmer_fake_capture_query_count"));
    require(dlclose(library) == 0, "balance private kernel observer module reference");
    require(set_capture && count, "exact private kernel observer functions are available");
    set_capture(CU_STREAM_CAPTURE_STATUS_ACTIVE);
    const std::array before{count(0), count(1)};
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    using InternalLaunch =
        cudaError_t (*)(cudaKernel_t, dim3, dim3, void**, std::size_t, cudaStream_t);
    for (const bool is_ptds : {false, true}) {
        const auto conventional = reinterpret_cast<Launch>(
            find_interceptor_symbol(is_ptds ? "cudaLaunchKernel_ptsz" : "cudaLaunchKernel"));
        const auto compiler = reinterpret_cast<InternalLaunch>(
            find_interceptor_symbol(is_ptds ? "__cudaLaunchKernel_ptsz" : "__cudaLaunchKernel"));
        require(conventional && compiler, "exact private Runtime launch wrappers are available");
        {
            const NativeDriverCallScope native;
            // The fake SDK accepts these inputs. Native validation must own
            // them; preview capture/parameter admission may not intervene.
            require(conventional(nullptr, dim3{0}, dim3{0}, nullptr, 0, nullptr) == cudaSuccess &&
                        compiler(nullptr, dim3{0}, dim3{0}, nullptr, 0, nullptr) == cudaSuccess,
                    "private conventional/compiler launches preserve original SDK success");
            require(is_inside_driver_call() && !is_inside_runtime_call() &&
                        !runtime_cooperative_launch_is_pending(),
                    "private kernel invocation preserves outer native guard without a plan");
        }
        require(!is_inside_driver_call() && !is_inside_runtime_call(),
                "private Runtime launch cannot leak either lexical guard");
    }
    require(before == std::array{count(0), count(1)},
            "private Runtime launches issue no capture query in either alias");
    set_capture(CU_STREAM_CAPTURE_STATUS_NONE);
}

void check_context_query_failure() {
    {
        const NativeDriverCallScope native;
        CUcontext context{};
        require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context &&
                    cuCtxDestroy(context) == CUDA_SUCCESS,
                "inject fake stream-context query failure");
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call caller(gate);
    const ProcessCallGate::Transition transition(gate);
    unsigned int callbacks{};
    unsigned int leaves{};
    unsigned int lookups{};
    RuntimeCaptureObservation observed;
    cudaError_t native_status{};
    const auto result = invoke_runtime_capture_original(
        [&](CUcontext, CUstream, bool, CudaSourceCaptureKind, const auto&) {
            ++callbacks;
            return CUDA_SUCCESS;
        },
        [&] {
            const auto dispatched = dispatch_runtime_capture_origin(
                kActualStream, false, CudaSourceCaptureKind::kBegin, [&] {
                    ++leaves;
                    return CUDA_SUCCESS;
                });
            require(dispatched && *dispatched == CUDA_ERROR_INVALID_CONTEXT,
                    "failed native metadata query preserves exact error");
            return cudaErrorInvalidResourceHandle;
        },
        observed, native_status,
        [&](CUstream, bool, CUcontext&) {
            ++lookups;
            return CUDA_SUCCESS;
        });
    require(result == CUDA_ERROR_UNKNOWN && native_status == cudaErrorInvalidResourceHandle &&
                observed.was_runtime_invoked && observed.original_driver_calls == 0 &&
                !observed.context && !observed.graph && callbacks == 0 && leaves == 0 &&
                lookups == 0 && !is_inside_runtime_call() && !is_inside_driver_call(),
            "failed query cannot manufacture an accepted origin or replay original");
}

void check_normal_timing_contract() {
    const ProcessCallGate::Call caller(cooperative_process_call_gate());
    const auto first = reinterpret_cast<CUevent>(0x6780);
    const auto last = reinterpret_cast<CUevent>(0x7890);
    unsigned int observations{};
    const auto ready = [&](CUevent event) {
        require(event == (observations == 0 ? first : last), "both exact events observed in order");
        ++observations;
        return CUDA_SUCCESS;
    };
    float output{42.0F};
    unsigned int originals{};
    const auto measured =
        measure_runtime_event_timing(&output, first, last, ready, [&](float* sample) {
            ++originals;
            require(observations == 2, "both observations precede native timestamp sampling");
            require(sample && sample != &output,
                    "normal timing stages its write-only native sample");
            *sample = 7.25F;
            return cudaSuccess;
        });
    require(measured == cudaSuccess && originals == 1 && output == 7.25F,
            "unknown native timing publishes exact original sample once without a boundary");
    for (const auto error :
         {cudaErrorNotReady, cudaErrorInvalidResourceHandle, cudaErrorApiFailureBase}) {
        output = 42.0F;
        originals = 0;
        observations = 0;
        const auto failed =
            measure_runtime_event_timing(&output, first, last, ready, [&](float* sample) {
                ++originals;
                *sample = 9.0F;
                return error;
            });
        require(failed == error && originals == 1 && output == 42.0F,
                "SDK error including outside CUresult preserves status and caller output");
    }
    observations = 0;
    require(measure_runtime_event_timing(
                nullptr, first, last, ready,
                [](float* sample) {
                    require(!sample, "null output remains null for original SDK validation");
                    return cudaErrorInvalidValue;
                }) == cudaErrorInvalidValue,
            "null output receives unchanged native validation error");
    bool has_thrown{};
    observations = 0;
    try {
        (void)measure_runtime_event_timing(&output, first, last, ready, [](float*) -> cudaError_t {
            throw std::runtime_error("native timing failure");
        });
    } catch (const std::runtime_error&) {
        has_thrown = true;
    }
    observations = 0;
    require(has_thrown &&
                measure_runtime_event_timing(&output, first, last, ready,
                                             [](float* sample) {
                                                 *sample = 3.5F;
                                                 return cudaSuccess;
                                             }) == cudaSuccess &&
                output == 3.5F,
            "native exception does not replay or retain a staged sample");
    for (const auto start_status : {CUDA_SUCCESS, CUDA_ERROR_NOT_READY, CUDA_ERROR_UNKNOWN}) {
        for (const auto end_status : {CUDA_SUCCESS, CUDA_ERROR_NOT_READY, CUDA_ERROR_UNKNOWN}) {
            output = 42.0F;
            const auto observe = [&](CUevent event) {
                return event == first ? start_status : end_status;
            };
            const auto expected =
                start_status == CUDA_ERROR_UNKNOWN || end_status == CUDA_ERROR_UNKNOWN
                    ? cudaErrorUnknown
                : start_status == CUDA_ERROR_NOT_READY || end_status == CUDA_ERROR_NOT_READY
                    ? cudaErrorNotReady
                    : cudaSuccess;
            const auto result =
                measure_runtime_event_timing(&output, first, last, observe, [](float* sample) {
                    *sample = 7.25F;
                    return cudaSuccess;
                });
            require(result == expected && output == (expected == cudaSuccess ? 7.25F : 42.0F),
                    "both event readiness results control exact staged output publication");
            require(measure_runtime_event_timing(&output, first, last, observe,
                                                 [](float*) {
                                                     return cudaErrorInvalidResourceHandle;
                                                 }) == cudaErrorInvalidResourceHandle,
                    "native validation errors precede all logical pending/failure statuses");
        }
    }
    {
        const NativeDriverCallScope native;
        require(!intercept_normal_runtime_event_elapsed_time(&output, first, last,
                                                             [](float*) { return cudaSuccess; }),
                "trusted private calls retain original timing path");
        require(!intercept_normal_runtime_device_completion([] { return cudaSuccess; }),
                "trusted private calls retain original device completion path");
    }
    {
        const ProcessCallGate::Transition source(cooperative_process_call_gate());
        require(!intercept_normal_runtime_event_elapsed_time(&output, first, last,
                                                             [](float*) { return cudaSuccess; }),
                "source ownership receives no normal timing permission");
        require(!intercept_normal_runtime_device_completion([] { return cudaSuccess; }),
                "source ownership receives no normal device completion permission");
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]).starts_with("--epoch-environment-")) {
            check_epoch_environment(std::string_view(argv[1]).substr(20));
            return EXIT_SUCCESS;
        }
        if (argc == 2 && (std::string_view(argv[1]) == "--epoch-startup" ||
                          std::string_view(argv[1]) == "--epoch-startup-late")) {
            check_process_epoch_startup(std::string_view(argv[1]) == "--epoch-startup-late");
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cold-capacity") {
            check_cold_sdk_capacity();
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--private-kernel-bypass") {
            check_private_runtime_kernel_bypass();
            return EXIT_SUCCESS;
        }
        unsigned int capture_calls{};
        require(setenv("GLIMMER_COOPERATIVE_DRIVER_PREVIEW", "1", 1) == 0 &&
                    setenv("GLIMMER_COOPERATIVE_RUNTIME_PREVIEW", "0", 1) == 0,
                "configure Driver-only preview capture policy");
        const auto capture_original = [&] {
            ++capture_calls;
            return cudaSuccess;
        };
        require(intercept_runtime_capture(CudaSourceCaptureKind::kBegin, capture_original) ==
                        cudaErrorNotSupported &&
                    capture_calls == 0,
                "disabled Runtime preview cannot bypass a live Driver preview owner");
        {
            const NativeDriverCallScope private_call;
            require(intercept_runtime_capture(CudaSourceCaptureKind::kBegin, capture_original) ==
                            cudaSuccess &&
                        capture_calls == 1,
                    "trusted private Runtime capture retains exact original dispatch");
        }
        require(setenv("GLIMMER_COOPERATIVE_DRIVER_PREVIEW", "1", 1) == 0 &&
                    setenv("GLIMMER_COOPERATIVE_RUNTIME_PREVIEW", "1", 1) == 0,
                "restore CPU fixture Runtime preview configuration");
        check_private_sdk_reset_cache_binding();
        check_private_sdk_destroy_cache_binding(false);
        check_private_sdk_destroy_cache_binding(true);
        check_scope();
        check_borrowed_context_completion();
        check_stream_context_abi();
        check_source_kernel_permission();
        check_normal_kernel_permission();
        check_runtime_graph_contract();
        check_normal_timing_contract();
        check_context_query_failure();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
