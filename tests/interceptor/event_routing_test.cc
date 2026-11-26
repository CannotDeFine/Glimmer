#include "event_audit_fixture.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
void* g_late_interceptor = nullptr;
void require(bool value, const std::string& message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void* resolve(void* library, const char* name, unsigned int mode) {
    void* pointer = nullptr;
    if (mode < 2) {
        pointer = dlsym(g_late_interceptor != nullptr ? g_late_interceptor
                        : mode == 0                   ? RTLD_DEFAULT
                                                      : library,
                        name);
    } else if (mode == 2) {
        auto proc = reinterpret_cast<CUresult (*)(const char*, void**, int, cuuint64_t)>(dlsym(
            g_late_interceptor != nullptr ? g_late_interceptor : library, "cuGetProcAddress"));
        const auto status = proc ? proc(name, &pointer, 12060, 0) : CUDA_ERROR_UNKNOWN;
        require(status == CUDA_SUCCESS,
                std::string(name) + " proc v1 status=" + std::to_string(status));
    } else {
        auto proc = reinterpret_cast<CUresult (*)(const char*, void**, int, cuuint64_t,
                                                  CUdriverProcAddressQueryResult*)>(
            dlsym(g_late_interceptor != nullptr ? g_late_interceptor : library,
                  "cuGetProcAddress_v2"));
        CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        const auto result = proc ? proc(name, &pointer, 12060, 0, &status) : CUDA_ERROR_UNKNOWN;
        require(result == CUDA_SUCCESS && status == CU_GET_PROC_ADDRESS_SUCCESS,
                std::string(name) + " proc v2 result=" + std::to_string(result) +
                    " query=" + std::to_string(status));
    }
    Dl_info info{};
    require(pointer && dladdr(pointer, &info) != 0 && info.dli_fname &&
                std::string_view(info.dli_fname).find("libglimmer_cuda_interceptor") !=
                    std::string_view::npos,
            std::string(name) + " bypassed interceptor");
    return pointer;
}
void verify(const glimmer::tests::EventAudit& actual, const glimmer::tests::EventAudit& expected,
            const char* name) {
    require(actual.calls == expected.calls && actual.output == expected.output &&
                actual.event == expected.event && actual.stream == expected.stream &&
                actual.end == expected.end && actual.flags == expected.flags,
            std::string(name) + " changed native arguments/alias");
}
template <typename Function, typename Result, typename... Arguments>
void check_native_call(void* library, unsigned int mode, glimmer::tests::EventAudit& audit,
                       const char* name, glimmer::tests::EventCall call, void* output, void* event,
                       void* stream, void* end, unsigned int flags, Result result,
                       Arguments... arguments) {
    audit = {};
    audit.result = static_cast<int>(result);
    glimmer::tests::EventAudit expected{};
    glimmer::tests::record_event_call(expected, call, output, event, stream, end, flags);
    const auto function = reinterpret_cast<Function>(resolve(library, name, mode));
    require(function(arguments...) == result, std::string(name) + " changed native error");
    verify(audit, expected, name);
}
void driver_cases(void* library) {
    auto get_audit = reinterpret_cast<glimmer::tests::EventAudit* (*)()>(
        dlsym(library, "glimmer_fake_event_audit"));
    require(get_audit != nullptr, "native event audit lookup");
    auto& audit = *get_audit();
    CUevent event = reinterpret_cast<CUevent>(0x7100);
    CUevent start = event;
    CUevent end = reinterpret_cast<CUevent>(0x7200);
    CUstream stream = reinterpret_cast<CUstream>(0x7300);
    float milliseconds = 0;
    unsigned int flags = 0x5a;
    for (unsigned int mode = 0; mode < 4; ++mode) {
        for (const auto result : {CUDA_SUCCESS, CUDA_ERROR_NOT_READY, CUDA_ERROR_INVALID_VALUE,
                                  CUDA_ERROR_LAUNCH_FAILED}) {
            check_native_call<CUresult (*)(CUevent* event, unsigned int flags)>(
                library, mode, audit, "cuEventCreate", glimmer::tests::EventCall::kCreate, &event,
                nullptr, nullptr, nullptr, flags, result, &event, flags);
            check_native_call<CUresult (*)(CUevent event, CUstream stream)>(
                library, mode, audit, "cuEventRecord", glimmer::tests::EventCall::kRecord, nullptr,
                event, stream, nullptr, 0, result, event, stream);
            check_native_call<CUresult (*)(CUevent event)>(
                library, mode, audit, "cuEventQuery", glimmer::tests::EventCall::kQuery, nullptr,
                event, nullptr, nullptr, 0, result, event);
            check_native_call<CUresult (*)(CUevent event)>(
                library, mode, audit, "cuEventDestroy_v2", glimmer::tests::EventCall::kDestroy,
                nullptr, event, nullptr, nullptr, 0, result, event);
            check_native_call<CUresult (*)(CUevent event, CUstream stream)>(
                library, mode, audit, "cuEventRecord_ptsz", glimmer::tests::EventCall::kRecordPtds,
                nullptr, event, stream, nullptr, 0, result, event, stream);
            check_native_call<CUresult (*)(CUevent event, CUstream stream, unsigned int flags)>(
                library, mode, audit, "cuEventRecordWithFlags",
                glimmer::tests::EventCall::kRecordFlags, nullptr, event, stream, nullptr, flags,
                result, event, stream, flags);
            check_native_call<CUresult (*)(CUevent event, CUstream stream, unsigned int flags)>(
                library, mode, audit, "cuEventRecordWithFlags_ptsz",
                glimmer::tests::EventCall::kRecordFlagsPtds, nullptr, event, stream, nullptr, flags,
                result, event, stream, flags);
            check_native_call<CUresult (*)(CUevent event)>(
                library, mode, audit, "cuEventSynchronize", glimmer::tests::EventCall::kSynchronize,
                nullptr, event, nullptr, nullptr, 0, result, event);
            check_native_call<CUresult (*)(float* milliseconds, CUevent start, CUevent end)>(
                library, mode, audit, "cuEventElapsedTime", glimmer::tests::EventCall::kElapsed,
                &milliseconds, start, nullptr, end, 0, result, &milliseconds, start, end);
            check_native_call<CUresult (*)(CUstream stream, CUevent event, unsigned int flags)>(
                library, mode, audit, "cuStreamWaitEvent", glimmer::tests::EventCall::kWait,
                nullptr, event, stream, nullptr, flags, result, stream, event, flags);
            check_native_call<CUresult (*)(CUstream stream, CUevent event, unsigned int flags)>(
                library, mode, audit, "cuStreamWaitEvent_ptsz",
                glimmer::tests::EventCall::kWaitPtds, nullptr, event, stream, nullptr, flags,
                result, stream, event, flags);
            check_native_call<CUresult (*)(CUevent event)>(
                library, mode, audit, "cuEventDestroy", glimmer::tests::EventCall::kDestroy,
                nullptr, event, nullptr, nullptr, 0, result, event);
        }
    }
    using Proc = CUresult (*)(const char*, void**, int, cuuint64_t);
    auto proc = reinterpret_cast<Proc>(
        dlsym(g_late_interceptor != nullptr ? g_late_interceptor : library, "cuGetProcAddress"));
    for (const auto* name : {"cuEventRecord", "cuEventRecordWithFlags", "cuStreamWaitEvent"}) {
        void* pointer = nullptr;
        require(proc(name, &pointer, 12060, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) ==
                        CUDA_SUCCESS &&
                    pointer == dlsym(g_late_interceptor != nullptr ? g_late_interceptor : library,
                                     (std::string(name) + "_ptsz").c_str()),
                std::string(name) + " proc-address PTDS selection");
    }
}
void missing_runtime_ptds(void* library) {
    auto get_audit = reinterpret_cast<glimmer::tests::EventAudit* (*)()>(
        dlsym(library, "glimmer_fake_event_audit"));
    require(get_audit != nullptr, "missing Runtime audit");
    auto& audit = *get_audit();
    audit = {};
    using Record = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    using RecordFlags = cudaError_t (*)(cudaEvent_t, cudaStream_t, unsigned int);
    using Wait = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);
    const auto record = reinterpret_cast<Record>(resolve(library, "cudaEventRecord_ptsz", 1));
    const auto flags =
        reinterpret_cast<RecordFlags>(resolve(library, "cudaEventRecordWithFlags_ptsz", 1));
    const auto wait = reinterpret_cast<Wait>(resolve(library, "cudaStreamWaitEvent_ptsz", 1));
    require(record(nullptr, nullptr) == cudaErrorNotSupported &&
                flags(nullptr, nullptr, 0) == cudaErrorNotSupported &&
                wait(nullptr, nullptr, 0) == cudaErrorNotSupported &&
                audit.calls == glimmer::tests::EventAudit{}.calls,
            "missing PTDS Runtime events must reject without legacy fallback");
    const auto legacy = reinterpret_cast<Record>(resolve(library, "cudaEventRecord", 1));
    require(legacy(nullptr, nullptr) == cudaSuccess &&
                audit.calls[static_cast<unsigned int>(glimmer::tests::EventCall::kRecord)] == 1,
            "missing PTDS must not disable independent legacy availability");
}
void runtime_cases(void* library) {
    auto get_audit = reinterpret_cast<glimmer::tests::EventAudit* (*)()>(
        dlsym(library, "glimmer_fake_event_audit"));
    require(get_audit != nullptr, "native event audit lookup");
    auto& audit = *get_audit();
    cudaEvent_t event = reinterpret_cast<cudaEvent_t>(0x7100);
    cudaEvent_t start = event;
    cudaEvent_t end = reinterpret_cast<cudaEvent_t>(0x7200);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(0x7300);
    float milliseconds = 0;
    unsigned int flags = 0x5a;
    for (unsigned int mode = 0; mode < 2; ++mode) {
        for (const auto result :
             {cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorLaunchFailure}) {
            check_native_call<cudaError_t (*)(cudaEvent_t* event)>(
                library, mode, audit, "cudaEventCreate", glimmer::tests::EventCall::kCreate, &event,
                nullptr, nullptr, nullptr, 0, result, &event);
            check_native_call<cudaError_t (*)(cudaEvent_t* event, unsigned int flags)>(
                library, mode, audit, "cudaEventCreateWithFlags",
                glimmer::tests::EventCall::kCreateWithFlags, &event, nullptr, nullptr, nullptr,
                flags, result, &event, flags);
            check_native_call<cudaError_t (*)(cudaEvent_t event, cudaStream_t stream)>(
                library, mode, audit, "cudaEventRecord", glimmer::tests::EventCall::kRecord,
                nullptr, event, stream, nullptr, 0, result, event, stream);
            check_native_call<cudaError_t (*)(cudaEvent_t event, cudaStream_t stream)>(
                library, mode, audit, "cudaEventRecord_ptsz",
                glimmer::tests::EventCall::kRecordPtds, nullptr, event, stream, nullptr, 0, result,
                event, stream);
            check_native_call<cudaError_t (*)(cudaEvent_t event, cudaStream_t stream,
                                              unsigned int flags)>(
                library, mode, audit, "cudaEventRecordWithFlags",
                glimmer::tests::EventCall::kRecordFlags, nullptr, event, stream, nullptr, flags,
                result, event, stream, flags);
            check_native_call<cudaError_t (*)(cudaEvent_t event, cudaStream_t stream,
                                              unsigned int flags)>(
                library, mode, audit, "cudaEventRecordWithFlags_ptsz",
                glimmer::tests::EventCall::kRecordFlagsPtds, nullptr, event, stream, nullptr, flags,
                result, event, stream, flags);
            check_native_call<cudaError_t (*)(cudaEvent_t event)>(
                library, mode, audit, "cudaEventQuery", glimmer::tests::EventCall::kQuery, nullptr,
                event, nullptr, nullptr, 0, result, event);
            check_native_call<cudaError_t (*)(cudaEvent_t event)>(
                library, mode, audit, "cudaEventSynchronize",
                glimmer::tests::EventCall::kSynchronize, nullptr, event, nullptr, nullptr, 0,
                result, event);
            check_native_call<cudaError_t (*)(cudaEvent_t event)>(
                library, mode, audit, "cudaEventDestroy", glimmer::tests::EventCall::kDestroy,
                nullptr, event, nullptr, nullptr, 0, result, event);
            check_native_call<cudaError_t (*)(float* milliseconds, cudaEvent_t start,
                                              cudaEvent_t end)>(
                library, mode, audit, "cudaEventElapsedTime", glimmer::tests::EventCall::kElapsed,
                &milliseconds, start, nullptr, end, 0, result, &milliseconds, start, end);
            check_native_call<cudaError_t (*)(cudaStream_t stream, cudaEvent_t event,
                                              unsigned int flags)>(
                library, mode, audit, "cudaStreamWaitEvent", glimmer::tests::EventCall::kWait,
                nullptr, event, stream, nullptr, flags, result, stream, event, flags);
            check_native_call<cudaError_t (*)(cudaStream_t stream, cudaEvent_t event,
                                              unsigned int flags)>(
                library, mode, audit, "cudaStreamWaitEvent_ptsz",
                glimmer::tests::EventCall::kWaitPtds, nullptr, event, stream, nullptr, flags,
                result, stream, event, flags);
        }
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const bool driver_only = argc == 2 && std::string_view(argv[1]) == "--driver-only";
        const bool missing_ptds =
            argc == 2 && std::string_view(argv[1]) == "--missing-runtime-ptds";
        if (driver_only) {
            g_late_interceptor = dlopen(GLIMMER_EVENT_INTERCEPTOR_PATH, RTLD_NOW | RTLD_GLOBAL);
            require(g_late_interceptor != nullptr, "late Driver interceptor loading");
        }
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        require(driver != nullptr, "fake Driver loading");
        driver_cases(driver);
        if (!driver_only) {
            void* runtime =
                dlopen(missing_ptds ? GLIMMER_EVENT_MISSING_RUNTIME_PATH : "libcudart.so",
                       RTLD_NOW | RTLD_GLOBAL);
            require(runtime != nullptr, "fake Runtime loading");
            if (missing_ptds) {
                missing_runtime_ptds(runtime);
            } else {
                runtime_cases(runtime);
            }
            require(dlclose(runtime) == 0, "Runtime close");
        }
        require(dlclose(driver) == 0, "Driver close");
        std::cout << (driver_only
                          ? "Driver-only event routing (Runtime sanitizer scope unverified)\n"
                          : "Driver/Runtime native event routing passed\n");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
