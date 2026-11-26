#include <cuda_runtime.h>
#include <cuda.h>

#include "execution_binary_kernel_fixture.cu"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>

namespace {
constexpr unsigned int kWorkers = 320, kCount = 4097, kGuards = 32;
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(status));
}
void check(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(status));
}
struct DriverLibrary {
    void* handle{::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL)};
    DriverLibrary() {
        if (!handle) throw std::runtime_error("load original Driver ABI provider");
    }
    ~DriverLibrary() {
        if (::dlclose(handle) != 0) std::cerr << "Driver fixture library release failed\n";
    }
    DriverLibrary(const DriverLibrary&) = delete;
    DriverLibrary& operator=(const DriverLibrary&) = delete;
    template <typename Function>
    Function lookup(const char* name) const {
        auto* result = ::dlsym(handle, name);
        if (!result)
            throw std::runtime_error(std::string("missing original Driver entry: ") + name);
        return reinterpret_cast<Function>(result);
    }
};
void run_cross_context() {
    DriverLibrary driver;
    const auto initialize = driver.lookup<decltype(&cuInit)>("cuInit");
    const auto create = driver.lookup<decltype(&cuCtxCreate_v2)>("cuCtxCreate_v2");
    const auto select = driver.lookup<decltype(&cuCtxSetCurrent)>("cuCtxSetCurrent");
    const auto destroy = driver.lookup<decltype(&cuCtxDestroy_v2)>("cuCtxDestroy_v2");
    check(initialize(0), "initialize Driver metadata");
    constexpr unsigned int count = kWorkers * 32;
    std::vector<unsigned int> source_values(count + kGuards), target_values(count + kGuards);
    for (unsigned int index = 0; index < source_values.size(); ++index) {
        source_values[index] = index * 2;
        target_values[index] = index * 3;
    }
    const auto bytes = source_values.size() * sizeof(source_values[0]);
    CUcontext source_context{}, target_context{};
    cudaStream_t source_stream{}, target_stream{};
    unsigned int *source{}, *target{};
    check(create(&source_context, 0, 0), "create source context");
    check(cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking), "source stream");
    check(cudaMalloc(reinterpret_cast<void**>(&source), bytes), "source allocation");
    check(cudaMemcpy(source, source_values.data(), bytes, cudaMemcpyHostToDevice), "source input");
    binary_add<<<kWorkers, 32, 0, source_stream>>>(source, 0, ExecutionBinaryPair{});
    check(cudaGetLastError(), "source warmup");
    check(cudaStreamSynchronize(source_stream), "source warmup completion");
    cudaEvent_t recorded{};
    check(cudaEventCreateWithFlags(&recorded, cudaEventDisableTiming), "source event");
    check(create(&target_context, 0, 0), "create target context");
    check(cudaStreamCreateWithFlags(&target_stream, cudaStreamNonBlocking), "target stream");
    check(cudaMalloc(reinterpret_cast<void**>(&target), bytes), "target allocation");
    check(cudaMemcpy(target, target_values.data(), bytes, cudaMemcpyHostToDevice), "target input");
    binary_add<<<kWorkers, 32, 0, target_stream>>>(target, 0, ExecutionBinaryPair{});
    binary_high<<<kWorkers, 32, 0, target_stream>>>(target, 0);
    check(cudaGetLastError(), "target warmup");
    check(cudaStreamSynchronize(target_stream), "target warmup completion");
    check(cudaMemcpy(target, target_values.data(), bytes, cudaMemcpyHostToDevice),
          "restore target");
    check(select(source_context), "select source");
    binary_add<<<kWorkers, 32, 0, source_stream>>>(source, count,
                                                   ExecutionBinaryPair{3, 8192, {0, 0}});
    check(cudaGetLastError(), "source producer");
    check(cudaEventRecord(recorded, source_stream), "record foreign source generation");
    check(select(target_context), "select target");
    if (cudaStreamWaitEvent(target_stream, recorded, cudaEventWaitExternal) !=
        cudaErrorInvalidValue)
        throw std::runtime_error("external wait flags incorrectly accepted outside capture");
    (void)cudaGetLastError();
    binary_high<<<kWorkers, 32, 0, target_stream>>>(target, 100000000);
    check(cudaGetLastError(), "independent target predecessor");
    check(cudaStreamWaitEvent(target_stream, recorded, 0), "foreign context wait");
    binary_add<<<kWorkers, 32, 0, target_stream>>>(target, count,
                                                   ExecutionBinaryPair{7, 0, {0, 0}});
    check(cudaGetLastError(), "dependent target computation");
    check(select(source_context), "reselect source for teardown");
    check(cudaEventDestroy(recorded), "destroy source public event");
    check(cudaMemcpy(source_values.data(), source, bytes, cudaMemcpyDeviceToHost), "source output");
    check(cudaFree(source), "release source allocation");
    check(cudaStreamDestroy(source_stream), "release source stream");
    check(destroy(source_context), "destroy source with accepted foreign references");
    check(select(target_context), "reselect surviving target");
    check(cudaStreamSynchronize(target_stream), "target logical completion");
    check(cudaMemcpy(target_values.data(), target, bytes, cudaMemcpyDeviceToHost), "target output");
    for (unsigned int index = 0; index < source_values.size(); ++index)
        if (source_values[index] != index * 2 + (index < count ? 3U : 0U) ||
            target_values[index] != index * 3 + (index < count ? 8U : 0U))
            throw std::runtime_error(
                "cross-context dependency, once-only result or guard mismatch");
    check(cudaFree(target), "release target allocation");
    check(cudaStreamDestroy(target_stream), "release target stream");
    check(destroy(target_context), "destroy target context");
    std::cout << "Ordinary cross-context CUDA events verified: source_values="
              << source_values.size() << "; target_values=" << target_values.size()
              << "; workers=" << kWorkers << "; full_device_qualified=false\n";
}
__global__ void event_gate(volatile unsigned int* state) {
    while (state[0] != 1) {
    }
    state[1] = 19;
}
void run() {
    std::vector<unsigned int> values(kCount + kGuards);
    for (unsigned int index = 0; index < values.size(); ++index) values[index] = index * 2;
    unsigned int *output{}, *gate_state{}, *release{};
    check(cudaMalloc(reinterpret_cast<void**>(&output), values.size() * sizeof(values[0])),
          "allocate output");
    check(cudaMalloc(reinterpret_cast<void**>(&gate_state), 2 * sizeof(unsigned int)),
          "allocate gate");
    check(cudaMallocHost(reinterpret_cast<void**>(&release), sizeof(unsigned int)),
          "pinned release");
    *release = 1;
    check(cudaMemcpy(output, values.data(), values.size() * sizeof(values[0]),
                     cudaMemcpyHostToDevice),
          "initial values");
    check(cudaMemcpy(gate_state, release, sizeof(unsigned int), cudaMemcpyHostToDevice),
          "warm gate state");
    cudaStream_t producer{}, consumer{}, independent{};
    check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), "producer");
    check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "consumer");
    check(cudaStreamCreateWithFlags(&independent, cudaStreamNonBlocking), "independent");
    // Ordinary warmups prepare all exact host identities before the pending
    // dependency subcase; no Glimmer API or application-owned scheduler exists.
    binary_add<<<kWorkers, 32, 0, producer>>>(output, 0, ExecutionBinaryPair{});
    binary_no_parameters<<<1, 1, 0, independent>>>();
    event_gate<<<1, 1, 0, independent>>>(gate_state);
    check(cudaGetLastError(), "warmup launches");
    check(cudaDeviceSynchronize(), "warmups");
    check(cudaMemset(gate_state, 0, 2 * sizeof(unsigned int)), "reset gate");
    cudaEvent_t unrecorded{}, selected{}, start{}, finish{};
    check(cudaEventCreateWithFlags(&unrecorded, cudaEventDisableTiming), "unrecorded event");
    check(cudaEventCreate(&selected), "rerecorded event");
    check(cudaEventCreate(&start), "start timestamp");
    check(cudaEventCreate(&finish), "finish timestamp");
    if (cudaEventRecordWithFlags(selected, producer, cudaEventRecordExternal) !=
            cudaErrorInvalidValue ||
        cudaStreamWaitEvent(consumer, selected, cudaEventWaitExternal) != cudaErrorInvalidValue)
        throw std::runtime_error("external event flags incorrectly accepted outside capture");
    (void)cudaGetLastError();
    event_gate<<<1, 1, 0, independent>>>(gate_state);
    check(cudaGetLastError(), "pending gate");
    check(cudaStreamWaitEvent(producer, unrecorded, 0), "unrecorded wait is a no-op");
    check(cudaEventRecord(unrecorded, independent), "future record after gated consumer");
    check(cudaMemcpyAsync(gate_state, release, sizeof(unsigned int), cudaMemcpyHostToDevice,
                          producer),
          "independent gate release");
    check(cudaEventSynchronize(unrecorded), "gate event completion");
    unsigned int observed[2]{};
    check(cudaMemcpy(observed, gate_state, sizeof(observed), cudaMemcpyDeviceToHost),
          "gate result");
    if (observed[0] != 1 || observed[1] != 19)
        throw std::runtime_error("unrecorded wait gained a future dependency");
    check(cudaEventRecord(start, producer), "record start");
    binary_add<<<kWorkers, 32, 0, producer>>>(output, kCount, ExecutionBinaryPair{3, 8192, {0, 0}});
    check(cudaGetLastError(), "producer launch");
    check(cudaEventRecord(selected, producer), "record producer version");
    check(cudaStreamWaitEvent(consumer, selected, 0), "consumer captures producer version");
    binary_add<<<kWorkers, 32, 0, consumer>>>(output, kCount, ExecutionBinaryPair{7, 0, {0, 0}});
    check(cudaGetLastError(), "consumer launch");
    binary_no_parameters<<<1, 1, 0, independent>>>();
    check(cudaGetLastError(), "independent launch");
    check(cudaEventRecord(selected, independent), "rerecord cannot retarget old wait");
    check(cudaEventDestroy(selected), "destroy while captured versions may be pending");
    check(cudaEventRecord(finish, consumer), "record logical consumer finish");
    check(cudaEventSynchronize(finish), "consumer completion");
    float milliseconds = -1;
    check(cudaEventElapsedTime(&milliseconds, start, finish), "original timestamps");
    if (!std::isfinite(milliseconds) || milliseconds < 0)
        throw std::runtime_error("invalid original elapsed time");
    check(cudaMemcpy(values.data(), output, values.size() * sizeof(values[0]),
                     cudaMemcpyDeviceToHost),
          "complete output snapshot");
    for (unsigned int index = 0; index < values.size(); ++index)
        if (values[index] != index * 2 + (index < kCount ? 10U : 0U))
            throw std::runtime_error("event dependency, once-only computation or guard mismatch");
    check(cudaEventDestroy(unrecorded), "destroy gate event");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(finish), "destroy finish");
    check(cudaStreamDestroy(producer), "destroy producer");
    check(cudaStreamDestroy(consumer), "destroy consumer");
    check(cudaStreamDestroy(independent), "destroy independent");
    check(cudaFreeHost(release), "release pinned memory");
    check(cudaFree(gate_state), "release gate memory");
    check(cudaFree(output), "release output");
    std::cout << "Ordinary CUDA events verified: outputs=" << values.size()
              << "; gated_unrecorded=1; rerecord_destroy=1; workers=" << kWorkers
              << "; full_device_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 1)
            run();
        else if (argc == 2 && std::string(argv[1]) == "--cross-context")
            run_cross_context();
        else
            throw std::runtime_error("select no arguments or --cross-context");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
