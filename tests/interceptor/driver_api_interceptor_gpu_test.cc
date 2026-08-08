#include <cuda.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;
constexpr std::size_t kSuccessfulRequestBytes = std::size_t{1} * 1024 * 1024;
constexpr std::size_t kRequestedBytes = std::size_t{16} * 1024 * 1024;
constexpr std::size_t kPitchWidthBytes = 1024;
constexpr std::size_t kPitchHeight = 1024;
constexpr unsigned int kPitchElementSizeBytes = 4;
constexpr unsigned int kStreamFlags = 0;

bool check(CUresult result, std::string_view operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }

    std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
    return false;
}

}  // namespace

int main() {
    if (!check(cuInit(0), "cuInit")) {
        return EXIT_FAILURE;
    }

    CUdevice device{};
    if (!check(cuDeviceGet(&device, 0), "cuDeviceGet")) {
        return EXIT_FAILURE;
    }

    CUcontext context{};
    if (!check(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate")) {
        return EXIT_FAILURE;
    }

    void* queried_memory_info = nullptr;
    CUdriverProcAddressQueryResult query_status{};
    const bool proc_address_lookup =
        check(cuGetProcAddress("cuMemGetInfo", &queried_memory_info, CUDA_VERSION,
                               CU_GET_PROC_ADDRESS_DEFAULT, &query_status),
              "cuGetProcAddress") &&
        queried_memory_info != nullptr && query_status == CU_GET_PROC_ADDRESS_SUCCESS;
    using MemoryInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);
    const auto queried_memory_info_function =
        reinterpret_cast<MemoryInfoFunction>(queried_memory_info);

    std::size_t free_bytes{};
    std::size_t total_bytes{};
    const bool has_memory_info =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2");
    const bool proc_address_call =
        proc_address_lookup &&
        check(queried_memory_info_function(&free_bytes, &total_bytes), "queried cuMemGetInfo") &&
        total_bytes == kQuotaBytes;

    std::size_t device_total_bytes{};
    const bool has_expected_device_total =
        check(cuDeviceTotalMem_v2(&device_total_bytes, device), "cuDeviceTotalMem_v2") &&
        device_total_bytes == kQuotaBytes;

    CUdeviceptr managed_pointer{};
    const bool managed_allocated =
        check(cuMemAllocManaged(&managed_pointer, kSuccessfulRequestBytes, CU_MEM_ATTACH_GLOBAL),
              "cuMemAllocManaged");
    const bool managed_released =
        !managed_allocated || check(cuMemFree_v2(managed_pointer), "cuMemFree_v2 managed");
    const bool has_expected_initial_free =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2 initial") &&
        free_bytes == kQuotaBytes;

    CUdeviceptr pitch_pointer{};
    std::size_t pitch_bytes{};
    const bool pitch_allocated =
        check(cuMemAllocPitch_v2(&pitch_pointer, &pitch_bytes, kPitchWidthBytes, kPitchHeight,
                                 kPitchElementSizeBytes),
              "cuMemAllocPitch_v2") &&
        pitch_bytes > 0;
    const std::size_t pitched_allocation_bytes = pitch_bytes * kPitchHeight;
    const bool has_pitched_usage =
        pitch_allocated && pitched_allocation_bytes <= kQuotaBytes &&
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2 pitched") &&
        free_bytes == kQuotaBytes - pitched_allocation_bytes;
    const bool pitch_released =
        !pitch_allocated || check(cuMemFree_v2(pitch_pointer), "cuMemFree_v2 pitched");

    void* unsupported_function = nullptr;
    CUdriverProcAddressQueryResult unsupported_status{};
    const bool unsupported_proc_address =
        check(cuGetProcAddress("glimmerUnsupportedFunction", &unsupported_function, CUDA_VERSION,
                               CU_GET_PROC_ADDRESS_DEFAULT, &unsupported_status),
              "unsupported cuGetProcAddress") &&
        unsupported_function == nullptr &&
        unsupported_status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;

    const bool has_expected_total = total_bytes == kQuotaBytes;

    CUdeviceptr successful_pointer{};
    const bool allocated_within_quota =
        check(cuMemAlloc_v2(&successful_pointer, kSuccessfulRequestBytes), "cuMemAlloc_v2");
    const bool has_reduced_free =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2") &&
        free_bytes == kQuotaBytes - kSuccessfulRequestBytes;
    const bool released_within_quota =
        !allocated_within_quota || check(cuMemFree_v2(successful_pointer), "cuMemFree_v2");
    const bool has_restored_free =
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2") &&
        free_bytes == kQuotaBytes;

    const CUresult invalid_allocation_result = cuMemAlloc_v2(nullptr, kSuccessfulRequestBytes);
    const bool propagated_driver_error = invalid_allocation_result != CUDA_SUCCESS;
    const CUresult unknown_free_result = cuMemFree_v2(static_cast<CUdeviceptr>(1));
    const bool unknown_free_preserved_usage =
        unknown_free_result != CUDA_SUCCESS &&
        check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2") &&
        free_bytes == kQuotaBytes;

    int memory_pools_supported = 0;
    const bool has_memory_pool_attribute =
        check(cuDeviceGetAttribute(&memory_pools_supported,
                                   CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device),
              "cuDeviceGetAttribute memory pools");
    bool async_checks = true;
    if (has_memory_pool_attribute && memory_pools_supported != 0) {
        CUstream stream = nullptr;
        async_checks &= check(cuStreamCreate(&stream, kStreamFlags), "cuStreamCreate");

        CUdeviceptr async_pointer = 0;
        const bool async_allocated =
            async_checks && check(cuMemAllocAsync(&async_pointer, kSuccessfulRequestBytes, stream),
                                  "cuMemAllocAsync");
        const bool async_usage_charged =
            async_allocated &&
            check(cuMemGetInfo_v2(&free_bytes, &total_bytes), "cuMemGetInfo_v2 async allocation") &&
            free_bytes == kQuotaBytes - kSuccessfulRequestBytes;
        const bool async_free_enqueued =
            async_allocated && check(cuMemFreeAsync(async_pointer, stream), "cuMemFreeAsync");
        const bool async_usage_retained = async_free_enqueued &&
                                          check(cuMemGetInfo_v2(&free_bytes, &total_bytes),
                                                "cuMemGetInfo_v2 pending async free") &&
                                          free_bytes == kQuotaBytes - kSuccessfulRequestBytes;
        const bool async_completed = async_free_enqueued &&
                                     check(cuStreamSynchronize(stream), "cuStreamSynchronize") &&
                                     check(cuMemGetInfo_v2(&free_bytes, &total_bytes),
                                           "cuMemGetInfo_v2 completed async free") &&
                                     free_bytes == kQuotaBytes;
        async_checks &= async_allocated && async_usage_charged && async_free_enqueued &&
                        async_usage_retained && async_completed;

        CUdeviceptr query_pointer = 0;
        const bool query_allocated =
            check(cuMemAllocAsync(&query_pointer, kSuccessfulRequestBytes, stream),
                  "cuMemAllocAsync query");
        const bool query_free_enqueued =
            query_allocated && check(cuMemFreeAsync(query_pointer, stream), "cuMemFreeAsync query");
        const CUresult query_result =
            query_free_enqueued ? cuStreamQuery(stream) : CUDA_ERROR_INVALID_VALUE;
        const bool query_observed =
            query_result == CUDA_SUCCESS || query_result == CUDA_ERROR_NOT_READY;
        const bool query_completed =
            query_observed &&
            (query_result == CUDA_SUCCESS ||
             check(cuStreamSynchronize(stream), "cuStreamSynchronize after query")) &&
            check(cuMemGetInfo_v2(&free_bytes, &total_bytes),
                  "cuMemGetInfo_v2 queried async free") &&
            free_bytes == kQuotaBytes;
        async_checks &= query_allocated && query_free_enqueued && query_completed;

        CUmemoryPool default_pool = nullptr;
        const bool pool_available =
            check(cuDeviceGetDefaultMemPool(&default_pool, device), "cuDeviceGetDefaultMemPool");
        CUdeviceptr pool_pointer = 0;
        const bool pool_allocated =
            pool_available && check(cuMemAllocFromPoolAsync(&pool_pointer, kSuccessfulRequestBytes,
                                                            default_pool, stream),
                                    "cuMemAllocFromPoolAsync");
        const bool pool_freed =
            pool_allocated && check(cuMemFreeAsync(pool_pointer, stream), "cuMemFreeAsync pool");
        const bool pool_completed =
            pool_freed && check(cuCtxSynchronize(), "cuCtxSynchronize pool");
        async_checks &= pool_available && pool_allocated && pool_freed && pool_completed;

        async_checks &= check(cuStreamDestroy(stream), "cuStreamDestroy");
    }

    CUdeviceptr device_pointer{};
    const CUresult allocation_result = cuMemAlloc_v2(&device_pointer, kRequestedBytes);
    const bool was_rejected = allocation_result == CUDA_ERROR_OUT_OF_MEMORY;
    const bool released_unexpected_allocation =
        allocation_result != CUDA_SUCCESS || check(cuMemFree_v2(device_pointer), "cuMemFree_v2");
    const bool destroyed_context = check(cuCtxDestroy(context), "cuCtxDestroy");

    if (!proc_address_call || !unsupported_proc_address || !has_expected_device_total ||
        !managed_allocated || !managed_released || !pitch_allocated || !has_pitched_usage ||
        !pitch_released || !has_memory_info || !has_expected_total || !has_expected_initial_free ||
        !allocated_within_quota || !has_reduced_free || !released_within_quota ||
        !has_restored_free || !propagated_driver_error || !unknown_free_preserved_usage ||
        !has_memory_pool_attribute || !async_checks || !was_rejected ||
        !released_unexpected_allocation || !destroyed_context) {
        std::cerr << "proc_address_call=" << proc_address_call
                  << " unsupported_proc_address=" << unsupported_proc_address
                  << " device_total=" << has_expected_device_total
                  << " managed_allocated=" << managed_allocated
                  << " managed_released=" << managed_released
                  << " pitch_allocated=" << pitch_allocated << " pitch_bytes=" << pitch_bytes
                  << " pitched_allocation_bytes=" << pitched_allocation_bytes
                  << " pitched_usage=" << has_pitched_usage << " pitch_released=" << pitch_released
                  << " memory_info=" << has_memory_info << " expected_total=" << has_expected_total
                  << " initial_free=" << has_expected_initial_free
                  << " allocated=" << allocated_within_quota << " reduced_free=" << has_reduced_free
                  << " released=" << released_within_quota << " restored_free=" << has_restored_free
                  << " propagated_error=" << propagated_driver_error
                  << " unknown_free=" << unknown_free_preserved_usage
                  << " memory_pools_attribute=" << has_memory_pool_attribute
                  << " async=" << async_checks << " rejected=" << was_rejected
                  << " released_unexpected=" << released_unexpected_allocation
                  << " destroyed=" << destroyed_context << '\n';
        std::cerr << "CUDA interceptor GPU test failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
