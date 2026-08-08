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
        !was_rejected || !released_unexpected_allocation || !destroyed_context) {
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
                  << " rejected=" << was_rejected
                  << " released_unexpected=" << released_unexpected_allocation
                  << " destroyed=" << destroyed_context << '\n';
        std::cerr << "CUDA interceptor GPU test failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
