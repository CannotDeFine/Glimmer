#pragma once

#include <cuda.h>
#include <cstddef>

#ifdef cuMemcpy
#undef cuMemcpy
#endif
#ifdef cuMemcpyAsync
#undef cuMemcpyAsync
#endif

#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
#ifdef cuMemcpyHtoD
#undef cuMemcpyHtoD
#endif
#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
#ifdef cuMemcpyDtoDAsync
#undef cuMemcpyDtoDAsync
#endif
#ifdef cuMemcpyDtoD
#undef cuMemcpyDtoD
#endif
#ifdef cuMemHostAlloc
#undef cuMemHostAlloc
#endif
#ifdef cuMemAllocHost
#undef cuMemAllocHost
#endif
#ifdef cuMemFreeHost
#undef cuMemFreeHost
#endif
#ifdef cuMemHostRegister
#undef cuMemHostRegister
#endif
#ifdef cuMemHostUnregister
#undef cuMemHostUnregister
#endif
#ifdef cuMemHostGetDevicePointer
#undef cuMemHostGetDevicePointer
#endif
#ifdef cuStreamCreate
#undef cuStreamCreate
#endif
#ifdef cuStreamCreateWithPriority
#undef cuStreamCreateWithPriority
#endif
#ifdef cuStreamGetFlags
#undef cuStreamGetFlags
#endif
#ifdef cuStreamGetPriority
#undef cuStreamGetPriority
#endif
#ifdef cuMemsetD8
#undef cuMemsetD8
#endif
#ifdef cuMemsetD8Async
#undef cuMemsetD8Async
#endif
#ifdef cuMemsetD16
#undef cuMemsetD16
#endif
#ifdef cuMemsetD16Async
#undef cuMemsetD16Async
#endif
#ifdef cuMemsetD32
#undef cuMemsetD32
#endif
#ifdef cuMemsetD32Async
#undef cuMemsetD32Async
#endif
#ifdef cuStreamBeginCapture
#undef cuStreamBeginCapture
#endif
#ifdef cuStreamEndCapture
#undef cuStreamEndCapture
#endif

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void* host,
                                                 std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpy(CUdeviceptr destination, CUdeviceptr source,
                                     std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr destination, CUdeviceptr source,
                                               std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr destination, const void* host,
                                                      std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination, const void* host,
                                            std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr destination, const void* host,
                                                 std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* host, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void* host, CUdeviceptr source,
                                                      std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2(void* host, CUdeviceptr source,
                                            std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void* host, CUdeviceptr source,
                                                 std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr destination, CUdeviceptr source,
                                                      std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr destination, CUdeviceptr source,
                                            std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemHostAlloc(void** pointer, std::size_t memory_bytes,
                                           unsigned int flags);
extern "C" CUresult CUDAAPI cuMemAllocHost_v2(void** pointer, std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemFreeHost(void* pointer);
extern "C" CUresult CUDAAPI cuMemHostRegister_v2(void* pointer, std::size_t memory_bytes,
                                                 unsigned int flags);
extern "C" CUresult CUDAAPI cuMemHostUnregister(void* pointer);
extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* device_pointer, void* pointer,
                                                         unsigned int flags);
extern "C" CUresult CUDAAPI cuStreamCreate(CUstream* stream, unsigned int flags);
extern "C" CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* stream, unsigned int flags,
                                                       int priority);
extern "C" CUresult CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int* flags);
extern "C" CUresult CUDAAPI cuStreamGetPriority(CUstream stream, int* priority);
extern "C" CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr pointer, unsigned char value,
                                          std::size_t count);
extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr pointer, unsigned char value,
                                               std::size_t count);
extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr pointer, unsigned char value,
                                            std::size_t count, CUstream stream);
extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr pointer, unsigned char value,
                                                 std::size_t count, CUstream stream);
extern "C" CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr pointer, unsigned short value,
                                           std::size_t count);
extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr pointer, unsigned short value,
                                                std::size_t count);
extern "C" CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr pointer, unsigned short value,
                                             std::size_t count, CUstream stream);
extern "C" CUresult CUDAAPI cuMemsetD16Async_ptsz(CUdeviceptr pointer, unsigned short value,
                                                  std::size_t count, CUstream stream);
extern "C" CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr pointer, unsigned int value,
                                           std::size_t count);
extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr pointer, unsigned int value,
                                                std::size_t count);
extern "C" CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr pointer, unsigned int value,
                                             std::size_t count, CUstream stream);
extern "C" CUresult CUDAAPI cuMemsetD32Async_ptsz(CUdeviceptr pointer, unsigned int value,
                                                  std::size_t count, CUstream stream);
extern "C" CUresult CUDAAPI cuStreamBeginCapture(CUstream stream);
extern "C" CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream stream);
extern "C" CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode);
extern "C" CUresult CUDAAPI cuStreamBeginCapture_v2_ptsz(CUstream stream, CUstreamCaptureMode mode);
extern "C" CUresult CUDAAPI cuStreamEndCapture(CUstream stream, CUgraph* graph);
extern "C" CUresult CUDAAPI cuStreamEndCapture_ptsz(CUstream stream, CUgraph* graph);
