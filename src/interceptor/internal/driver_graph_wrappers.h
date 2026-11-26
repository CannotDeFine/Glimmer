#pragma once

#include <cuda.h>
#include <cstddef>

#ifdef cuGraphInstantiate
#undef cuGraphInstantiate
#endif
#ifdef cuGraphInstantiateWithParams
#undef cuGraphInstantiateWithParams
#endif
#ifdef cuGraphExecUpdate
#undef cuGraphExecUpdate
#endif

extern "C" CUresult CUDAAPI cuGraphInstantiate(CUgraphExec*, CUgraph, CUgraphNode*, char*,
                                               std::size_t);
extern "C" CUresult CUDAAPI cuGraphInstantiate_v2(CUgraphExec*, CUgraph, CUgraphNode*, char*,
                                                  std::size_t);
extern "C" CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec*, CUgraph, unsigned long long);
extern "C" CUresult CUDAAPI cuGraphInstantiateWithParams(CUgraphExec*, CUgraph,
                                                         CUDA_GRAPH_INSTANTIATE_PARAMS*);
extern "C" CUresult CUDAAPI cuGraphInstantiateWithParams_ptsz(CUgraphExec*, CUgraph,
                                                              CUDA_GRAPH_INSTANTIATE_PARAMS*);
extern "C" CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec, CUgraph, CUgraphNode*,
                                              CUgraphExecUpdateResult*);
extern "C" CUresult CUDAAPI cuGraphExecUpdate_v2(CUgraphExec, CUgraph,
                                                 CUgraphExecUpdateResultInfo*);
