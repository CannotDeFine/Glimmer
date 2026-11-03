#pragma once

#include "glimmer/transform/ptx_transform.h"

// Compatibility include; transformation owns no interceptor/SDK state.
namespace glimmer::interceptor {
using transform::kMaxPersistentPtxBytes;
using transform::PersistentPtx;
using transform::PtxExecutionKind;
using transform::PtxKernelVariant;
using transform::PtxParameterLayout;
using transform::transform_persistent_ptx;
using transform::transform_transparent_ptx;
}  // namespace glimmer::interceptor
