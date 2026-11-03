#pragma once

#include <string_view>

// Shared CPU/assembler/owned-session input. Every thread updates one output
// through local storage; guards and repeated non-idempotent updates are checked.
inline constexpr std::string_view kCompilerPtx = R"ptx(
.version 7.4
.target sm_70
.address_size 64
.file 1 "compiler fixture.cu"
.weak .entry add_once(.param .u64 output, .param .u32 guard)
.maxntid 32, 1, 1
.minnctapersm 2
.maxnreg 64
{
 .reg .b32 %r<9>;
 .reg .b64 %rd<6>;
 .local .align 16 .b8 scratch[16];
 .loc 1 10 0
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [guard];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4; add.u32 %r6, %r5, %r1;
 mul.wide.u32 %rd2, %r6, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r7, [%rd3]; mov.u32 %r8, %nctaid.x;
 mov.u64 %rd4, scratch;
 cvta.local.u64 %rd5, %rd4; cvta.to.local.u64 %rd4, %rd5;
 st.local.u32 [%rd4], %r5; st.local.u32 [%rd4+4], %r8;
 .loc 1 20 0
 {
  .reg .b32 %temporary;
  ld.local.u32 %temporary, [%rd4];
  add.u32 %r7, %r7, %temporary;
  {
   .reg .b32 %temporary;
   .loc 1 26 0
   ld.local.u32 %temporary, [%rd4+4];
   add.u32 %r7, %r7, %temporary;
  }
 }
 st.global.u32 [%rd3], %r7;
 {
  .loc 1 32 0
  ret;
 }
}
.section .debug_str {
 $debug_string: .b8 99, 111, 100, 101, 0
}
)ptx";
