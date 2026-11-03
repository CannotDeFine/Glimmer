#pragma once

#include <string_view>

// Logical grid reads occur in a nested callee, not just in the kernel.
inline constexpr std::string_view kDeviceCallPtx = R"ptx(
.version 7.4
.target sm_70
.address_size 64
.func empty_helper { ret; }
.func (.reg .u32 %result) register_helper(.reg .u32 %input) {
 mov.u32 %result, %input;
 ret.uni;
}
.func (.param .b32 result) leaf(.param .b32 input) {
 .reg .u32 %r<10>;
 ld.param.u32 %r0, [input];
 mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x;
 mad.lo.u32 %r3, %r1, %r2, %r0;
 mov.u32 %r4, %nctaid.x; add.u32 %r3, %r3, %r4;
 mov.u32 %r5, %ctaid.y; add.u32 %r3, %r3, %r5;
 mov.u32 %r6, %ctaid.z; add.u32 %r3, %r3, %r6;
 mov.u32 %r7, %nctaid.y; add.u32 %r3, %r3, %r7;
 mov.u32 %r8, %nctaid.z; add.u32 %r3, %r3, %r8;
 sub.u32 %r3, %r3, 2;
 st.param.b32 [result], %r3;
 ret;
}
.func (.param .b32 result) middle(.param .b32 input) {
 .reg .u32 %value;
 .param .b32 argument;
 .param .b32 returned;
 ld.param.u32 %value, [input];
 call empty_helper;
 call.uni (%value), register_helper, (%value);
 st.param.b32 [argument], %value;
 call.uni (returned), leaf, (argument);
 ld.param.b32 %value, [returned];
 st.param.b32 [result], %value;
 ret;
}
.visible .entry add_once(.param .u64 output, .param .u32 guard)
.maxntid 32, 1, 1
{
 .reg .b32 %r<9>;
 .reg .b64 %rd<4>;
 .param .b32 argument;
 .param .b32 returned;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [guard];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4; add.u32 %r6, %r5, %r1;
 mul.wide.u32 %rd2, %r6, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r7, [%rd3];
 st.param.b32 [argument], %r4;
 call.uni (returned), middle, (argument);
 ld.param.b32 %r8, [returned];
 add.u32 %r7, %r7, %r8;
 st.global.u32 [%rd3], %r7;
 ret;
}
)ptx";
