#pragma once

#include <string_view>

// Public native layout is independently specified as offsets 0/8/16/48,
// sizes 4/8/32/4 and 52 total bytes. All eight aggregate words affect output;
// values[6] also checks logical grid X, and values[7] guards the final CTA tail.
// The read-modify-write is deliberately non-idempotent to detect replay/omission.
inline constexpr std::string_view kAggregateParameterText = R"ptx(.param .u32 prefix,
 .param .u64 output,
 .param .align 16 .b8 values[32],
 .param .u32 suffix)ptx";

inline constexpr std::string_view kAggregatePtx = R"ptx(
.version 7.4
.target sm_70
.address_size 64
.visible .entry aggregate_once(.param .u32 prefix,
 .param .u64 output,
 .param .align 16 .b8 values[32],
 .param .u32 suffix)
.maxntid 32, 1, 1
{
 .reg .b32 %r<20>;
 .reg .b64 %rd<3>;
 .reg .pred %p<2>;
 ld.param.u32 %r0, [prefix];
 ld.param.u32 %r1, [suffix];
 ld.param.u64 %rd0, [output];
 ld.param.u32 %r2, [values];
 ld.param.u32 %r3, [values+4];
 ld.param.u32 %r4, [values+8];
 ld.param.u32 %r5, [values+12];
 ld.param.u32 %r6, [values+16];
 ld.param.u32 %r7, [values+20];
 ld.param.u32 %r8, [values+24];
 ld.param.u32 %r9, [values+28];
 add.u32 %r10, %r2, %r3;
 add.u32 %r10, %r10, %r4;
 add.u32 %r10, %r10, %r5;
 add.u32 %r10, %r10, %r6;
 add.u32 %r10, %r10, %r7;
 add.u32 %r10, %r10, %r8;
 add.u32 %r10, %r10, %r9;
 mov.u32 %r11, %ctaid.x;
 mov.u32 %r12, %ntid.x;
 mov.u32 %r13, %tid.x;
 mad.lo.u32 %r14, %r11, %r12, %r13;
 setp.ge.u32 %p0, %r14, %r9;
 @%p0 bra aggregate_done;
 mov.u32 %r15, %nctaid.x;
 setp.ne.u32 %p1, %r15, %r8;
 selp.u32 %r18, 16777216, 0, %p1;
 mul.wide.u32 %rd1, %r14, 4;
 add.u64 %rd2, %rd0, %rd1;
 ld.global.u32 %r16, [%rd2];
 add.u32 %r17, %r14, %r10;
 add.u32 %r17, %r17, %r0;
 add.u32 %r17, %r17, %r1;
 add.u32 %r17, %r17, %r18;
 add.u32 %r16, %r16, %r17;
 st.global.u32 [%rd2], %r16;
aggregate_done:
 ret;
}
)ptx";
