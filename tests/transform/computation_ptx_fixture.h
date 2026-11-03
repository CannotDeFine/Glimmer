#pragma once

#include <string_view>

// Two disjoint atomic updates must add index + logical grid exactly once.
// Additional scalar instructions are retained for offline native assembly.
inline constexpr std::string_view kAtomicPtx = R"ptx(
.version 7.4
.target sm_70
.address_size 64
.visible .entry add_once(.param .u64 output, .param .u32 guard)
.maxntid 32, 1, 1
{
 .reg .b32 %r<10>;
 .reg .b64 %rd<4>;
 .reg .f32 %f<3>;
 .reg .pred %p;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [guard];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4; add.u32 %r6, %r5, %r1;
 mul.wide.u32 %rd2, %r6, 4; add.u64 %rd3, %rd1, %rd2;
 mov.u32 %r8, %nctaid.x;
 atom.relaxed.gpu.global.add.u32 %r7, [%rd3], %r5;
 red.release.gpu.global.add.u32 [%rd3], %r8;
 atom.global.acquire.sys.add.u32 %r7, [%rd3], 0;
 atom.acquire.global.add.u32 %r7, [%rd3], 0;
 ld.acquire.gpu.global.u32 %r7, [%rd3];
 st.release.gpu.global.u32 [%rd3], %r7;
 set.eq.u32.u32 %r9, %r5, %r5;
 slct.u32.s32 %r9, %r5, %r5, 0;
 add.cc.u32 %r9, %r5, 1; addc.u32 %r9, %r9, 0;
 sub.cc.u32 %r9, %r5, 1; subc.u32 %r9, %r9, 0;
 mad.lo.cc.u32 %r9, %r5, 2, 1; madc.lo.u32 %r9, %r9, 1, 0;
 bfi.b32 %r9, %r5, %r5, 0, 8; bfe.u32 %r9, %r5, 0, 8;
 bfind.u32 %r9, %r5; brev.b32 %r9, %r5; clz.b32 %r9, %r5;
 popc.b32 %r9, %r5; prmt.b32 %r9, %r5, %r5, 0;
 lop3.b32 %r9, %r5, %r5, %r5, 0x96;
 shf.l.wrap.b32 %r9, %r5, %r5, 1;
 sad.u32 %r9, %r5, %r5, 0; cnot.b32 %r9, %r5;
 mov.f32 %f0, 0f3f800000;
 rcp.approx.ftz.f32 %f1, %f0; sqrt.rn.f32 %f1, %f0;
 rsqrt.approx.ftz.f32 %f1, %f0; sin.approx.ftz.f32 %f1, %f0;
 cos.approx.ftz.f32 %f1, %f0; lg2.approx.ftz.f32 %f1, %f0;
 ex2.approx.ftz.f32 %f1, %f0; copysign.f32 %f1, %f0, %f0;
 testp.finite.f32 %p, %f0;
 ret;
}
)ptx";

// A full 32-lane CTA reduction, broadcast and votes contribute zero on success.
// Missing/repeated shared atomics or incorrect lane participation alter output.
inline constexpr std::string_view kWarpPtx = R"ptx(
.version 7.4
.target sm_70
.address_size 64
.func (.param .b32 result) warp_value(.param .b32 input) {
 .reg .b32 %r<8>;
 .reg .b64 %wide;
 .reg .pred %p<2>;
 ld.param.b32 %r0, [input];
 mov.u32 %r1, %laneid;
 mov.u32 %r2, %lanemask_lt;
 activemask.b32 %r3;
 and.b32 %r4, %r1, 1; setp.eq.u32 %p0, %r4, 0;
 vote.sync.ballot.b32 %r4, %p0, -1;
 vote.sync.all.pred %p1, %p0, -1;
 vote.sync.any.pred %p1, %p0, -1;
 vote.sync.uni.pred %p1, %p0, -1;
 popc.b32 %r4, %r4; sub.u32 %r4, %r4, 16;
 mov.u32 %r5, %ctaid.x;
 match.any.sync.b32 %r5, %r5, -1;
 popc.b32 %r5, %r5; sub.u32 %r5, %r5, 32;
 shfl.sync.idx.b32 %r6|%p1, %r1, 0, 31, -1;
 shfl.sync.up.b32 %r7, %r1, 1, 0, -1;
 shfl.sync.down.b32 %r7, %r1, 1, 31, -1;
 shfl.sync.bfly.b32 %r7, %r1, 1, 31, -1;
 mov.b64 %wide, 0;
 match.any.sync.b64 %r7, %wide, -1;
 match.all.sync.b32 %r7|%p1, %r6, -1;
 match.all.sync.b64 %r7|%p1, %wide, -1;
 add.u32 %r0, %r0, %r4; add.u32 %r0, %r0, %r5; add.u32 %r0, %r0, %r6;
 st.param.b32 [result], %r0;
 ret;
}
.visible .entry add_once(.param .u64 output, .param .u32 guard)
.reqntid 32, 1, 1
{
 .reg .b32 %r<12>;
 .reg .b64 %rd<4>;
 .reg .pred %p;
 .shared .u32 total;
 .param .b32 argument;
 .param .b32 returned;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [guard];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 setp.eq.u32 %p, %r4, 0; @%p st.shared.u32 [total], 0;
 bar.sync 0;
 add.u32 %r9, %r4, 1;
 atom.shared.add.u32 %r10, [total], %r9;
 red.relaxed.cta.shared.add.u32 [total], 0;
 cvta.shared.u64 %rd0, total;
 atom.add.u32 %r11, [%rd0], 0;
 bar.warp.sync -1;
 ld.shared.u32 %r10, [total]; sub.u32 %r10, %r10, 528;
 mad.lo.u32 %r5, %r2, %r3, %r4; add.u32 %r6, %r5, %r1;
 mul.wide.u32 %rd2, %r6, 4; add.u64 %rd3, %rd1, %rd2;
 mov.u32 %r8, %nctaid.x; add.u32 %r5, %r5, %r8; add.u32 %r5, %r5, %r10;
 st.param.b32 [argument], %r5;
 call.uni (returned), warp_value, (argument);
 ld.param.b32 %r5, [returned];
 fence.acq_rel.gpu;
 atom.global.add.u32 %r7, [%rd3], %r5;
 ret;
}
)ptx";
