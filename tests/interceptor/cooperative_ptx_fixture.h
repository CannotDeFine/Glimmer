#pragma once

// A shared-memory block publishes progress before doing bounded work. Output
// holds one exactly-once counter per logical thread across retreat/resume.
inline constexpr char kCooperativePtx[] = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry cooperative(.param .u64 output, .param .u64 marker, .param .u32 work)
{
    .reg .pred %p<4>;
    .reg .u32 %r<24>;
    .reg .u64 %rd<8>;
    .shared .align 4 .u32 shared_value;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [marker];
    ld.param.u32 %r0, [work];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %tid.y;
    mov.u32 %r3, %tid.z;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ntid.y;
    mov.u32 %r6, %ntid.z;
    mad.lo.u32 %r7, %r3, %r5, %r2;
    mad.lo.u32 %r8, %r7, %r4, %r1;
    mov.u32 %r9, %ctaid.x;
    mov.u32 %r10, %ctaid.y;
    mov.u32 %r11, %ctaid.z;
    mov.u32 %r12, %nctaid.x;
    mov.u32 %r13, %nctaid.y;
    mov.u32 %r14, %nctaid.z;
    setp.ne.u32 %p0, %r8, 0;
    @%p0 bra initialized;
    mad.lo.u32 %r15, %r11, %r13, %r10;
    mad.lo.u32 %r15, %r15, %r12, %r9;
    st.shared.u32 [shared_value], %r15;
    st.global.u32 [%rd1], 1;
    membar.sys;
initialized:
    bar.sync 0;
    ld.shared.u32 %r15, [shared_value];
    mul.lo.u32 %r16, %r4, %r5;
    mul.lo.u32 %r16, %r16, %r6;
    mad.lo.u32 %r17, %r15, %r16, %r8;
    mul.wide.u32 %rd2, %r17, 4;
    add.u64 %rd3, %rd0, %rd2;
    mov.u32 %r18, 0;
delay:
    setp.ge.u32 %p1, %r18, %r0;
    @%p1 bra completed;
    ld.volatile.global.u32 %r20, [%rd3];
    add.u32 %r18, %r18, 1;
    bra delay;
completed:
    ld.global.u32 %r19, [%rd3];
    add.u32 %r19, %r19, 1;
    st.global.u32 [%rd3], %r19;
    ret;
}
.visible .entry priority_probe(.param .u64 marker)
{
    .reg .u64 %rd;
    ld.param.u64 %rd, [marker];
    st.global.u32 [%rd], 2;
    membar.sys;
    ret;
}
)ptx";

inline constexpr char kNativeOncePtx[] = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry native_once(.param .u64 marker)
{
    .reg .u64 %rd;
    .reg .u32 %r;
    ld.param.u64 %rd, [marker];
    ld.global.u32 %r, [%rd];
    add.u32 %r, %r, 1;
    st.global.u32 [%rd], %r;
    membar.sys;
    ret;
}
)ptx";
