#pragma once

// Each complete logical CTA adds index + gridDim.x + iterations * (lane ^ 1)
// to every output. Zero-trip and repeated full-CTA/warp participation both have
// observable semantics. Use a one-warp, 32-thread block for numerical execution.
inline constexpr char kUniformLoopPtx[] = R"ptx(.version 7.4
.target sm_70
.address_size 64
.visible .entry uniform_loop(
    .param .u64 output,
    .param .u32 iterations
)
.reqntid 32
{
    .reg .pred %p<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [output];
    ld.param.u32 %r0, [iterations];
    mov.u32 %r1, 0;
    mov.u32 %r2, 0;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %nctaid.x;
    mad.lo.u32 %r7, %r4, %r5, %r3;
    setp.eq.u32 %p0, %r0, 0;
    @%p0 bra done;
again:
    bar.sync 0;
    shfl.sync.bfly.b32 %r8, %r3, 1, 31, -1;
    add.u32 %r2, %r2, %r8;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p1, %r1, %r0;
    @%p1 bra again;
done:
    add.u32 %r9, %r7, %r6;
    add.u32 %r10, %r9, %r2;
    mul.wide.u32 %rd1, %r7, 4;
    add.u64 %rd2, %rd0, %rd1;
    atom.global.add.u32 %r11, [%rd2], %r10;
    ret;
}
)ptx";

// All 64 lanes execute both barriers, but even/odd lanes take different store
// paths first. Each iteration adds twice the lane's 3/7 value plus its xor-1
// peer's 7/3 value: 13 for even lanes, 17 for odd lanes. The final atomic adds
// index + gridDim.x + iterations * that value, preserving an exact-once oracle.
inline constexpr char kReconvergentLoopPtx[] = R"ptx(.version 7.4
.target sm_70
.address_size 64
.visible .entry reconvergent_loop(
    .param .u64 output,
    .param .u32 iterations
)
.reqntid 64
{
    .reg .pred %p<3>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<7>;
    .shared .align 4 .b8 storage[256];
    ld.param.u64 %rd0, [output];
    ld.param.u32 %r0, [iterations];
    mov.u32 %r1, 0;
    mov.u32 %r2, 0;
    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %nctaid.x;
    mad.lo.u32 %r7, %r4, %r5, %r3;
    and.b32 %r8, %r3, 1;
    setp.eq.u32 %p2, %r8, 0;
    mov.u64 %rd3, storage;
    mul.wide.u32 %rd4, %r3, 4;
    add.u64 %rd4, %rd3, %rd4;
    xor.b32 %r9, %r3, 1;
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd5, %rd3, %rd5;
    setp.eq.u32 %p0, %r0, 0;
    @%p0 bra done;
again:
    @!%p2 bra odd;
    mov.u32 %r10, 3;
    st.shared.u32 [%rd4], %r10;
    bra joined;
odd:
    mov.u32 %r10, 7;
    st.shared.u32 [%rd4], %r10;
joined:
    bar.sync 0;
    ld.shared.u32 %r11, [%rd5];
    mad.lo.u32 %r2, %r10, 2, %r2;
    add.u32 %r2, %r2, %r11;
    bar.sync 0;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p1, %r1, %r0;
    @%p1 bra again;
done:
    add.u32 %r12, %r7, %r6;
    add.u32 %r13, %r12, %r2;
    mul.wide.u32 %rd1, %r7, 4;
    add.u64 %rd2, %rd0, %rd1;
    atom.global.add.u32 %r14, [%rd2], %r13;
    ret;
}
)ptx";
