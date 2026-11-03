#pragma once

inline constexpr char kPersistentTestPtx[] = R"ptx(
// The application uses ordinary PTX and has no Glimmer task interface.
.version 7.0
.target sm_50
.address_size 64

.visible .entry write_indices(
    .param .u64 output,
    .param .u32 element_count
)
{
    .reg .b32 %r<7>;
    .reg .b64 %rd<4>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [element_count];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r2, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 ret;
    mov.u32 %r5, %nctaid.x;
    add.u32 %r6, %r2, %r5;
    mul.wide.u32 %rd2, %r2, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r6;
    ret;
}

.visible .entry native_barrier(.param .u64 output)
{
    .reg .u64 %pointer, %offset;
    .reg .u32 %index, %thread;
    .reg .pred %leader;
    ld.param.u64 %pointer, [output];
    mov.u32 %index, %ctaid.x;
    mov.u32 %thread, %tid.x;
    mul.wide.u32 %offset, %index, 4;
    add.u64 %pointer, %pointer, %offset;
    setp.eq.u32 %leader, %thread, 0;
    bar.sync 0;
    @%leader st.global.u32 [%pointer], 17;
    ret;
}
)ptx";
