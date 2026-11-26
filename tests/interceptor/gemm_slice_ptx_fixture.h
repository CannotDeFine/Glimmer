#pragma once

#include <string>

// A numerical/queue ABI fixture, not a vendor binary rewrite. The optional
// original/mapped cubin pair supplies the independently qualified binary test.
inline std::string gemm_slice_ptx_fixture(bool is_mapped) {
    std::string source = R"ptx(
.version 8.0
.target sm_70
.address_size 64
.visible .entry gemm_test(.param .align 8 .b8 args[)ptx";
    source += is_mapped ? "184" : "176";
    source += R"ptx(]) {
 .reg .b32 %m, %n, %k, %lda, %ldb, %ldc, %ldd, %z, %x, %y, %i, %r, %c, %q;
 .reg .b64 %a, %b, %out, %in, %sa, %sb, %sc, %sd, %tmp, %aa, %bb, %cc, %dd;
 .reg .f32 %av, %bv, %cv, %sum, %alpha, %beta;
 .reg .pred %p;
 ld.param.u64 %a, [args]; ld.param.u64 %b, [args+8];
 ld.param.u64 %out, [args+16]; ld.param.u64 %in, [args+168];
 ld.param.u64 %sa, [args+56]; ld.param.u64 %sb, [args+64];
 ld.param.u64 %sc, [args+72]; ld.param.u64 %sd, [args+80];
 ld.param.u32 %lda, [args+88]; ld.param.u32 %ldb, [args+92];
 ld.param.u32 %ldc, [args+96]; ld.param.u32 %ldd, [args+100];
 ld.param.u32 %m, [args+104]; ld.param.u32 %n, [args+108];
 ld.param.u32 %k, [args+112]; ld.param.f32 %alpha, [args+152];
 ld.param.f32 %beta, [args+156];
)ptx";
    source += is_mapped ? " ld.param.u32 %z, [args+176];\n" : " mov.u32 %z, %ctaid.z;\n";
    source += R"ptx(
 cvt.u64.u32 %tmp, %z;
 mul.lo.u64 %sa, %sa, %tmp; mul.lo.u64 %sb, %sb, %tmp;
 mul.lo.u64 %sc, %sc, %tmp; mul.lo.u64 %sd, %sd, %tmp;
 shl.b64 %sa, %sa, 2; shl.b64 %sb, %sb, 2;
 shl.b64 %sc, %sc, 2; shl.b64 %sd, %sd, 2;
 add.u64 %a, %a, %sa; add.u64 %b, %b, %sb;
 add.u64 %out, %out, %sc; add.u64 %in, %in, %sd;
 mov.u32 %x, %ctaid.x; shl.b32 %x, %x, 7;
 mov.u32 %y, %ctaid.y; shl.b32 %y, %y, 5;
 mov.u32 %i, %tid.x;
ELEMENT:
 setp.ge.u32 %p, %i, 4096; @%p bra END;
 and.b32 %r, %i, 127; add.u32 %r, %r, %x;
 shr.u32 %c, %i, 7; add.u32 %c, %c, %y;
 setp.ge.u32 %p, %r, %m; @%p bra NEXT;
 setp.ge.u32 %p, %c, %n; @%p bra NEXT;
 mov.u32 %q, 0; mov.f32 %sum, 0f00000000;
REDUCE:
 setp.ge.u32 %p, %q, %k; @%p bra STORE;
 mad.lo.u32 %z, %q, %lda, %r; mul.wide.u32 %tmp, %z, 4;
 add.u64 %aa, %a, %tmp;
 mad.lo.u32 %z, %q, %ldb, %c; mul.wide.u32 %tmp, %z, 4;
 add.u64 %bb, %b, %tmp;
 ld.global.f32 %av, [%aa]; ld.global.f32 %bv, [%bb];
 fma.rn.f32 %sum, %av, %bv, %sum;
 add.u32 %q, %q, 1; bra REDUCE;
STORE:
 mad.lo.u32 %z, %c, %ldc, %r; mul.wide.u32 %tmp, %z, 4;
 add.u64 %cc, %out, %tmp;
 mad.lo.u32 %z, %c, %ldd, %r; mul.wide.u32 %tmp, %z, 4;
 add.u64 %dd, %in, %tmp;
 ld.global.f32 %cv, [%dd]; mul.rn.f32 %cv, %cv, %beta;
 fma.rn.f32 %sum, %alpha, %sum, %cv; st.global.f32 [%cc], %sum;
NEXT:
 add.u32 %i, %i, 256; bra ELEMENT;
END:
 ret;
}
)ptx";
    return source;
}
