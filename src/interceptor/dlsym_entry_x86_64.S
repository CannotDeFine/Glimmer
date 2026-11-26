// Linux x86-64 SysV entry. Forwarding must preserve libc's actual caller, not
// merely pass equivalent arguments from a different DSO's C++ stack frame.
    .text
    .globl dlsym
    .type dlsym, @function
    .hidden glimmer_dispatch_dlsym
dlsym:
    .cfi_startproc
    endbr64
    // Entry RSP is 8 mod 16. Reserve aligned call space and save arguments.
    subq $40, %rsp
    .cfi_adjust_cfa_offset 40
    movq %rdi, 0(%rsp)
    movq %rsi, 8(%rsp)
    movq 40(%rsp), %rdx
    leaq 16(%rsp), %rcx
    call glimmer_dispatch_dlsym
    // RAX is the selected result; the stack output is an optional native target.
    movq 16(%rsp), %r11
    movq 0(%rsp), %rdi
    movq 8(%rsp), %rsi
    addq $40, %rsp
    .cfi_adjust_cfa_offset -40
    testq %r11, %r11
    je .Lselected_symbol
    jmp *%r11
.Lselected_symbol:
    ret
    .cfi_endproc
    .size dlsym, .-dlsym
    .section .note.GNU-stack,"",@progbits
