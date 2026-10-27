@ void arm_kernel4x4_vec(const float *ap, const float *bp, float *c, int ldc_bytes, int kc)
@
@ c[0..3][0..3] (row stride ldc_bytes) += sum_k ap[k][0..3]^T * bp[k][0..3], kc even.
@ VFP11 short-vector mode (FPSCR.LEN = 4): `vmla.f32 s16, s8, s0` updates
@ s16..s19 += s8..s11 * s0 (an operand in s0-s7 is a scalar), so one
@ instruction does 4 multiply-accumulates.  Rows of C live in s16-s19,
@ s20-s23, s24-s27, s28-s31; a in s0-s3 / s4-s7 and b in s8-s11 / s12-s15
@ alternate (two k per loop iteration, loads one step ahead).
@ The loop reads one k step (16 bytes) past the end of both panels.
@ (Measured on the Pi Zero: moving each load pair one vmla later, to avoid
@ waiting on the vmla still reading those registers, and prefetching the next
@ C tiles made no difference at 512^3 and was slower at 1024^3.)

#ifndef PLD_AHEAD
#define PLD_AHEAD 128
#endif

.text
.align 2
.globl arm_kernel4x4_vec
.type arm_kernel4x4_vec, %function
arm_kernel4x4_vec:
    push    {r4-r7, lr}
    vpush   {s16-s31}
    ldr     r4, [sp, #84]           @ kc (5th argument: above 5 pushed regs + 16 VFP regs)
    mov     r5, r2
    vldmia  r5, {s16-s19}
    add     r5, r5, r3
    vldmia  r5, {s20-s23}
    add     r5, r5, r3
    vldmia  r5, {s24-s27}
    add     r5, r5, r3
    vldmia  r5, {s28-s31}
    vmrs    r6, fpscr
    bic     r7, r6, #0x370000       @ LEN and STRIDE
    orr     r7, r7, #0x30000        @ LEN = 4, STRIDE = 1
    vmsr    fpscr, r7
    vldmia  r0!, {s0-s3}
    vldmia  r1!, {s8-s11}
1:
    pld     [r0, #PLD_AHEAD]        @ the A panel streams from DRAM: prefetch it
    vldmia  r0!, {s4-s7}
    vldmia  r1!, {s12-s15}
    vmla.f32 s16, s8, s0
    vmla.f32 s20, s8, s1
    vmla.f32 s24, s8, s2
    vmla.f32 s28, s8, s3
    vldmia  r0!, {s0-s3}
    vldmia  r1!, {s8-s11}
    vmla.f32 s16, s12, s4
    vmla.f32 s20, s12, s5
    vmla.f32 s24, s12, s6
    vmla.f32 s28, s12, s7
    subs    r4, r4, #2
    bne     1b
    vmsr    fpscr, r6               @ back to scalar mode
    mov     r5, r2
    vstmia  r5, {s16-s19}
    add     r5, r5, r3
    vstmia  r5, {s20-s23}
    add     r5, r5, r3
    vstmia  r5, {s24-s27}
    add     r5, r5, r3
    vstmia  r5, {s28-s31}
    vpop    {s16-s31}
    pop     {r4-r7, pc}
.size arm_kernel4x4_vec, .-arm_kernel4x4_vec
