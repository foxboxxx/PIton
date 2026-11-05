@ VFP11 short-vector elementwise kernels: the strongest CPU baseline for the
@ memory-bound primitives (labelled `arm-vfp` in the benchmark output).
@
@ FPSCR.LEN = 4, STRIDE = 1 makes one arithmetic instruction operate on four
@ consecutive registers. s0-s7 is the scalar bank; s8-s15, s16-s23 and s24-s31
@ are vector banks, and a vector operation must not cross a bank, so each
@ routine works on eight floats per iteration as two length-4 operations.
@ `vldmia`/`vstmia` move eight floats per instruction, which is what actually
@ matters here: these loops are bounded by SDRAM, not by the FPU.
@
@ Every routine requires n > 0 and n divisible by 8 (main.c picks such sizes).
@ Pointer/int arguments in r0-r3. Float arguments and results are in s0 under the
@ hard-float ABI (__ARM_PCS_VFP); under softfp (libpi's -mfloat-abi=softfp) a fifth
@ argument is on the stack and a float result goes back in r0.

.text
.align 2

@ Set FPSCR.LEN = 4, STRIDE = 1, keeping the old value in r12.
.macro VEC_ON
    vmrs    r12, fpscr
    bic     r4, r12, #0x370000      @ clear LEN and STRIDE
    orr     r4, r4, #0x30000        @ LEN = 4 (field 3 -> 4 registers), STRIDE = 1
    vmsr    fpscr, r4
.endm

.macro VEC_OFF
    vmsr    fpscr, r12
.endm

@ void arm_copy_vfp(const float *x, float *out, int n)
.globl arm_copy_vfp
.type arm_copy_vfp, %function
arm_copy_vfp:
    push    {r4, lr}
    vpush   {s16-s23}
1:
    pld     [r0, #128]
    vldmia  r0!, {s16-s23}
    vstmia  r1!, {s16-s23}
    subs    r2, r2, #8
    bne     1b
    vpop    {s16-s23}
    pop     {r4, pc}
.size arm_copy_vfp, .-arm_copy_vfp

@ void arm_add_vfp(const float *x, const float *y, float *out, int n)
.globl arm_add_vfp
.type arm_add_vfp, %function
arm_add_vfp:
    push    {r4, lr}
    vpush   {s16-s31}
    VEC_ON
1:
    pld     [r0, #128]
    pld     [r1, #128]
    vldmia  r0!, {s8-s15}
    vldmia  r1!, {s16-s23}
    vadd.f32 s24, s8, s16           @ s24-s27 = s8-s11 + s16-s19
    vadd.f32 s28, s12, s20          @ s28-s31 = s12-s15 + s20-s23
    vstmia  r2!, {s24-s31}
    subs    r3, r3, #8
    bne     1b
    VEC_OFF
    vpop    {s16-s31}
    pop     {r4, pc}
.size arm_add_vfp, .-arm_add_vfp

@ void arm_mul_vfp(const float *x, const float *y, float *out, int n)
.globl arm_mul_vfp
.type arm_mul_vfp, %function
arm_mul_vfp:
    push    {r4, lr}
    vpush   {s16-s31}
    VEC_ON
1:
    pld     [r0, #128]
    pld     [r1, #128]
    vldmia  r0!, {s8-s15}
    vldmia  r1!, {s16-s23}
    vmul.f32 s24, s8, s16
    vmul.f32 s28, s12, s20
    vstmia  r2!, {s24-s31}
    subs    r3, r3, #8
    bne     1b
    VEC_OFF
    vpop    {s16-s31}
    pop     {r4, pc}
.size arm_mul_vfp, .-arm_mul_vfp

@ void arm_fma_vfp(const float *x, const float *y, float *out, int n, float alpha)
@ out = x * alpha + y.  alpha is used from s0, in the scalar bank.
.globl arm_fma_vfp
.type arm_fma_vfp, %function
arm_fma_vfp:
    push    {r4, lr}
#ifndef __ARM_PCS_VFP
    vldr    s0, [sp, #8]            @ softfp: alpha is the stacked fifth argument
#endif
    vpush   {s16-s31}
    VEC_ON
1:
    pld     [r0, #128]
    pld     [r1, #128]
    vldmia  r1!, {s24-s31}          @ y into the accumulator registers
    vldmia  r0!, {s8-s15}           @ x
    vmla.f32 s24, s8, s0            @ s24-s27 += s8-s11 * alpha (s0 is a scalar)
    vmla.f32 s28, s12, s0
    vstmia  r2!, {s24-s31}
    subs    r3, r3, #8
    bne     1b
    VEC_OFF
    vpop    {s16-s31}
    pop     {r4, pc}
.size arm_fma_vfp, .-arm_fma_vfp

@ float arm_sum_vfp(const float *x, int n)
@ Four independent accumulators, summed at the end: a single dependent chain
@ would be bounded by the ~8-cycle VFP11 add latency instead of throughput.
@ The four partial sums are added in a fixed order, so the result is
@ reproducible but not identical to a sequential sum.
.globl arm_sum_vfp
.type arm_sum_vfp, %function
arm_sum_vfp:
    push    {r4, lr}
    vpush   {s16-s31}
    VEC_ON
    vldmia  r0!, {s24-s27}          @ first four elements seed the accumulators
    sub     r1, r1, #4
    cmp     r1, #0
    beq     2f
1:
    pld     [r0, #128]
    vldmia  r0!, {s8-s11}
    vadd.f32 s24, s24, s8           @ s24-s27 += s8-s11, four lanes at once
    subs    r1, r1, #4
    bne     1b
2:
    VEC_OFF
    vadd.f32 s24, s24, s25
    vadd.f32 s26, s26, s27
    vadd.f32 s0, s24, s26
#ifndef __ARM_PCS_VFP
    vmov    r0, s0                  @ softfp: a float result is returned in r0
#endif
    vpop    {s16-s31}
    pop     {r4, pc}
.size arm_sum_vfp, .-arm_sum_vfp
