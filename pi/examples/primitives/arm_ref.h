// CPU baselines for every benchmarked Triton primitive.
//
// The build is freestanding (-nostdlib, no libgcc), so there is no libm and no
// __aeabi_idiv: exp, log, sqrt, rsqrt, reciprocal and integer division are all
// implemented here. Nothing in this file may use `/` or `%` on a runtime
// integer, or the link fails with an undefined reference.
//
// Baseline labels used in the benchmark output:
//
//   arm-naive    plain scalar C, the obvious implementation, -O3, caches on
//   arm-unroll   same arithmetic, 4x unrolled with prefetch
//   arm-vfp      uses a VFP instruction or short-vector assembly that the
//                naive version does not (vsqrt, vdiv, vldmia/vmla)
//   arm-fast     deliberately lower precision in exchange for speed (bit-trick
//                seeds, short polynomials, fewer Newton steps); the measured
//                error is reported per row and runs to ~1% for log
//   arm-blocked  cache-blocked / packed-panel version
//   qpu          the PIton-compiled Triton kernel on all 12 QPUs
//
// Scalar helpers are exposed so the fused benchmarks (softmax) can share them.
#ifndef ARM_REF_H
#define ARM_REF_H
#include <stdint.h>

// ---- scalar transcendentals ------------------------------------------------
// *_acc: accurate (<= a few ulp).  *_fast: cheap and approximate; see the
// arm-fast note above.  For reference, PIton's own exp/log on the QPU measure
// accurate to ~0.2 ppm, so these are not matched to a low-precision unit.
float ref_exp_acc(float x);
float ref_exp_fast(float x);
float ref_log_acc(float x);
float ref_log_fast(float x);
float ref_sqrt_newton(float x);
float ref_sqrt_vfp(float x);
float ref_sqrt_fast(float x);
float ref_rsqrt_newton(float x);
float ref_rsqrt_vfp(float x);
float ref_rsqrt_fast(float x);
float ref_recip_vdiv(float x);
float ref_recip_fast(float x);

// ---- scalar integer division ----------------------------------------------
// Both are exact for every 32-bit input pair (b != 0) and truncate toward
// zero, like C and like Triton.  shift: restoring shift-subtract, 32 steps.
// est: float reciprocal estimate plus an integer correction, which is the
// algorithm PIton emits for `//` and `%`.
int32_t ref_idiv_shift(int32_t a, int32_t b);
int32_t ref_imod_shift(int32_t a, int32_t b);
int32_t ref_idiv_est(int32_t a, int32_t b);
int32_t ref_imod_est(int32_t a, int32_t b);

// ---- elementwise: scalar and unrolled -------------------------------------
void ref_copy_naive(const float *x, float *out, int n);
void ref_copy_unroll(const float *x, float *out, int n);
void ref_add_naive(const float *x, const float *y, float *out, int n);
void ref_add_unroll(const float *x, const float *y, float *out, int n);
void ref_mul_naive(const float *x, const float *y, float *out, int n);
void ref_mul_unroll(const float *x, const float *y, float *out, int n);
void ref_fma_naive(const float *x, const float *y, float *out, int n, float alpha);
void ref_fma_unroll(const float *x, const float *y, float *out, int n, float alpha);
void ref_relu_naive(const float *x, float *out, int n, float alpha);
void ref_relu_unroll(const float *x, float *out, int n, float alpha);

// ---- elementwise: one unary function over the whole vector -----------------
void ref_map(const float *x, float *out, int n, float (*f)(float));
void ref_map_unroll(const float *x, float *out, int n, float (*f)(float));

// ---- integer division over a vector ---------------------------------------
void ref_idivmod_shift(const int32_t *a, const int32_t *b, int32_t *q, int32_t *r, int n);
void ref_idivmod_est(const int32_t *a, const int32_t *b, int32_t *q, int32_t *r, int n);

// ---- row reductions (row-major, `rows` rows of `cols`, row stride `cols`) --
void ref_rowsum_naive(const float *x, float *out, int rows, int cols);
void ref_rowsum_unroll(const float *x, float *out, int rows, int cols);
void ref_rowsum_vfp(const float *x, float *out, int rows, int cols);
void ref_rowmax_naive(const float *x, float *out, int rows, int cols);
void ref_rowmax_unroll(const float *x, float *out, int rows, int cols);

// ---- transpose -------------------------------------------------------------
void ref_transpose_naive(const float *x, float *out, int rows, int cols);
void ref_transpose_blocked(const float *x, float *out, int rows, int cols);

// ---- softmax (one row at a time, max / exp / sum / divide) -----------------
void ref_softmax_acc(const float *x, float *out, int rows, int cols);
void ref_softmax_fast(const float *x, float *out, int rows, int cols);

// ---- assembly, VFP11 short-vector mode (arm_vfp_ops.S) --------------------
// All require n > 0 and n divisible by 8 (arm_sum_vfp: divisible by 4).
void arm_copy_vfp(const float *x, float *out, int n);
void arm_add_vfp(const float *x, const float *y, float *out, int n);
void arm_mul_vfp(const float *x, const float *y, float *out, int n);
void arm_fma_vfp(const float *x, const float *y, float *out, int n, float alpha);
float arm_sum_vfp(const float *x, int n);

#endif
