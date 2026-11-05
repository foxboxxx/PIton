// Straightforward single-precision attention on the ARM1176 (the CPU baseline
// of examples/attention_pi.py): for every query row, all scores, their maximum,
// exp and sum, then the weighted sum of the value rows.  No blocking, no
// assembly; built with -O3 and run with the data cache on.
#ifndef ARM_ATTENTION_H
#define ARM_ATTENTION_H

// exp(x) for x <= 0: 2^n times a degree-6 polynomial in the remaining fraction
// (the accurate exp of pi/examples/primitives, `ref_exp_acc`, ~1 ulp).
float arm_expf(float x);

// out[n, d] = softmax(q k^T * sm_scale) v for row-major q, k, v [n, d];
// causal: row i sees keys 0..i.  scores: n floats of scratch.
void arm_attention(const float *q, const float *k, const float *v, float *out,
                   int n, int d, int causal, float sm_scale, float *scores);
#endif
