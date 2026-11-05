// ARM1176 baselines for the image_filters benchmark: the same filters in
// plain C (-O3, VFP, run with the L1 caches on), clamped borders, row-major
// images with stride w. Every one gives the QPU's bits except the square root
// (VFP vsqrt is correctly rounded; PIton's tl.sqrt is within a few ulp).
#ifndef ARM_FILTERS_H
#define ARM_FILTERS_H

// Direct 2-D stencils, the straightforward translation of blur_kernel /
// sobel_kernel / blur_sobel_kernel: every output pixel sums its clamped taps.
void arm_blur3(const float *src, float *dst, int h, int w);
void arm_blur5(const float *src, float *dst, int h, int w);
void arm_sobel(const float *src, float *dst, int h, int w);
void arm_blur_sobel(const float *src, float *dst, int h, int w);

// The usual CPU forms: the 5x5 blur as a row pass and a column pass through
// tmp (clamping only at the edges), and blur then Sobel as two passes. (The
// two-pass blur + Sobel clamps each 3x3 pass, so it differs from the fused
// 5x5 stencil in the outer two pixels.)
void arm_blur5_separable(const float *src, float *tmp, float *dst, int h, int w);
void arm_blur_sobel_two_pass(const float *src, float *tmp, float *dst, int h, int w);

#endif
