// See arm_filters.h. The weights are small integers and the images integers
// 0..255 (or multiples of 1/16 after a blur), so every sum is exact in float32
// and the order of the additions does not matter.
#include "arm_filters.h"

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

static inline float vsqrt(float x) {
#if defined(__arm__)
    float r;
    __asm__("vsqrt.f32 %0, %1" : "=t"(r) : "t"(x));   // no libm in the bare-metal build
    return r;
#else
    return __builtin_sqrtf(x);                          // the host simulation in the tests
#endif
}

static const float binomial3[3] = {1, 2, 1}, binomial5[5] = {1, 4, 6, 4, 1};
static const float deriv3[3] = {-1, 0, 1}, deriv5[5] = {-1, -2, 0, 2, 1};

// R is a constant at each call site, so the tap loops unroll.
static inline void blur_direct(const float *src, float *dst, int h, int w, const int R) {
    const float *k = R == 1 ? binomial3 : binomial5;
    const float scale = R == 1 ? 1.0f / 16 : 1.0f / 256;
    for (int y = 0; y < h; y++) {
        const float *row[5];
        for (int i = 0; i <= 2 * R; i++)
            row[i] = src + clampi(y + i - R, 0, h - 1) * w;
        float *out = dst + y * w;
        for (int x = 0; x < w; x++) {
            int col[5];
            for (int j = 0; j <= 2 * R; j++)
                col[j] = clampi(x + j - R, 0, w - 1);
            float acc = 0;
            for (int i = 0; i <= 2 * R; i++) {
                float s = 0;
                for (int j = 0; j <= 2 * R; j++)
                    s += k[j] * row[i][col[j]];
                acc += k[i] * s;
            }
            out[x] = acc * scale;
        }
    }
}

void arm_blur3(const float *src, float *dst, int h, int w) { blur_direct(src, dst, h, w, 1); }
void arm_blur5(const float *src, float *dst, int h, int w) { blur_direct(src, dst, h, w, 2); }

static inline void sobel_direct(const float *src, float *dst, int h, int w, const int R) {
    const float *s = R == 1 ? binomial3 : binomial5, *d = R == 1 ? deriv3 : deriv5;
    float wx[5][5], wy[5][5];           // gx and gy weights of every tap (zero ones included)
    for (int i = 0; i <= 2 * R; i++)
        for (int j = 0; j <= 2 * R; j++) {
            wx[i][j] = s[i] * d[j];
            wy[i][j] = d[i] * s[j];
        }
    for (int y = 0; y < h; y++) {
        const float *row[5];
        for (int i = 0; i <= 2 * R; i++)
            row[i] = src + clampi(y + i - R, 0, h - 1) * w;
        float *out = dst + y * w;
        for (int x = 0; x < w; x++) {
            int col[5];
            for (int j = 0; j <= 2 * R; j++)
                col[j] = clampi(x + j - R, 0, w - 1);
            float gx = 0, gy = 0;
            for (int i = 0; i <= 2 * R; i++)
                for (int j = 0; j <= 2 * R; j++) {
                    float v = row[i][col[j]];
                    gx += wx[i][j] * v;
                    gy += wy[i][j] * v;
                }
            if (R == 2) {           // the 3x3 blur's normalisation
                gx *= 1.0f / 16;
                gy *= 1.0f / 16;
            }
            out[x] = vsqrt(gx * gx + gy * gy);
        }
    }
}

void arm_sobel(const float *src, float *dst, int h, int w) { sobel_direct(src, dst, h, w, 1); }
void arm_blur_sobel(const float *src, float *dst, int h, int w) { sobel_direct(src, dst, h, w, 2); }

void arm_blur5_separable(const float *src, float *tmp, float *dst, int h, int w) {
    for (int y = 0; y < h; y++) {
        const float *in = src + y * w;
        float *out = tmp + y * w;
        for (int x = 0; x < w; x++) {
            if (x >= 2 && x < w - 2) {
                out[x] = (in[x - 2] + 4 * in[x - 1] + 6 * in[x] + 4 * in[x + 1] + in[x + 2]) * (1.0f / 16);
            } else {
                out[x] = (in[clampi(x - 2, 0, w - 1)] + 4 * in[clampi(x - 1, 0, w - 1)] + 6 * in[x]
                          + 4 * in[clampi(x + 1, 0, w - 1)] + in[clampi(x + 2, 0, w - 1)]) * (1.0f / 16);
            }
        }
    }
    for (int y = 0; y < h; y++) {
        const float *r0 = tmp + clampi(y - 2, 0, h - 1) * w, *r1 = tmp + clampi(y - 1, 0, h - 1) * w;
        const float *r2 = tmp + y * w;
        const float *r3 = tmp + clampi(y + 1, 0, h - 1) * w, *r4 = tmp + clampi(y + 2, 0, h - 1) * w;
        float *out = dst + y * w;
        for (int x = 0; x < w; x++)
            out[x] = (r0[x] + 4 * r1[x] + 6 * r2[x] + 4 * r3[x] + r4[x]) * (1.0f / 16);
    }
}

void arm_blur_sobel_two_pass(const float *src, float *tmp, float *dst, int h, int w) {
    blur_direct(src, tmp, h, w, 1);
    sobel_direct(tmp, dst, h, w, 1);
}
