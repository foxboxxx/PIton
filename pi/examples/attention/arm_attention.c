#include <stdint.h>
#include "arm_attention.h"

// Freestanding build: no libm, no libgcc (no runtime integer division).

typedef union {
    float f;
    uint32_t u;
} fbits;

float arm_expf(float x) {
    if (x < -87.0f)
        return 0.0f;
    float y = x * 1.4426950408889634f;
    int n = (int)(y < 0.0f ? y - 0.5f : y + 0.5f);
    float f = y - (float)n;
    float p = 1.0f + f * (0.6931471805599453f
            + f * (0.2402265069591007f
            + f * (0.0555041086648216f
            + f * (0.0096181291076285f
            + f * (0.0013333558146428f
            + f *  0.0001540353039338f)))));
    fbits s;
    s.u = (uint32_t)(127 + n) << 23;      // x <= 0: 2^n is normal for n >= -126
    return p * s.f;
}

void arm_attention(const float *q, const float *k, const float *v, float *out,
                   int n, int d, int causal, float sm_scale, float *scores) {
    for (int i = 0; i < n; i++) {
        const float *qi = q + i * d;
        int keys = causal ? i + 1 : n;
        float m = -3.4028234663852886e+38f;
        for (int j = 0; j < keys; j++) {
            const float *kj = k + j * d;
            float s = 0.0f;
            for (int c = 0; c < d; c++)
                s += qi[c] * kj[c];
            s *= sm_scale;
            scores[j] = s;
            if (s > m)
                m = s;
        }
        float sum = 0.0f;
        for (int j = 0; j < keys; j++) {
            float p = arm_expf(scores[j] - m);
            scores[j] = p;
            sum += p;
        }
        float inv = 1.0f / sum;
        float *oi = out + i * d;
        for (int c = 0; c < d; c++)
            oi[c] = 0.0f;
        for (int j = 0; j < keys; j++) {
            float p = scores[j] * inv;
            const float *vj = v + j * d;
            for (int c = 0; c < d; c++)
                oi[c] += p * vj[c];
        }
    }
}
