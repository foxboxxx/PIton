#include "arm_ref.h"

// No libm, no libgcc: every transcendental and the integer division are built
// from bit manipulation, polynomials and Newton iterations.  See arm_ref.h for
// what each baseline label means.

#define LOG2E   1.4426950408889634f
#define LN2     0.6931471805599453f
#define SQRT2   1.4142135623730951f
#define INF_F   3.4028234663852886e+38f    // saturates rather than returning inf

typedef union {
    float f;
    uint32_t u;
    int32_t i;
} fbits;

// ---------------------------------------------------------------------------
// exp
// ---------------------------------------------------------------------------

// Scale by 2^n with a single exponent insert (n must keep the result finite).
static inline float ldexp_pow2(int n) {
    fbits s;
    int e = 127 + n;
    if (e <= 0)
        return 0.0f;
    if (e >= 255)
        return INF_F;
    s.u = (uint32_t)e << 23;
    return s.f;
}

static inline int round_to_int(float y) {
    return (int)(y < 0.0f ? y - 0.5f : y + 0.5f);
}

// exp(x) = 2^(x*log2(e)), split into an integer power of two and a degree-6
// polynomial for 2^f on f in [-0.5, 0.5].
float ref_exp_acc(float x) {
    if (x > 88.0f)
        return INF_F;
    if (x < -88.0f)
        return 0.0f;
    float y = x * LOG2E;
    int n = round_to_int(y);
    float f = y - (float)n;
    float p = 1.0f + f * (0.6931471805599453f
            + f * (0.2402265069591007f
            + f * (0.0555041086648216f
            + f * (0.0096181291076285f
            + f * (0.0013333558146428f
            + f *  0.0001540353039338f)))));
    return p * ldexp_pow2(n);
}

// Same split, but only a quadratic for 2^f on [-0.5, 0.5]: about 0.5%
// relative, in exchange for four fewer multiply-adds per element.
float ref_exp_fast(float x) {
    if (x > 88.0f)
        return INF_F;
    if (x < -88.0f)
        return 0.0f;
    float y = x * LOG2E;
    int n = round_to_int(y);
    float f = y - (float)n;
    float p = 1.0f + f * (0.6960656421638072f + f * 0.2240713099781492f);
    return p * ldexp_pow2(n);
}

// ---------------------------------------------------------------------------
// log
// ---------------------------------------------------------------------------

// log(x) = e*ln2 + log(m), m in [1/sqrt2, sqrt2).  log(m) via the atanh series
// in t = (m-1)/(m+1), where |t| <= 0.1716 and t^9/9 is already below 1e-8.
// The one divide is a hardware vdiv.
float ref_log_acc(float x) {
    if (x <= 0.0f)
        return -INF_F;
    fbits v;
    v.f = x;
    int e = (int)((v.u >> 23) & 0xff) - 127;
    v.u = (v.u & 0x007fffffu) | 0x3f800000u;
    float m = v.f;
    if (m > SQRT2) {
        m *= 0.5f;
        e += 1;
    }
    float t = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;
    float s = t * (2.0f + t2 * (0.6666666666666666f
                 + t2 * (0.4f
                 + t2 * (0.2857142857142857f
                 + t2 *  0.2222222222222222f))));
    return s + (float)e * LN2;
}

// Exponent straight from the bit pattern, plus a quadratic in the mantissa.
float ref_log_fast(float x) {
    if (x <= 0.0f)
        return -INF_F;
    fbits v;
    v.f = x;
    int e = (int)((v.u >> 23) & 0xff) - 127;
    v.u = (v.u & 0x007fffffu) | 0x3f800000u;
    float m = v.f;                       // [1, 2)
    // Quadratic through log2 at m = 1, 1.5, 2, so the two endpoints are exact
    // and the worst case in between is about 1%.  A quadratic cannot do better
    // than that for log2 over a whole octave; the point of this baseline is to
    // be cheap, not accurate.
    float l2 = -1.6797f + m * (2.01955f - m * 0.33985f);
    return ((float)e + l2) * LN2;
}

// ---------------------------------------------------------------------------
// sqrt / rsqrt / reciprocal
// ---------------------------------------------------------------------------

float ref_sqrt_vfp(float x) {
    float r;
    __asm__("vsqrt.f32 %0, %1" : "=t"(r) : "t"(x));
    return r;
}

// Seed from halving the exponent, then three Newton steps on y = 1/sqrt(x)
// (multiply only), and one multiply by x.  No vsqrt, no divide.
static inline float rsqrt_seed(float x) {
    fbits v;
    v.f = x;
    v.i = 0x5f3759df - (v.i >> 1);       // the classic inverse-square-root seed
    return v.f;
}

float ref_rsqrt_newton(float x) {
    if (x <= 0.0f)
        return INF_F;
    float y = rsqrt_seed(x);
    float h = 0.5f * x;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
}

float ref_rsqrt_fast(float x) {
    if (x <= 0.0f)
        return INF_F;
    float y = rsqrt_seed(x);
    return y * (1.5f - 0.5f * x * y * y);     // one step: ~1.8e-3
}

float ref_rsqrt_vfp(float x) {
    if (x <= 0.0f)
        return INF_F;
    return 1.0f / ref_sqrt_vfp(x);            // vsqrt followed by vdiv
}

float ref_sqrt_newton(float x) {
    if (x <= 0.0f)
        return 0.0f;
    return x * ref_rsqrt_newton(x);
}

float ref_sqrt_fast(float x) {
    if (x <= 0.0f)
        return 0.0f;
    return x * ref_rsqrt_fast(x);
}

float ref_recip_vdiv(float x) {
    return 1.0f / x;
}

// Exponent-negation seed plus two Newton steps: multiplies only, no vdiv.
float ref_recip_fast(float x) {
    fbits v;
    v.f = x;
    v.u = 0x7eb53567u - v.u;             // seed for 1/x
    float y = v.f;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

// ---------------------------------------------------------------------------
// integer division (no __aeabi_idiv in a freestanding build)
// ---------------------------------------------------------------------------

// Restoring shift-subtract on the magnitudes, 32 iterations.
static void udivmod_shift(uint32_t a, uint32_t b, uint32_t *q, uint32_t *r) {
    uint32_t quot = 0, rem = 0;
    for (int i = 31; i >= 0; i--) {
        rem = (rem << 1) | ((a >> i) & 1u);
        if (rem >= b) {
            rem -= b;
            quot |= 1u << i;
        }
    }
    *q = quot;
    *r = rem;
}

// Unsigned negation, so INT32_MIN maps to 2^31 without 64-bit arithmetic
// (which would pull in a libgcc helper this build does not link).
static inline uint32_t abs32(int32_t v) {
    return v < 0 ? 0u - (uint32_t)v : (uint32_t)v;
}

static void idivmod_shift(int32_t a, int32_t b, int32_t *q, int32_t *r) {
    uint32_t uq, ur;
    udivmod_shift(abs32(a), abs32(b), &uq, &ur);
    // truncate toward zero; the remainder takes the sign of the dividend
    *q = ((a < 0) != (b < 0)) ? -(int32_t)uq : (int32_t)uq;
    *r = (a < 0) ? -(int32_t)ur : (int32_t)ur;
}

int32_t ref_idiv_shift(int32_t a, int32_t b) {
    int32_t q, r;
    idivmod_shift(a, b, &q, &r);
    return q;
}

int32_t ref_imod_shift(int32_t a, int32_t b) {
    int32_t q, r;
    idivmod_shift(a, b, &q, &r);
    return r;
}

// Float reciprocal estimate, then correct the quotient with integer
// multiplies until the remainder is in range.  This is the shape of the code
// PIton emits, and like it, the result is exact for every input pair: the
// float estimate is off by at most a couple of units, and the loops below fix
// that up with exact integer arithmetic.
static void udivmod_est(uint32_t a, uint32_t b, uint32_t *q, uint32_t *r) {
    if (a < b) {
        *q = 0;
        *r = a;
        return;
    }
    float recip = 1.0f / (float)b;
    uint32_t quot = (uint32_t)((float)a * recip);
    // (float)a * recip can land either side of the true quotient
    uint32_t prod = quot * b;
    while (prod > a) {                       // estimate too large
        quot -= 1u;
        prod -= b;
    }
    while (a - prod >= b) {                  // estimate too small
        quot += 1u;
        prod += b;
    }
    *q = quot;
    *r = a - prod;
}

static void idivmod_est(int32_t a, int32_t b, int32_t *q, int32_t *r) {
    uint32_t uq, ur;
    udivmod_est(abs32(a), abs32(b), &uq, &ur);
    *q = ((a < 0) != (b < 0)) ? -(int32_t)uq : (int32_t)uq;
    *r = (a < 0) ? -(int32_t)ur : (int32_t)ur;
}

int32_t ref_idiv_est(int32_t a, int32_t b) {
    int32_t q, r;
    idivmod_est(a, b, &q, &r);
    return q;
}

int32_t ref_imod_est(int32_t a, int32_t b) {
    int32_t q, r;
    idivmod_est(a, b, &q, &r);
    return r;
}

void ref_idivmod_shift(const int32_t *a, const int32_t *b, int32_t *q, int32_t *r, int n) {
    for (int i = 0; i < n; i++)
        idivmod_shift(a[i], b[i], &q[i], &r[i]);
}

void ref_idivmod_est(const int32_t *a, const int32_t *b, int32_t *q, int32_t *r, int n) {
    for (int i = 0; i < n; i++)
        idivmod_est(a[i], b[i], &q[i], &r[i]);
}

// ---------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------

void ref_copy_naive(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i];
}

void ref_copy_unroll(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(x + i + 32);
        float a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
        out[i] = a; out[i + 1] = b; out[i + 2] = c; out[i + 3] = d;
    }
    for (; i < n; i++)
        out[i] = x[i];
}

void ref_add_naive(const float *x, const float *y, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] + y[i];
}

void ref_add_unroll(const float *x, const float *y, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(x + i + 32);
        __builtin_prefetch(y + i + 32);
        out[i] = x[i] + y[i];
        out[i + 1] = x[i + 1] + y[i + 1];
        out[i + 2] = x[i + 2] + y[i + 2];
        out[i + 3] = x[i + 3] + y[i + 3];
    }
    for (; i < n; i++)
        out[i] = x[i] + y[i];
}

void ref_mul_naive(const float *x, const float *y, float *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] * y[i];
}

void ref_mul_unroll(const float *x, const float *y, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(x + i + 32);
        __builtin_prefetch(y + i + 32);
        out[i] = x[i] * y[i];
        out[i + 1] = x[i + 1] * y[i + 1];
        out[i + 2] = x[i + 2] * y[i + 2];
        out[i + 3] = x[i + 3] * y[i + 3];
    }
    for (; i < n; i++)
        out[i] = x[i] * y[i];
}

void ref_fma_naive(const float *x, const float *y, float *out, int n, float alpha) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] * alpha + y[i];
}

void ref_fma_unroll(const float *x, const float *y, float *out, int n, float alpha) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(x + i + 32);
        __builtin_prefetch(y + i + 32);
        out[i] = x[i] * alpha + y[i];
        out[i + 1] = x[i + 1] * alpha + y[i + 1];
        out[i + 2] = x[i + 2] * alpha + y[i + 2];
        out[i + 3] = x[i + 3] * alpha + y[i + 3];
    }
    for (; i < n; i++)
        out[i] = x[i] * alpha + y[i];
}

void ref_relu_naive(const float *x, float *out, int n, float alpha) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = v > 0.0f ? v * alpha : 0.0f;
    }
}

void ref_relu_unroll(const float *x, float *out, int n, float alpha) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(x + i + 32);
        float a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
        out[i]     = a > 0.0f ? a * alpha : 0.0f;
        out[i + 1] = b > 0.0f ? b * alpha : 0.0f;
        out[i + 2] = c > 0.0f ? c * alpha : 0.0f;
        out[i + 3] = d > 0.0f ? d * alpha : 0.0f;
    }
    for (; i < n; i++) {
        float v = x[i];
        out[i] = v > 0.0f ? v * alpha : 0.0f;
    }
}

// The function pointer is deliberately not inlined: all the transcendental
// baselines go through the same loop, so the loop itself is not what differs.
void ref_map(const float *x, float *out, int n, float (*f)(float)) {
    for (int i = 0; i < n; i++)
        out[i] = f(x[i]);
}

void ref_map_unroll(const float *x, float *out, int n, float (*f)(float)) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __builtin_prefetch(x + i + 32);
        out[i]     = f(x[i]);
        out[i + 1] = f(x[i + 1]);
        out[i + 2] = f(x[i + 2]);
        out[i + 3] = f(x[i + 3]);
    }
    for (; i < n; i++)
        out[i] = f(x[i]);
}

// ---------------------------------------------------------------------------
// row reductions
// ---------------------------------------------------------------------------

void ref_rowsum_naive(const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = x + i * cols;
        float s = 0.0f;
        for (int j = 0; j < cols; j++)
            s += row[j];
        out[i] = s;
    }
}

// Four independent accumulators: the naive version is limited by the VFP add
// latency, not by throughput.  Summation order differs from naive, so the
// results differ in the last bits.
void ref_rowsum_unroll(const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = x + i * cols;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int j = 0;
        for (; j + 4 <= cols; j += 4) {
            s0 += row[j];
            s1 += row[j + 1];
            s2 += row[j + 2];
            s3 += row[j + 3];
        }
        for (; j < cols; j++)
            s0 += row[j];
        out[i] = (s0 + s1) + (s2 + s3);
    }
}

void ref_rowsum_vfp(const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        out[i] = arm_sum_vfp(x + i * cols, cols);
}

void ref_rowmax_naive(const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = x + i * cols;
        float m = row[0];
        for (int j = 1; j < cols; j++)
            if (row[j] > m)
                m = row[j];
        out[i] = m;
    }
}

void ref_rowmax_unroll(const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *row = x + i * cols;
        float m0 = row[0], m1 = row[0], m2 = row[0], m3 = row[0];
        int j = 0;
        for (; j + 4 <= cols; j += 4) {
            if (row[j] > m0)     m0 = row[j];
            if (row[j + 1] > m1) m1 = row[j + 1];
            if (row[j + 2] > m2) m2 = row[j + 2];
            if (row[j + 3] > m3) m3 = row[j + 3];
        }
        for (; j < cols; j++)
            if (row[j] > m0)
                m0 = row[j];
        float a = m0 > m1 ? m0 : m1;
        float b = m2 > m3 ? m2 : m3;
        out[i] = a > b ? a : b;
    }
}

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

// Strided stores: every write lands in a different cache line.
void ref_transpose_naive(const float *x, float *out, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j * rows + i] = x[i * cols + j];
}

#define TR_BLK 16

void ref_transpose_blocked(const float *x, float *out, int rows, int cols) {
    for (int i0 = 0; i0 < rows; i0 += TR_BLK) {
        int imax = i0 + TR_BLK < rows ? i0 + TR_BLK : rows;
        for (int j0 = 0; j0 < cols; j0 += TR_BLK) {
            int jmax = j0 + TR_BLK < cols ? j0 + TR_BLK : cols;
            for (int i = i0; i < imax; i++)
                for (int j = j0; j < jmax; j++)
                    out[j * rows + i] = x[i * cols + j];
        }
    }
}

// ---------------------------------------------------------------------------
// softmax: max, exp, sum, divide -- the same four steps as the Triton kernel
// ---------------------------------------------------------------------------

static void softmax_rows(const float *x, float *out, int rows, int cols, float (*expf_)(float)) {
    for (int i = 0; i < rows; i++) {
        const float *row = x + i * cols;
        float *o = out + i * cols;
        float m = row[0];
        for (int j = 1; j < cols; j++)
            if (row[j] > m)
                m = row[j];
        float s = 0.0f;
        for (int j = 0; j < cols; j++) {
            float e = expf_(row[j] - m);
            o[j] = e;
            s += e;
        }
        float inv = 1.0f / s;
        for (int j = 0; j < cols; j++)
            o[j] *= inv;
    }
}

void ref_softmax_acc(const float *x, float *out, int rows, int cols) {
    softmax_rows(x, out, rows, cols, ref_exp_acc);
}

void ref_softmax_fast(const float *x, float *out, int rows, int cols) {
    softmax_rows(x, out, rows, cols, ref_exp_fast);
}
