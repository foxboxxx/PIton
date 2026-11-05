// Game of Life CPU baselines and grid helpers (see life_arm.h). No division
// anywhere: the ARM1176 has no divide instruction and libpi links no
// __aeabi_uidiv.
#include "life_arm.h"

static inline uint32_t lcg_next(uint32_t *s) {   // piton_lcg() of piton_rt.h
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static inline uint32_t popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return (x * 0x01010101u) >> 24;
}

// murmur3's finaliser; the trace hash sums it over the salted words, so the
// host can compute it vectorised (examples/life.py grid_hash).
static inline uint32_t fmix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    return x ^ (x >> 16);
}

#define LIFE_SALT 0x9e3779b9u

static inline uint32_t pack32(volatile const int32_t *cells) {
    uint32_t w = 0;
    for (unsigned j = 0; j < 32; j++)
        w |= ((uint32_t)cells[j] & 1u) << j;
    return w;
}

void life_soup_cells(volatile int32_t *cells, unsigned rows, unsigned cols, uint32_t seed, unsigned eighths) {
    uint32_t s = seed;
    unsigned n = rows * cols;
    for (unsigned i = 0; i < n; i++)
        cells[i] = (lcg_next(&s) >> 29) < eighths;
}

void life_soup_words(volatile uint32_t *words, unsigned rows, unsigned cols, uint32_t seed, unsigned eighths) {
    uint32_t s = seed;
    unsigned n = rows * (cols >> 5);
    for (unsigned i = 0; i < n; i++) {
        uint32_t w = 0;
        for (unsigned j = 0; j < 32; j++)
            w |= (uint32_t)((lcg_next(&s) >> 29) < eighths) << j;
        words[i] = w;
    }
}

life_trace_t life_trace_cells(volatile const int32_t *cells, unsigned rows, unsigned cols) {
    life_trace_t t = { 0, 0, 0 };
    unsigned n = rows * cols;
    uint32_t salt = 0;
    for (unsigned i = 0; i < n; i += 32, salt += LIFE_SALT) {
        uint32_t w = 0;
        for (unsigned j = 0; j < 32; j++) {
            uint32_t v = (uint32_t)cells[i + j];
            t.bad += v > 1u;
            w |= (v & 1u) << j;
        }
        t.hash += fmix32(w ^ salt);
        t.pop += popcount32(w);
    }
    return t;
}

life_trace_t life_trace_words(volatile const uint32_t *words, unsigned rows, unsigned cols) {
    life_trace_t t = { 0, 0, 0 };
    unsigned n = rows * (cols >> 5);
    uint32_t salt = 0;
    for (unsigned i = 0; i < n; i++, salt += LIFE_SALT) {
        uint32_t w = words[i];
        t.hash += fmix32(w ^ salt);
        t.pop += popcount32(w);
    }
    return t;
}

void life_pack(volatile const int32_t *cells, uint32_t *words, unsigned rows, unsigned cols) {
    unsigned n = rows * (cols >> 5);
    for (unsigned i = 0; i < n; i++)
        words[i] = pack32(cells + 32 * i);
}

unsigned life_diff_cells(volatile const int32_t *cells, const uint32_t *ref, unsigned rows, unsigned cols) {
    unsigned n = rows * (cols >> 5), differ = 0;
    for (unsigned i = 0; i < n; i++)
        differ += popcount32(pack32(cells + 32 * i) ^ ref[i]);
    return differ;
}

unsigned life_diff_words(volatile const uint32_t *words, const uint32_t *ref, unsigned rows, unsigned cols) {
    unsigned n = rows * (cols >> 5), differ = 0;
    for (unsigned i = 0; i < n; i++)
        differ += popcount32(words[i] ^ ref[i]);
    return differ;
}

void life_crop_cells(volatile const int32_t *cells, unsigned cols, unsigned crop, uint32_t *out) {
    unsigned per_row = crop >> 5;
    for (unsigned r = 0; r < crop; r++)
        for (unsigned k = 0; k < per_row; k++)
            out[r * per_row + k] = pack32(cells + r * cols + 32 * k);
}

void life_crop_words(volatile const uint32_t *words, unsigned cols, unsigned crop, uint32_t *out) {
    unsigned per_row = crop >> 5, nw = cols >> 5;
    for (unsigned r = 0; r < crop; r++)
        for (unsigned k = 0; k < per_row; k++)
            out[r * per_row + k] = words[r * nw + k];
}

// ---------------------------------------------------------------------------
// CPU baselines
// ---------------------------------------------------------------------------

void life_arm_naive(const int32_t *src, int32_t *dst, unsigned rows, unsigned cols) {
    for (unsigned r = 0; r < rows; r++) {
        const int32_t *up = src + (r ? r - 1 : rows - 1) * cols;
        const int32_t *mid = src + r * cols;
        const int32_t *down = src + (r + 1 < rows ? r + 1 : 0) * cols;
        int32_t *out = dst + r * cols;
        for (unsigned c = 0; c < cols; c++) {
            unsigned left = c ? c - 1 : cols - 1, right = c + 1 < cols ? c + 1 : 0;
            int32_t n = up[left] + up[c] + up[right] + mid[left] + mid[right]
                      + down[left] + down[c] + down[right];
            out[c] = n == 3 || (n == 2 && mid[c]);
        }
    }
}

static inline int32_t rule(int32_t total, int32_t alive) {   // total includes the cell
    return total == 3 || (total == 4 && alive);
}

void life_arm_rows(const int32_t *src, int32_t *dst, unsigned rows, unsigned cols, int32_t *v) {
    for (unsigned r = 0; r < rows; r++) {
        const int32_t *up = src + (r ? r - 1 : rows - 1) * cols;
        const int32_t *mid = src + r * cols;
        const int32_t *down = src + (r + 1 < rows ? r + 1 : 0) * cols;
        int32_t *out = dst + r * cols;
        for (unsigned c = 0; c < cols; c++)
            v[c] = up[c] + mid[c] + down[c];
        unsigned last = cols - 1;
        out[0] = rule(v[last] + v[0] + v[cols > 1 ? 1 : 0], mid[0]);
        for (unsigned c = 1; c < last; c++)
            out[c] = rule(v[c - 1] + v[c] + v[c + 1], mid[c]);
        if (last)
            out[last] = rule(v[last - 1] + v[last] + v[0], mid[last]);
    }
}

static void row_bit_sums(const uint32_t *row, unsigned nw, uint32_t *s0, uint32_t *s1) {
    for (unsigned k = 0; k < nw; k++) {
        uint32_t w = row[k], wl = row[k ? k - 1 : nw - 1], wr = row[k + 1 < nw ? k + 1 : 0];
        uint32_t west = (w << 1) | (wl >> 31), east = (w >> 1) | (wr << 31);
        uint32_t x = west ^ w;
        s0[k] = x ^ east;
        s1[k] = (west & w) | (x & east);
    }
}

void life_arm_bits(const uint32_t *src, uint32_t *dst, unsigned rows, unsigned cols, uint32_t *scratch) {
    unsigned nw = cols >> 5;
    uint32_t *a0 = scratch, *a1 = a0 + nw, *b0 = a1 + nw, *b1 = b0 + nw, *c0 = b1 + nw, *c1 = c0 + nw;
    row_bit_sums(src + (rows - 1) * nw, nw, a0, a1);
    row_bit_sums(src, nw, b0, b1);
    for (unsigned r = 0; r < rows; r++) {
        row_bit_sums(src + (r + 1 < rows ? r + 1 : 0) * nw, nw, c0, c1);
        const uint32_t *mid = src + r * nw;
        uint32_t *out = dst + r * nw;
        for (unsigned k = 0; k < nw; k++) {
            uint32_t t0 = a0[k] ^ b0[k] ^ c0[k];
            uint32_t k0 = (a0[k] & b0[k]) | (c0[k] & (a0[k] ^ b0[k]));
            uint32_t s = a1[k] ^ b1[k], s2 = s ^ c1[k];
            uint32_t t1 = s2 ^ k0;
            uint32_t t2 = (a1[k] & b1[k]) ^ (s & c1[k]) ^ (s2 & k0);
            out[k] = (t0 & t1 & ~t2) | (mid[k] & t2 & ~(t0 | t1));
        }
        uint32_t *t;
        t = a0; a0 = b0; b0 = c0; c0 = t;
        t = a1; a1 = b1; b1 = c1; c1 = t;
    }
}
