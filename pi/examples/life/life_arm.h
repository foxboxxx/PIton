// Game of Life on the ARM1176: the CPU baselines and the helpers the firmware
// uses to make, check and print grids. Plain C (no libpi), so the same file is
// also compiled on the host by examples/life_pi.py --dry-run and the tests.
//
// Layouts, as in examples/life.py:
//   cells  int32 0/1, row-major [rows][cols]
//   words  uint32 [rows][cols / 32], bit j of word k is column 32k + j
// Both are tori. cols must be a multiple of 32 wherever words are involved.
#ifndef LIFE_ARM_H
#define LIFE_ARM_H
#include <stdint.h>

typedef struct {
    uint32_t pop;     // live cells
    uint32_t hash;    // sum over the packed words w_i of fmix32(w_i ^ i * 0x9e3779b9), mod 2^32
    uint32_t bad;     // cells that are neither 0 nor 1 (cells layout only)
} life_trace_t;

// The soup of examples/life.py soup(): cell i (row-major) is alive when the top
// three bits of the i-th piton_lcg() output from `seed` are below `eighths`.
void life_soup_cells(volatile int32_t *cells, unsigned rows, unsigned cols, uint32_t seed, unsigned eighths);
void life_soup_words(volatile uint32_t *words, unsigned rows, unsigned cols, uint32_t seed, unsigned eighths);

life_trace_t life_trace_cells(volatile const int32_t *cells, unsigned rows, unsigned cols);
life_trace_t life_trace_words(volatile const uint32_t *words, unsigned rows, unsigned cols);

// Pack `cells` (any values: bit 0 is kept) into words.
void life_pack(volatile const int32_t *cells, uint32_t *words, unsigned rows, unsigned cols);
// Cells that differ from the packed reference `ref`.
unsigned life_diff_cells(volatile const int32_t *cells, const uint32_t *ref, unsigned rows, unsigned cols);
unsigned life_diff_words(volatile const uint32_t *words, const uint32_t *ref, unsigned rows, unsigned cols);
// The top-left crop x crop cells (crop a multiple of 32) as packed words.
void life_crop_cells(volatile const int32_t *cells, unsigned cols, unsigned crop, uint32_t *out);
void life_crop_words(volatile const uint32_t *words, unsigned cols, unsigned crop, uint32_t *out);

// One generation, src -> dst (no aliasing).
// naive: the eight neighbours of every cell, wrap-around by compare and select.
void life_arm_naive(const int32_t *src, int32_t *dst, unsigned rows, unsigned cols);
// rows: vertical 3-cell sums of each row once (scratch: cols ints), then
// horizontal sums of those: 6 loads and 4 adds per cell instead of 8 and 7.
void life_arm_rows(const int32_t *src, int32_t *dst, unsigned rows, unsigned cols, int32_t *scratch);
// bits: the bit-sliced adder of life_bits_kernel on packed words, 32 cells
// per operation, with the horizontal sums of the previous two rows kept
// (scratch: 6 * cols / 32 words).
void life_arm_bits(const uint32_t *src, uint32_t *dst, unsigned rows, unsigned cols, uint32_t *scratch);

#endif
