// Row hashes of a float32 image, printed by main.c and recomputed on the host
// (row_hashes() in examples/image_filters_pi.py, checked against this file by
// tests/test_image_filters.py): the sum over x of mix32(bits ^ x * golden),
// wrapping, where mix32 is MurmurHash3's finaliser.
#ifndef IMAGE_FILTERS_ROW_HASH_H
#define IMAGE_FILTERS_ROW_HASH_H

#include <stdint.h>

static inline uint32_t mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

static inline uint32_t row_hash(volatile const uint32_t *row, int w) {
    uint32_t h = 0;
    for (int x = 0; x < w; x++)
        h += mix32(row[x] ^ ((uint32_t)x * 0x9E3779B9u));
    return h;
}

#endif
