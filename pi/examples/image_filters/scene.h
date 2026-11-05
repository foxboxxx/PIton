// The image_filters test scene, generated on the Pi instead of uploaded.
//
// Integer arithmetic only, the twin of make_scene() in
// examples/image_filters.py (tests/test_image_filters.py compiles this file
// on the host and compares every pixel): a gradient, a checkerboard, a disc,
// a ring, a striped triangle, the word PITON in a 5x7 font and +-8 of hash
// noise, drawn on a 1024 x 768 canvas scaled to w x h. The scale factors are
// divisions the host does (scene_params()): the ARM1176 has no divide.
#ifndef IMAGE_FILTERS_SCENE_H
#define IMAGE_FILTERS_SCENE_H

#include <stdint.h>

typedef struct {
    int32_t w, h;        // image size
    int32_t sx, sy;      // (1024 << 10) / w, (768 << 10) / h: pixel -> canvas
    int32_t gx, gy;      // (120 << 16) / w, (60 << 16) / h: background gradient
} scene_params_t;

// "PITON", one byte per glyph row, bit 4 = leftmost column
static const uint8_t scene_font[5][7] = {
    {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
    {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
    {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
    {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
    {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},
};

static inline int32_t scene_pixel(const scene_params_t *p, int32_t x, int32_t y) {
    int32_t u = (x * p->sx) >> 10, v = (y * p->sy) >> 10;
    int32_t img = 40 + ((x * p->gx) >> 16) + ((y * p->gy) >> 16);
    if (u >= 80 && u < 440 && v >= 520 && v < 720)
        img = (((u >> 5) ^ (v >> 5)) & 1) ? 100 : 30;
    int32_t du = u - 250, dv = v - 340;
    if (du * du + dv * dv < 130 * 130)
        img = 215;
    du = u - 780;
    dv = v - 260;
    int32_t r2 = du * du + dv * dv;
    if (r2 >= 90 * 90 && r2 < 130 * 130)
        img = 20;
    int32_t e1 = (u - 560) * (440 - 720) - (v - 720) * (770 - 560);
    int32_t e2 = (u - 770) * (720 - 440) - (v - 440) * (980 - 770);
    if (v <= 720 && e1 <= 0 && e2 <= 0)
        img = (((u + v) >> 4) & 1) ? 170 : 230;
    if (u >= 50 && v >= 30) {
        int32_t fu = (u - 50) >> 4, fv = (v - 30) >> 4;      // 16 canvas units per font pixel
        if (fu < 8 * 5 && fv < 7 && (fu & 7) < 5 && ((scene_font[fu >> 3][fv] >> (4 - (fu & 7))) & 1))
            img = 250;
    }
    uint32_t hash = (uint32_t)(y * p->w + x) * 0x9E3779B1u;
    hash ^= hash >> 16;
    hash *= 0x85EBCA6Bu;
    hash ^= hash >> 13;
    img += (int32_t)(hash & 15u) - 8;
    return img < 0 ? 0 : img > 255 ? 255 : img;
}

#endif
