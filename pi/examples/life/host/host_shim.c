// Host stand-in for the Pi runtime, for examples/life_pi.py --dry-run: GPU
// memory is a static arena, and the three QPU launchers run the ARM code of
// life_arm.c on it (life_arm_naive for life_tile and life_rows, life_arm_bits
// for life_bits). This checks the firmware's flow, checks and log format,
// not the QPU kernels (the emulator tests do that).
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "rpi.h"
#include "piton_rt.h"
#include "arm_sgemm.h"
#include "life_arm.h"
#include "life_tile.h"
#include "life_rows.h"
#include "life_bits.h"

#define ARENA_BYTES (64u << 20)
static uint8_t arena[ARENA_BYTES] __attribute__((aligned(4096)));
static uint32_t arena_used;

static void *host_ptr(uint32_t bus) {
    if (bus < PITON_GPU_BASE || bus - PITON_GPU_BASE >= arena_used) {
        fprintf(stderr, "host shim: bus address %x outside the arena\n", bus);
        exit(2);
    }
    return arena + (bus - PITON_GPU_BASE);
}

uint32_t timer_get_usec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)((uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u);
}

void clean_reboot(void) { exit(1); }

void piton_init(void) { printf("LIFE_HOST simulation: QPU launches run the ARM code\n"); }

uint32_t piton_qpu_count(void) { return 12; }

uint32_t piton_clock_hz(uint32_t clock_id, int max) {
    (void)max;
    switch (clock_id) {
    case PITON_CLOCK_ARM: return 700000000u;
    case PITON_CLOCK_SDRAM: return 400000000u;
    default: return 250000000u;
    }
}

piton_buf_t piton_alloc(uint32_t size) {
    piton_buf_t b;
    size = (size + 4095) & ~4095u;
    if (size > ARENA_BYTES - arena_used) {
        fprintf(stderr, "host shim: arena full\n");
        exit(2);
    }
    b.handle = 1;
    b.bus = PITON_GPU_BASE + arena_used;
    b.cpu = arena + arena_used;
    b.size = size;
    arena_used += size;
    return b;
}

void piton_free(piton_buf_t *b) { (void)b; }

void arm_caches_on(void) {}
void arm_caches_off(void) {}

static uint32_t elapsed_since(uint32_t start) { return timer_get_usec() - start; }

uint32_t life_tile_launch(const life_tile_args_t *a, unsigned gx, unsigned gy, unsigned gz) {
    (void)gx; (void)gy; (void)gz;
    uint32_t start = timer_get_usec();
    life_arm_naive(host_ptr(a->src_ptr), host_ptr(a->dst_ptr), a->n_rows, a->n_cols);
    return elapsed_since(start);
}

uint32_t life_rows_launch(const life_rows_args_t *a, unsigned gx, unsigned gy, unsigned gz) {
    (void)gx; (void)gy; (void)gz;
    uint32_t start = timer_get_usec();
    life_arm_naive(host_ptr(a->src_ptr), host_ptr(a->dst_ptr), a->n_rows, a->n_cols);
    return elapsed_since(start);
}

uint32_t life_bits_launch(const life_bits_args_t *a, unsigned gx, unsigned gy, unsigned gz) {
    static uint32_t scratch[6 * 4096];
    (void)gx; (void)gy; (void)gz;
    if (a->n_words > 4096) {
        fprintf(stderr, "host shim: rows wider than 4096 words\n");
        exit(2);
    }
    uint32_t start = timer_get_usec();
    life_arm_bits(host_ptr(a->src_ptr), host_ptr(a->dst_ptr), a->n_rows, 32 * a->n_words, scratch);
    return elapsed_since(start);
}

void notmain(void);

int main(void) {
    notmain();
    return 0;
}
