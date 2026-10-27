// PIton bare-metal runtime for the VideoCore IV QPUs (Pi Zero / Pi 1).
//
// Mailbox memory allocation + direct V3D user-program queue launches.
// The mailbox / V3D register sequence follows the CS240LX GPU lab
// (labs/16-gpu, Rohan Chanani and Asanshay Gupta).
#ifndef PITON_RT_H
#define PITON_RT_H

#include <stdint.h>

#define PITON_GPU_BASE 0x40000000u   // bus alias (L2 cached) of ARM physical 0

#ifndef PITON_LAUNCH_TIMEOUT_US
#define PITON_LAUNCH_TIMEOUT_US (20 * 1000 * 1000)
#endif

typedef struct {
    uint32_t handle;      // mailbox handle
    uint32_t bus;         // GPU bus address
    volatile void *cpu;   // ARM address
    uint32_t size;
} piton_buf_t;

// Power on the QPUs. Panics if the firmware has V3D disabled (e.g. gpu_mem=16
// or start_cd.elf).
void piton_init(void);
void piton_shutdown(void);

// Allocate `size` bytes of GPU memory (page aligned, L2 cached).
piton_buf_t piton_alloc(uint32_t size);
void piton_free(piton_buf_t *b);

// Run `n` program instances of the code at bus address `code` through the
// V3D user-program request queue. `unifs[i]` is the bus address of program
// i's uniforms. The hardware scheduler hands each request to the next idle
// QPU. Returns the elapsed time in microseconds.
uint32_t piton_launch(uint32_t code, const uint32_t *unifs, uint32_t n);

// Bus address of a pointer inside a piton_buf_t.
static inline uint32_t piton_bus(const piton_buf_t *b, const volatile void *p) {
    return b->bus + (uint32_t)((const volatile char *)p - (const volatile char *)b->cpu);
}

// Deterministic test data shared with the Python harness (piton.runtime.pi).
static inline uint32_t piton_lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}
static inline float piton_lcg_float(uint32_t *s) {   // [-1, 1) with 24-bit mantissa
    uint32_t r = piton_lcg(s) >> 8;
    return (float)r * (1.0f / 8388608.0f) - 1.0f;
}

uint32_t piton_qpu_count(void);

#endif
