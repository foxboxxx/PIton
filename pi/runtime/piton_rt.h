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

// piton_launch for kernels that store through shared VPM slots
// (PITON_STORE_SLOTS, piton/backend/lower.py). Hardware semaphores keep their
// counts across launches and are 0 after reset: `init` (one program that
// releases each slot semaphore) runs to completion first, then the n
// programs, then `fini` (acquires each), so the semaphores end at 0 again.
// Both come from piton.runtime.launcher.semaphore_programs. Returns the total
// time in microseconds.
uint32_t piton_launch_slots(uint32_t code, const uint32_t *unifs, uint32_t n,
                            uint32_t init, uint32_t fini);

// piton_launch_slots without the per-launch init and fini (the default; PITON_SLOTS_ONCE=0: the other):
// `init` runs before the first slot launch after boot, and the semaphores stay
// at 1 between launches. Saves two one-program launches and their cache
// clears per call (~10 us each), which short launches notice.
uint32_t piton_launch_slots_once(uint32_t code, const uint32_t *unifs, uint32_t n, uint32_t init);

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

// Clock rates from the firmware (mailbox): current, or the maximum the
// firmware allows (config.txt).  Ids: PITON_CLOCK_ARM, _CORE, _V3D, _SDRAM.
#define PITON_CLOCK_ARM   3
#define PITON_CLOCK_CORE  4
#define PITON_CLOCK_V3D   5
#define PITON_CLOCK_SDRAM 8
uint32_t piton_clock_hz(uint32_t clock_id, int max);
void piton_memory_split(uint32_t *arm_size, uint32_t *vc_base, uint32_t *vc_size);

// Send a complete property-interface message (channel 8): `msg` must be
// 16-byte aligned and the ARM data cache off. Returns nonzero if the firmware
// processed it; check each tag's response bit for individual results.
int piton_mbox_property(volatile uint32_t *msg);

// V3D performance counters (Reference Guide section 10, counter ids as in
// Linux vc4). QPU counts are summed over all QPUs in V3D clock cycles; the
// 32-bit counters wrap after ~1.4 s of full-GPU time, so measure short spans.
enum {
    PITON_PCTR_QPU_IDLE = 13,           // cycles with no program
    PITON_PCTR_QPU_VALID = 16,          // cycles executing valid instructions
    PITON_PCTR_QPU_TMU_STALL = 17,      // cycles waiting for TMU results
    PITON_PCTR_QPU_SCOREBOARD = 18,     // cycles waiting for the scoreboard
    PITON_PCTR_ICACHE_HIT = 20,
    PITON_PCTR_ICACHE_MISS = 21,
    PITON_PCTR_UNIF_HIT = 22,
    PITON_PCTR_UNIF_MISS = 23,
    PITON_PCTR_TMU_QUADS = 24,
    PITON_PCTR_TMU_MISS = 25,
    PITON_PCTR_VDW_STALL = 26,          // cycles the VDW waited for the VPM
    PITON_PCTR_L2_HIT = 28,
    PITON_PCTR_L2_MISS = 29,
};
#define PITON_PCTR_MAX 16
// Select up to 16 events, clear and enable their counters.
void piton_perf_start(const uint8_t *events, unsigned n);
void piton_perf_read(uint32_t *counts, unsigned n);

#endif
