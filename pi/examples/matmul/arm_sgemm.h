// ARM1176 CPU baseline for the QPU matmul: caches (via an identity-mapped MMU),
// a cycle counter, and two single-precision matmuls (C = A * B, row-major).
#ifndef ARM_SGEMM_H
#define ARM_SGEMM_H
#include <stdint.h>

// Identity-map the address space with the MMU so the L1 data cache can be used:
// the first 128 MB (program, stack, ARM-side matrices) is normal write-back
// memory, everything else (GPU memory, peripherals) stays strongly ordered, as
// with the MMU off.  Call only after all mailbox / QPU work: the runtime's
// mailbox buffer lives in ARM memory.
void arm_caches_on(void);
void arm_caches_off(void);

// cycle counter (PMU) and the measured ARM clock in MHz
uint32_t arm_cycles(void);
uint32_t arm_clock_mhz(void);

// straightforward i-k-j triple loop
void arm_matmul_naive(const float *A, const float *B, float *C, int M, int N, int K);

// blocked: packed A/B panels, 4x4 register-blocked micro-kernel, K blocks of
// ARM_KC and column blocks of ARM_NC sized for the 16 KB L1.  M, N multiples
// of 4.  pa: M*ARM_KC + 8 floats, pb: ARM_KC*ARM_NC + 8 floats.  vector_kernel:
// use the assembly micro-kernel in VFP short-vector mode (arm_kernel_vfp.S)
// instead of the compiled C one.
#ifndef ARM_KC
#define ARM_KC 64         // best of (KC, NC) in {32..256} x {8..64} at 512^3 on the Pi Zero
#endif
#ifndef ARM_NC
#define ARM_NC 32
#endif
void arm_matmul_blocked(const float *A, const float *B, float *C, int M, int N, int K,
                        float *pa, float *pb, int vector_kernel);
#endif
