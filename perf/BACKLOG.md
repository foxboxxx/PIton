# Optimisation backlog and loop log

The loop works on branch `perf-loop` (never `main`): every change is switchable
with a `PITON_*` variable, A/B'd on the Pi with `perf/ab.py`, and committed on
its own (with its `perf/results/ab-*.json`) only when accepted, so a bad change
can be reverted one commit at a time.

**Acceptance:** host tests (`.venv/bin/python -m pytest -q tests perf`) pass with
the switch on and off; `PITON_HW=1 pytest tests/test_hw.py` passes with it on;
best-of-5 on the Pi > 2% faster on a target case; no case > 2% slower.

**Hangs:** the runner power-cycles the Pi through the HUGE board
(`piton/runtime/power.py`, `tools/pi_power`) whenever the bootloader does not
answer, so risky items (A1, A2, B3, E1) can run unattended. F1/F2 (clocks) stay
out of the loop.

Status: `todo`, `wip`, `accepted <commit>`, `rejected (<why>)`.

## A. Store path (limit for ~15 of 23 kernels)
| # | change | status |
|---|---|---|
| A1 | shared 16-row VPM slots under semaphores: 1 KiB store DMAs (probe 917-966 vs 641 MB/s) | wip (batch 1, opt/a1-store-slots) |
| A2 | double-buffer the slots: start the DMA, compute the next chunk, then wait | wip (with A1) |
| A3 | store through the L2-bypassing bus alias | rejected: with the MMU off the ARM sees GPU memory only through the L2-cached alias (cpu = bus - 0x40000000), so its NaN fill can sit dirty in L2 and later overwrite or hide uncached VDW writes, and the ARM cannot clean the VC L2; streaming kernels also gain nothing (1 MiB inputs never fit the 128 KB L2) |
| A4 | larger BLOCK per program (fewer partial/tail DMAs) | done d1da2e0: blocks over ~512 elements now reroll instead of failing allocation; add 1.07x, mul 1.09x, fma 1.08x at BLOCK 1024 (perf/tune.py) |

## B. Load path
| # | change | status |
|---|---|---|
| B1 | both texture units (TMU0/TMU1 alternating) | wip (batch 1, opt/b1-tmu, with B2) |
| B2 | deeper TMU prefetch | wip (with B1) |
| B3 | bulk loads by VPM DMA read (VCD) for matmul's B | todo |
| B4 | fewer uniforms per program | todo |

## C. Instruction issue
| # | change | status |
|---|---|---|
| C1 | port-aware register allocation (file A / B / accumulator by likely pair) | wip (batch 1, opt/c1-port-alloc, with C2) |
| C2 | remove allocator copies where both operands landed in one file | wip (with C1) |
| C3 | fill the 3 branch delay slots in loops (rerolled loops, Mandelbrot) | wip (batch 1, opt/c3-loop-slots, with C5) |
| C4 | software pipelining of loop bodies | todo |
| C5 | unroll rerolled loops 2x when they still fit the icache | wip (with C3) |
| C6 | unmasked full-block loads + one induction variable | todo |
| C7 | MIR hoisting/CSE; pack ldtmu/ldunif signals onto ALU instructions | todo |
| C8 | fill SFU latency with independent ops | todo |
| C9 | opt-in fast-math (fewer Newton steps / shorter polynomials) | todo |
| C10 | integer divide via float reciprocal + one correction | wip (batch 2, opt/c10-idiv) |

## D. Instruction cache
| # | change | status |
|---|---|---|
| D1 | reroll row reductions (carry partial max/sum across chunks) | wip (batch 1, opt/d1-reroll-reduce, with D2) |
| D2 | reroll 2-D tiles | wip (with D1) |
| D3 | autotune the reroll threshold (450) and chunk size | rejected: SUB 64 is best (32 is 1.4x slower for exp/log/sqrt; 128 gives the same code or time) |
| D4 | align hot loops to 32-instruction cache lines | deferred: hot loops already fit the ~500-instruction cache, so alignment only affects their first iteration |

## E. Launch and runtime
| # | change | status |
|---|---|---|
| E1 | persistent kernels: 12 programs loop over the grid | wip (batch 2, opt/e1-persistent) |
| E2 | skip the L2/slice-cache flush when inputs did not change | rejected for the loop: the flush is outside piton_launch's timed region, so no measured kernel changes (would only shorten full-call time) |
| E3 | prebuilt uniform streams; ARM D-cache on in the launch loop | rejected for the loop: uniform pointers are already built before the timed launch; the issue loop (0.74 us/program) overlaps execution for every suite case |

## F. Clocks (not in the loop)
F1 GPU clock above 250 MHz, F2 SDRAM clock: need a reboot and a stress test; excluded.

## G. Autotuning
G1 BLOCK / UNROLL / TILE_H / ROWS per kernel and Mandelbrot; G2 matmul tile sweep after
compiler changes; G3 on/off matrix of all switches.

## H. Algorithms and kernels
H1 fusion; H2 (wip, batch 2, opt/h2-mandel, with H4) Mandelbrot rectangle skipping, cardioid/bulb tests, periodicity; H3 reuse
pixels when panning; H4 escape-check cadence, regroup live lanes; H5 opt-in u8/16-bit
outputs; H6 matmul register blocking for more A reuse.

## Log
| date | item | result |
|---|---|---|
| 2026-11-03 | infra | power switch + auto-recovery + `perf/ab.py` (905a3f6) |
| 2026-11-03 | baseline | 23 cases, `perf/results/pi-2026-11-03-baseline{,2}.json`; boot-to-boot noise mostly <0.5%, up to 5% (sqrt), so `ab.py` re-runs results outside +-2% (fd5b179) |
| 2026-11-03 | A4/G1 | BLOCK sweeps: 2-input kernels 6-9% faster at BLOCK>=1024, 1-input 0-4%; suite kept at BLOCK 256 so comparisons stay fixed (d1da2e0) |
| 2026-11-03 | G1 | mandel UNROLL: 1 -> 162.8 ms, 16 (suite) 42.4 ms, 22 41.96 ms (-1%), 32 63 ms (icache); row_sum/row_max BLOCK_M 4 -> 2: 144 -> 129 us (1.11x, 128 programs balance the 12 QPUs better) |
| 2026-11-03 | D3 | reroll SUB sweep: no gain |
| 2026-11-03 | G1/G2 | dense_relu BLOCK_SIZE_K 8 -> 4: 5040 -> 4400 us (1.145x); GROUP_SIZE_M>1 is slower. matmul_8x64x8 GROUP_SIZE_M (never swept before, the Oct 30 sweep fixed 4): 256^3 group 1 3450 vs 4148 us (1.20x); 512^3 group 2 8.858 GFLOPS vs 8.618 (group 4), group 1 7.95 and noisy; 1024^3 group 2 = group 4 (8.02 GFLOPS). Best grouping depends on size: an autotuner should pick it |
| 2026-11-03 | G1 (app) | digits MLP dense tiles BK 8 -> 4: full batch 10.859 -> 9.881 ms (1.099x, 13.54x over ARM VFP, 181,864 images/s), batch 256 1.10x, batch 1 unchanged; 1743/1797 correct, all 28,752 logits within bounds |
