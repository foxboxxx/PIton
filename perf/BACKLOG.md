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
| A1 | shared 16-row VPM slots under semaphores: 1 KiB store DMAs (probe 917-966 vs 641 MB/s) | todo |
| A2 | double-buffer the slots: start the DMA, compute the next chunk, then wait | todo |
| A3 | store through the L2-bypassing bus alias | todo |
| A4 | larger BLOCK per program (fewer partial/tail DMAs) | todo |

## B. Load path
| # | change | status |
|---|---|---|
| B1 | both texture units (TMU0/TMU1 alternating) | todo |
| B2 | deeper TMU prefetch | todo |
| B3 | bulk loads by VPM DMA read (VCD) for matmul's B | todo |
| B4 | fewer uniforms per program | todo |

## C. Instruction issue
| # | change | status |
|---|---|---|
| C1 | port-aware register allocation (file A / B / accumulator by likely pair) | todo |
| C2 | remove allocator copies where both operands landed in one file | todo |
| C3 | fill the 3 branch delay slots in loops (rerolled loops, Mandelbrot) | todo |
| C4 | software pipelining of loop bodies | todo |
| C5 | unroll rerolled loops 2x when they still fit the icache | todo |
| C6 | unmasked full-block loads + one induction variable | todo |
| C7 | MIR hoisting/CSE; pack ldtmu/ldunif signals onto ALU instructions | todo |
| C8 | fill SFU latency with independent ops | todo |
| C9 | opt-in fast-math (fewer Newton steps / shorter polynomials) | todo |
| C10 | integer divide via float reciprocal + one correction | todo |

## D. Instruction cache
| # | change | status |
|---|---|---|
| D1 | reroll row reductions (carry partial max/sum across chunks) | todo |
| D2 | reroll 2-D tiles | todo |
| D3 | autotune the reroll threshold (450) and chunk size | todo |
| D4 | align hot loops to 32-instruction cache lines | todo |

## E. Launch and runtime
| # | change | status |
|---|---|---|
| E1 | persistent kernels: 12 programs loop over the grid | todo |
| E2 | skip the L2/slice-cache flush when inputs did not change | todo |
| E3 | prebuilt uniform streams; ARM D-cache on in the launch loop | todo |

## F. Clocks (not in the loop)
F1 GPU clock above 250 MHz, F2 SDRAM clock: need a reboot and a stress test; excluded.

## G. Autotuning
G1 BLOCK / UNROLL / TILE_H / ROWS per kernel and Mandelbrot; G2 matmul tile sweep after
compiler changes; G3 on/off matrix of all switches.

## H. Algorithms and kernels
H1 fusion; H2 Mandelbrot rectangle skipping, cardioid/bulb tests, periodicity; H3 reuse
pixels when panning; H4 escape-check cadence, regroup live lanes; H5 opt-in u8/16-bit
outputs; H6 matmul register blocking for more A reuse.

## Log
| date | item | result |
|---|---|---|
| 2026-11-03 | infra | power switch + auto-recovery + `perf/ab.py` (905a3f6) |
