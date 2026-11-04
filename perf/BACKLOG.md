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
| A1 | shared 16-row VPM slots under semaphores: 1 KiB store DMAs (probe 917-966 vs 641 MB/s) | accepted, on by default (`PITON_STORE_SLOTS=0` turns it off): copy 1.48x, sqrt 1.40x, recip 1.39x, exp 1.24x, add 1.13x |
| A2 | double-buffer the slots: start the DMA, compute the next chunk, then wait | not built: 4 slots for 12 QPUs, the VDW engine stays saturated; halving the DMA or the slot count would cost more than the overlap gains |
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
| C1 | port-aware register allocation (file A / B / accumulator by likely pair) | accepted, on by default (`PITON_PORT_ALLOC=0` turns it off): mandel 1.22x / 1.27x, transpose 1.09x, softmax 1.04x |
| C2 | remove allocator copies where both operands landed in one file | accepted with C1 (copies only where the plain allocation was not shorter) |
| C3 | fill the 3 branch delay slots in loops (rerolled loops, Mandelbrot) | wip (batch 1, opt/c3-loop-slots, with C5) |
| C4 | software pipelining of loop bodies | todo |
| C5 | unroll rerolled loops 2x when they still fit the icache | wip (with C3) |
| C6 | unmasked full-block loads + one induction variable | todo |
| C7 | MIR hoisting/CSE; pack ldtmu/ldunif signals onto ALU instructions | todo |
| C8 | fill SFU latency with independent ops | todo |
| C9 | opt-in fast-math (fewer Newton steps / shorter polynomials) | todo |
| C10 | integer divide via float reciprocal + one correction | accepted, on by default (`PITON_FAST_IDIV=0` restores the branching division): idiv 2.78x, dense_relu 1.15x, matmul 1.03-1.05x |

## D. Instruction cache
| # | change | status |
|---|---|---|
| D1 | reroll row reductions (carry partial max/sum across chunks) | accepted, on by default (`PITON_REROLL_REDUCE=rows`; `0` turns it off): softmax 1.73x |
| D2 | reroll 2-D tiles | built as transposed loads (`PITON_REROLL_REDUCE=tiles`), off: transpose 0.71x on the Pi (each request's 16 lanes read 16 rows 1 KiB apart) |
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
| 2026-11-03 | C1/C2 | port-aware allocation: mandel_32x16 42397 -> 34873 us (1.216x), mandel_16x16x2 41139 -> 32529 (1.265x), transpose 889 -> 813 (1.093x), softmax 879 -> 844 (1.041x), exp/exp2/log 1.02x, nothing slower than 0.993x; host tests both ways, test_hw 10/10 (`ab-2026-11-03-port_alloc.json`) |
| 2026-11-03 | A1 | store slots on top of C1: copy 1652 -> 1118 us (1.478x), scaled_relu 1.482x, sqrt 1.404x, recip 1.393x, rsqrt 1.379x, exp2 1.272x, exp 1.243x, log2 1.187x, log 1.164x, add 1.129x, mul 1.091x, fma 1.059x, matmul_8x64x8 1.033x; softmax 0.981x, transpose 0.996x (within the 2% rule); test_hw 10/10 (`ab-2026-11-03-store_slots.json`) |
| 2026-11-03 | profile | after C1 + A1: geometric mean 1.165x over the 23 cases against the morning baseline (`pi-2026-11-03-c1a1.json`, `perf/compare.py`); exp/log/recip/rsqrt now 87-89% busy (compute bound), copy at the store engine's ~940 MB/s, add/mul/fma 22% busy (load side). row_sum/row_max 142/146 us is their usual range (141-146 in every profile since 11-02; the 138/139 baseline was a fast boot) |
| 2026-11-03 | G1 | BLOCK sweep with slots: exp 256 -> 1356 us, 512 1695, 1024 1676; sqrt 256 1175, 512 1725, 1024 1638. Blocks above 256 lose the slots (the rerolled store queue holds one 16-row slot, at most 4 chunks), so 256 stays best (`tune-2026-11-03-{exp,sqrt}-BLOCK-c1a1.json`) |
| 2026-11-03 | C10 | branch-free exact division (`//` and `%` share one estimate, constant divisors take shortcuts), on top of C1 + A1: idiv 7836 -> 2824 us (2.775x), dense_relu 4989 -> 4354 (1.146x), matmul_8x64x8 1.048x, matmul_8x64x4 1.028x, transpose 1.024x, nothing else changes; host tests both ways (684), test_hw 11/11 (`ab-2026-11-03-fast_idiv.json`) |
| 2026-11-03 | D1/D2 | rerolled reductions and transposed tile loads on top of C1 + A1 + C10: both parts softmax 834 -> 487 us (1.713x) but transpose 769 -> 1080 (0.712x), rejected (`ab-2026-11-03-reroll_reduce-both-rejected.json`); rows only: softmax 844 -> 488 (1.730x), everything else the same code; host tests both ways (762), test_hw 12/12 (`ab-2026-11-03-reroll_reduce.json`) |
