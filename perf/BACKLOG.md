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
| B5 | TMU_NOSWAP for kernels that ask for it (B1's noswap part), so a slice's four QPUs share each physical TMU's cache | accepted, on by default (`PITON_TMU_NOSWAP=0` turns it off): `compile(..., tmu_noswap=True)` (the tutorial matmul: suite cases, matmul_pi, packed matmul, pi/examples/matmul, the primitives' op_matmul) writes TMU_NOSWAP=1 where the program is not coarsened and its requests are balanced, every other program that loads writes 0; matmul_pi 512^3 8.7-8.9 -> 10.2 GFLOPS, 1024^3 7.9-8.1 -> 9.35, packed 1.19-1.28x, suite matmul 1.10x. B1's prefetch / hoist / deep parts dropped (0.986-1.008x) |

## C. Instruction issue
| # | change | status |
|---|---|---|
| C1 | port-aware register allocation (file A / B / accumulator by likely pair) | accepted, on by default (`PITON_PORT_ALLOC=0` turns it off): mandel 1.22x / 1.27x, transpose 1.09x, softmax 1.04x |
| C2 | remove allocator copies where both operands landed in one file | accepted with C1 (copies only where the plain allocation was not shorter) |
| C3 | fill the 3 branch delay slots in loops (rerolled loops, Mandelbrot) | rejected: mandel 1.013x / 1.014x after C1, the rest within noise (branch int/c3-loop-slots, `PITON_LOOP_SLOTS`) |
| C4 | software pipelining of loop bodies | todo |
| C5 | unroll rerolled loops 2x when they still fit the icache | rejected with C3 |
| C6 | unmasked full-block loads + one induction variable | todo |
| C7 | MIR hoisting/CSE; pack ldtmu/ldunif signals onto ALU instructions | accepted with C8, on by default (`PITON_ISSUE=0` turns it off): MIR CSE, cheaper special-value checks, either-ALU ops on the mul ALU, flag groups in pre-scheduling; log 1.26x, log2 1.24x, softmax 1.21x, exp 1.20x, exp2 1.15x, idiv 1.12x |
| C8 | fill SFU latency with independent ops | accepted with C7 (`PITON_ISSUE`) |
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
| E1 | persistent kernels: 12 programs loop over the grid | accepted as coarsening, on by default (`PITON_PERSISTENT=0` turns it off, `=k` forces k): k consecutive ids per hardware program for kernels moving >= 4 bytes per instruction; add 1.21x, mul 1.24x, fma 1.31x, dense_relu 1.10x |
| E2 | skip the L2/slice-cache flush when inputs did not change | rejected for the loop: the flush is outside piton_launch's timed region, so no measured kernel changes (would only shorten full-call time) |
| E3 | prebuilt uniform streams; ARM D-cache on in the launch loop | rejected for the loop: uniform pointers are already built before the timed launch; the issue loop (0.74 us/program) overlaps execution for every suite case |
| E4 | E1's factor chosen per launch: the coarsened program reads its ids per hardware program from uniform coarsen0, and every launch path picks k from the launched grid by the compile-time rule, so one build runs small launches with one id per program | accepted, on by default (`PITON_COARSEN_RUNTIME=0` restores the compile-time factor): thresholds 64 ids / 32 hardware programs for ids that store >= 0.4 bytes per instruction, E1's 256 / 128 for the rest (E5); digits batch 1 1.19x, 16 1.31x, 64 1.08x, 256 1.02x on top of C7 |
| E5 | coarsen by what the ids store, not what they move: a Pi sweep of one build per kernel launched with each k (`perf/coarsen_sweep.py`) gains at >= 0.48 bytes stored per instruction (add, the digits' first layer) and loses at <= 0.32 (the digits' second and third layers, matmul) even on full grids | accepted, on by default (`PITON_COARSEN_STORES=0` coarsens by E1's moved-bytes rule alone): `COARSEN_MIN_STORE_BYTES = 0.4`; small: digits batch 64 1.024x (median of 4 pairs), 16 1.013x, the rest within 1% |

## F. Clocks (not in the loop)
F1 GPU clock above 250 MHz, F2 SDRAM clock: need a reboot and a stress test; excluded.

## G. Autotuning
G1 BLOCK / UNROLL / TILE_H / ROWS per kernel and Mandelbrot; G2 matmul tile sweep after
compiler changes; G3 on/off matrix of all switches.

## H. Algorithms and kernels
H1 fusion; H2 (accepted with H4, on by default, `PITON_MANDEL_FAST=0` restores the plain kernel: cardioid/bulb lanes, lane-flag exit checks, early loop exit; counts identical) Mandelbrot rectangle skipping, cardioid/bulb tests, periodicity; H3 reuse
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
| 2026-11-03 | E1 | coarsening on top of C1 + A1 + C10. First rule (k=4 for any short id) rejected: add 1.214x, mul 1.242x, fma 1.292x, dense_relu 1.119x but recip 0.975x, rsqrt 0.966x, sqrt 0.979x, idiv 0.940x (256 programs leave an uneven last round on 12 QPUs; `ab-2026-11-03-persistent-k4-rejected.json`). Coarsening only kernels that move >= 4 bytes per executed instruction (compute bound: 1.9-2.7, memory bound: 7.7-8.9): add 1649 -> 1367 us (1.206x), mul 1.236x, fma 1800 -> 1378 (1.306x), dense_relu 4331 -> 3926 (1.103x), every other case the same code or within 1%; host tests both ways (789), test_hw 11/11 with the switch on and with k forced to 4 (`ab-2026-11-03-persistent.json`) |
| 2026-11-03 | H2/H4 | fast Mandelbrot level 1 on top of C1 + A1 + C10 + E1: mandel_32x16 34872 -> 28141 us (1.239x), mandel_16x16x2 32531 -> 27989 (1.162x), counts identical to the plain kernel; test_hw 11/11; HDMI demo `--bench-only` on the Pi: every configuration exact, zoom 10000 frame 65.5 ms (92.8 ms on 11-02), 62-177x the ARM (`ab-2026-11-03-mandel_fast.json`, `benchmarks/results/mandelbrot-pi-2026-11-03-fast.log`) |
| 2026-11-03 | C3/C5 | loop delay slots + 2x unroll on top of C1 + A1 + C10 + E1: rejected, mandel 1.013x / 1.014x, everything else within +-1% (`ab-2026-11-03-loop_slots-rejected.json`); C1's allocation already shortened the loops it targeted |
| 2026-11-03 | profile | after C1 + A1 + C10 + E1 + H2: geometric mean 1.288x over the 23 cases against the morning baseline (`pi-2026-11-03-e1h2.json`): idiv 2.77x, mandel 1.47-1.51x, copy/scaled_relu 1.47x, recip/rsqrt/sqrt 1.40x, fma 1.38x, mul 1.35x, add 1.33x, exp/exp2 1.27-1.32x, dense_relu 1.27x, log/log2 1.19-1.20x, transpose 1.13x, matmul 1.04-1.08x, softmax 1.07x (D1 pending), row_sum/row_max/compute_peak unchanged |
| 2026-11-03 | D1/D2 | rerolled reductions and transposed tile loads on top of C1 + A1 + C10: both parts softmax 834 -> 487 us (1.713x) but transpose 769 -> 1080 (0.712x), rejected (`ab-2026-11-03-reroll_reduce-both-rejected.json`); rows only: softmax 844 -> 488 (1.730x), everything else the same code; host tests both ways (762), test_hw 12/12 (`ab-2026-11-03-reroll_reduce.json`) |
| 2026-11-03 | G2 | matmul sweep after C1 + A1 + C10 + E1 + H2 + D1, 8 tiles x groups 1/2/4 (`benchmarks/results/matmul-sweep-2026-11-03.jsonl`, all 48 correct): 8x64x8 still best; 512^3 group 2 8.99 GFLOPS (group 4 8.88, group 1 7.76), 1024^3 group 2 7.65 (group 4 7.53, group 1 7.64); group 2's edge is within boot noise, defaults unchanged. 1024^3 with each accepted switch off (`matmul-1024-switches-2026-11-03.log`): all on 7.68, store slots off 7.89, port allocation off 7.58, fast division off 7.53, all three off 7.42; the morning's 8.02 was not reproduced even with all off, so it was a faster boot. Store slots may cost ~2.7% at 1024 (rows 4 KiB apart): retest |
| 2026-11-03 | A1 rule | store slots only where a kernel stores >= 0.1 bytes per executed instruction (`SLOTS_MIN_STORE_BYTES`; `PITON_STORE_SLOTS=all` keeps them everywhere): with slots everywhere, matmul 8x64x8 at 1024^3 ran 7.55-7.66 vs 8.01-8.05 GFLOPS without (a slot store waits for its DMAs; 256^3 1.05x faster with slots, 512^3 even). With the rule 1024^3 7.89-8.02 vs 7.64-7.68 (1.03-1.05x), 512^3 8.88 vs 8.84-8.86; every suite case compiles to the same code (`ab-2026-11-03-store-slots-sparse.json`: transpose only, 0.995x, heap-order noise); host tests both ways (1007), test_hw 12/12 (`benchmarks/results/matmul-slots-sparse-2026-11-03.log`) |
| 2026-11-04 | A1 runtime | slot semaphores set up once per boot (`piton_launch_slots_once`; `PITON_SLOTS_ONCE=0` keeps init / fini around every launch, two one-program launches with cache clears, ~10 us each): digits batch 1 0.235 -> 0.218 ms (1.078x), batch 16 0.338 -> 0.318 (1.063x), full batch unchanged (9.289 ms); suite slot kernels 1.007-1.020x, nothing below 0.984x (`ab-2026-11-04-slots-once.json`); test_hw 12/12 |
| 2026-11-03 | digits | with C1 + A1 + C10 + E1 + H2 + D1 (`benchmarks/results/digits-pi-2026-11-03-all.{json,log}`): full batch 9.881 -> 9.281 ms (1.065x, 14.42x over ARM VFP), batch 256 1.05x, but batch 1 0.194 -> 0.235 ms and batch 16 0.243 -> 0.338: E1's compile-time k (the layers are compiled for the largest batch) halves small launches' hardware programs (PITON_PERSISTENT=0: 0.195 / 0.249 ms), and slot init/fini launches add ~20 us (PITON_STORE_SLOTS=0: 0.213 / 0.317). Fixes: once-per-boot slot setup (accepted), launch-time k (in progress) |
| 2026-11-04 | profile | after C1 + A1 (+ sparse-store rule, once-per-boot setup) + C10 + E1 + H2 + D1: geometric mean 1.328x over the 23 cases against the 11-03 baseline (`pi-2026-11-04-d1once.json`): idiv 2.78x, softmax 1.89x, mandel 1.47-1.51x, copy/scaled_relu 1.48x, recip/rsqrt/sqrt 1.41-1.42x, fma 1.40x, add/mul 1.34x, exp/exp2 1.28-1.33x, dense_relu 1.27x, log/log2 1.19-1.21x, transpose 1.16x, matmul 1.04-1.07x; row_sum/row_max/compute_peak unchanged (their usual 141-146 us) |
| 2026-11-04 | C7 | issue efficiency (`PITON_ISSUE`) on top of the 11-04 perf-loop: exp 1345 -> 1118 us (1.203x), exp2 1.149x, log 1443 -> 1144 (1.261x), log2 1.237x, recip 1.041x, rsqrt 1.043x, idiv 2814 -> 2521 (1.116x), softmax 478 -> 396 (1.207x), transpose 1.049x, mandel_32x16 1.017x; nothing below 0.984x (`ab-2026-11-04-issue.txt`; the JSON was lost when the machine restarted); host tests both ways (1079), test_hw 12/12. recip is coarsened by 4 with it (straight-line, 5.1 bytes per instruction): coarsening off vs on 0.997x, neutral |
| 2026-11-04 | E4 rule 1 | launch k by E1's thresholds (>= 256 ids, >= 128 hardware programs): digits batch 1 0.218 -> 0.183 ms (1.19x), 16 0.320 -> 0.238 (1.34x), 64 1.05x, full batch even, but 256 1.520 -> 1.550 (0.981x); suite the same code or within 2% (`ab-2026-11-04-coarsen_runtime-rule256.json`); test_hw 12/12 |
| 2026-11-04 | E4 sweep | one build launched with each k (`perf/results/coarsen-launch-2026-11-04.jsonl`): add k=4 1.21x from 128 ids (32 hardware programs), even at 64, 0.96-0.98x at 32-64 with k=4; dense_relu (digits layer 1, 0.48 bytes stored per instruction) k=2 1.11-1.15x from 64 ids, 0.96x at 32, 0.59-0.65x at 2-8; digits layer 2 (0.26) k=2 0.76-0.96x below its full grid and 0.988x on it (450); layer 3 (0.32) 0.73-0.97x, 0.849x on its full grid (225); matmul 8x64x8 (0.14) 0.79-0.93x at 32-128 ids; row_sum (run-time loop) 1.05x at 16-32 ids (43-89 us), 0.97x at 64. Every launch's outputs exact, untouched elements left as the fill |
| 2026-11-04 | E4 rule 2 | launch and compile-time thresholds 64 ids / 32 hardware programs for every kernel: rejected, the newly coarsened suite cases lose (row_sum 0.934x, row_max 0.946x, matmul_8x64x4 0.945x, matmul_8x64x8 0.941x at grids 64/128; `ab-2026-11-04-coarsen_runtime-rule64.json`); digits 1.18x / 1.32x / 1.07x / 0.983x / 1.004x (batch 1 / 16 / 64 / 256 / full); test_hw 12/12 |
| 2026-11-04 | E4 | launch k with per-kernel thresholds (64 ids / 32 hardware programs where ids store >= 0.4 bytes per instruction, E1's 256 / 128 elsewhere), on top of C7 (PITON_ISSUE=1): digits batch 1 0.2075 -> 0.174 ms (1.19x), 16 0.309 -> 0.2355 (1.31x), 64 0.537 -> 0.497 (1.08x), 256 1.493 -> 1.465 (1.02x), full 9.31 -> 9.39 (0.99x) (`benchmarks/results/digits-pi-2026-11-04-coarsen-runtime-{0,1}-{1,2}.log`); test_hw 12/12. Suite: only the coarsened cases change code (one uniform; dense_relu one instruction); best-of-5 swings +-3% on these memory-bound cases between boots (add 0.977x, then 1.024x on a recheck), so dense_relu and fma got 6 boots each way: dense_relu 0.989x (median of per-boot bests; best 0.983x), fma 1.001x (`ab-2026-11-04-coarsen_runtime{,-recheck,-recheck6}.json`). Accepted. Follow-up: a constant-k program for full-grid launches would recover dense_relu's ~1% |
| 2026-11-04 | E5 | coarsening only where ids store >= 0.4 bytes per instruction, on top of C7 + E4: no suite case changes code (`ab-2026-11-04-coarsen_stores.json`: transpose / log2 / sqrt / softmax differ only by heap-order compiles), matmul_pi 512^3 / 1024^3 the same (8.92-8.98, 7.96-8.07 GFLOPS: not coarsened at the base either); digits over 4 alternating pairs (median ratio / best ratio): batch 1 1.003 / 1.000, 16 1.013 / 1.018, 64 1.024 / 1.042, 256 0.991 / 1.003, full 1.003 / 0.989 (layer 2 runs its plain program: the sweep's 0.988x on its full grid); test_hw 12/12. Accepted on the batch 64 target (`benchmarks/results/digits-pi-2026-11-04-coarsen-stores-{0,1}-{1..4}.log`) |
| 2026-11-04 | B5 v1 | TMU_NOSWAP in every balanced program that is not coarsened, on top of C7 + E4 + E5: matmul 1.086-1.097x, matmul_pi 512^3 1.13-1.15x, 1024^3 1.16x, but row_max 0.926x and the digits network 0.935x (`benchmarks/results/tmu-noswap-2026-11-04-v1.log`, `ab-2026-11-04-tmu_noswap-v1.json`): rejected |
| 2026-11-04 | B5 v2 | only for dots, 0 written everywhere else: row_max / row_sum fixed (1.014x / 1.022x), matmul 1.095x, but digits still 0.94-0.95x (`-v2.log`, `-v2.json`). Each layer built plain, noswap 0 vs 1 (`benchmarks/results/tmu-noswap-2026-11-04-layers.log`): matmul_8x64x8 1.111x, dense_relu (2 column tiles) 1.041x, digits layer 2 (1 column tile) 0.867x, layer 3 0.942x: noswap pays where programs running together on a slice load the same A rows, which the compiler cannot see, and E5 leaves layer 2 plain. Rejected |
| 2026-11-04 | B5 v3 | per-kernel request (`tmu_noswap=True`, the tutorial matmul), 0 elsewhere: suite matmul_8x64x4 1.098x, matmul_8x64x8 1.099x, everything else within 2% except mul 0.979x / row_sum 0.951x on one sample each, which 6 boots each way put at 0.994x / 0.993x (median of per-boot bests; row_max 1.007x; `ab-2026-11-04-tmu_noswap{,-recheck6}.json`); matmul_pi 512^3 8.74 / 8.94 -> 10.17 / 10.21 GFLOPS, 1024^3 7.92 / 8.06 -> 9.35 / 9.35; packed matmul 512^3 7.72 -> 9.24 / 9.92, 1024^3 7.22 / 7.23 -> 8.57 / 8.56; digits even (`-v3.log`); test_hw 12/12, host tests both ways (1249 on opt/noswap). Accepted |
