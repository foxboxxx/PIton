# perf: suite, profilers and compiler findings

Every kernel PIton ships, in one registry, with three ways to look at it:

| file | what it does |
|---|---|
| `suite.py` | 23 cases: tutorials (add, softmax, matmul), every primitive, the digits dense layer, both Mandelbrot kernels, the compute-peak recurrence. Inputs come from the `piton_rt.h` LCG (vectorised twin), so the Pi can regenerate them. |
| `test_perf_suite.py` | each case against numpy in the strict emulator (runs with the normal `pytest`) |
| `analyze.py` | dynamic instruction mix: every instruction weighted by how often it executes in the emulator; packing loss and why neighbouring single-ALU instructions did not pair |
| `profile_pi.py` | each case on the Pi at full size with the V3D performance counters (busy / TMU stall / idle / cache misses), sampled outputs checked |

```bash
.venv/bin/python -m pytest perf -q
.venv/bin/python perf/analyze.py                  # all cases; -v lists hottest regions
.venv/bin/python perf/analyze.py exp -v
.venv/bin/python perf/profile_pi.py               # needs the Pi (fresh boot)
PITON_PRESCHED=0 .venv/bin/python perf/profile_pi.py --label before   # A/B a compiler change
```

Results are written to `perf/results/` (`analysis*.json`, `pi-<date>*.json`).

## What the Pi found (2026-11-02, hardware counters and experiments)

| finding | evidence | change |
|---|---|---|
| Long straight-line programs are **instruction-fetch bound**: the QPU instruction cache holds ~500 instructions, and exp/log/sqrt/idiv at BLOCK=256 (800-4300 instructions) miss once per 32-instruction line (3.06% of fetches); those stalls are not counted as busy | `icache_experiment.py`: exp 2819 us unrolled vs 1731 us as a loop over 64-element chunks, misses 3.06% -> 0.01% | `piton/reroll.py`: 1-D elementwise blocks become a loop over 64-element chunks (halved until the loop fits) when the unrolled program exceeds 450 instructions |
| Elementwise kernels are **store bound**: one VDW DMA engine serves all 12 QPUs at ~0.18 us per DMA + ~0.85 ns per byte; with 4 VPM rows per QPU every DMA is 256 bytes (~640 MB/s) | `store_experiment.py`: store-only = copy = 1.6 ms per MiB, with or without the mutex, for 1024 or 512 programs; `vdw_probe.py`: 256/512/1024-byte DMAs reach 643/804/917 MB/s from 12 QPUs | none yet: only 64 VPM rows are addressable (`vdw_probe.capacity`, VPMBASE does not extend it), so larger DMAs need VPM slots shared between QPUs |
| **Launcher lost completions**: the 8-bit completed-program count was read only between bursts of issuing, so 256+ short programs could finish unseen and the launch timed out (the Mandelbrot 9,984 = 39 x 256 hang) | empty kernel: 0 of 1024 counted | `piton_launch` reads the count once per batch of issued requests (cannot wrap) and costs 0.74 us per program |

Pi timings, best of 5, before -> after all changes: exp 3041 -> 1730 us (1.76x), log 2888 -> 1706
(1.69x), exp2 2961 -> 1735 (1.71x), recip 2580 -> 1687 (1.53x), rsqrt 2425 -> 1649 (1.47x), sqrt
2299 -> 1641 (1.40x), idiv 13065 -> 7849 (1.66x), softmax 999 -> 908 (1.10x), add/mul 1990 -> 1880
(1.06x). Math kernels now sit at the store-engine floor (~1.65 ms per MiB written).

## What the emulator profile found (2026-11-02)

`pack` = (max(add ops, mul ops) + ldi + branches) / executed: 1.0 means every
add/mul pair that could share an instruction did.

1. **Dependence chains in transcendental code (fixed: `piton/backend/presched.py`).**
   exp/log/sqrt/rsqrt/recip lowered one 16-lane register at a time; allocation then
   reused the same accumulators for every chain, so 80-100% of unpaired neighbours
   were blocked by dependences. Interleaving chains before allocation (pressure-
   limited, kept only when it schedules shorter) cuts executed instructions
   2-10%: exp2 -10%, exp -6.6%, recip -6.3%, sqrt -5.2%, transpose -5.0%,
   idiv -4.5%, softmax -4.4%, matmul 8x64x4 -1.3%; nothing gets longer.
2. **Register-file read ports are now the main blocker.** An instruction can read
   one register from file A and one from B; two ops that each read an A and a B
   register can never pair, so pairing needs accumulators, small immediates or a
   shared operand. Mandelbrot: 94% of unpaired neighbours are port conflicts
   (10.5 instructions per 16-pixel iteration against a 7-op add-unit floor),
   softmax 61%. Rerolled kernels are back to mostly dependences (exp 69%): a
   64-element chunk has 4 independent chains instead of 16. Next step:
   port-aware allocation (decide A/B/accumulator with the likely pairs in view).
   Estimated ~1.4x on Mandelbrot's loop. `PITON_PORT_ALLOC=1` (regalloc.py,
   "Port-aware file choice"; off by default) reallocates with the pairs the
   scheduler missed in view and keeps the result when it schedules shorter:
   executed instructions mandel_32x16 -20.7%, mandel_16x16x2 -24.5% (loop
   309 -> 238 instructions, 7.4 per 16-pixel iteration), transpose -8.9%,
   softmax -8.0%, exp -6.8%, exp2 -5.8%, recip -4.4%, dense_relu -3.1%,
   idiv -2.7%; nothing gets longer, and the same TMU and VDW traffic.
3. **Elementwise address and mask arithmetic.** Each 16-element load takes ~6
   instructions: an element index and a byte address both advance per register,
   and every register is compared and masked even when the whole block is in
   bounds; ~50% of ALU ops are `mov`s. A versioned unmasked load path (stores
   already have one) and one induction variable would remove most of it. On the
   Pi these kernels are memory bound (`copy`: ~0.3 ms of issue time inside
   2.2 ms), so expect little wall-clock gain.
4. **Allocator copies:** ~32 per elementwise or transcendental kernel (3-7% of
   instructions) where both operands of an op landed in the same file.
   `PITON_PORT_ALLOC=1` weighs each conflict by how often it runs (copies
   move out of loops): exp 8 -> 1, transpose 30 -> 4, matmul 8x64x8 22 -> 13;
   add/mul/fma/rsqrt keep theirs, which pair for free there.
5. **matmul is at its structure bound** (pack 0.88-0.89): the A broadcast into
   r5 needs the mul unit once per four MACs, and the hardware run is limited by
   the B stream through the TMU, not by issue.
6. **Neighbours the scheduler left apart** (C11, `PITON_PAIR=1`, off by
   default; sched.py "Pairing"). Most "free" unpaired neighbours were not free:
   a mul-output rotation takes the B read port (matmul 8x64x8's A broadcasts
   next to TMU address adds that read B: 72% of its unpaired neighbours) and
   row_sum/row_max's were a register-file latency; analyze.py now says so. The
   losses that were the scheduler's: a move took the add ALU from a ready
   add-only op (elementwise and math kernels), the critical-path priority ran
   add-bound blocks out of add work (Mandelbrot: 225 add-only to 128 mul-only
   ops per loop body), each dot broadcast's add ALU stayed idle because the
   lowering pre-pairs every add with the next multiply, the port-aware
   allocation saw the pairs only one schedule missed and its greedy search
   ends where its first hints lead (mandel_32x16's Pi build stopped at a
   246-instruction loop, its emulator-size build at 232; a second search
   from the kept schedules' pairs finds 236), and a nop for a hazard on the
   way into a loop ran on every iteration (compute_peak, Mandelbrot). The
   shortest schedule is not always the fastest: matmul_8x64x4's ran 1-6%
   longer in the emulator at TMU latencies of 24-120 instructions, so a loop
   that streams through the TMU keeps the schedule that waits least
   (tmu_loop_cost; now 0.957-0.991). Executed instructions geomean 0.945, both at emulator
   size and for the Pi builds (recip 0.876, add 0.871, transpose 0.926,
   softmax 0.937, mandel_32x16 0.948, mandel_16x16x2 0.932, matmul
   0.991/0.985, dense_relu 0.984, the digits layers 0.96-0.975), never more;
   the paired share rises in every kernel (mandel_32x16 39 -> 49%, recip 30 ->
   48%). Scaled by the busy fractions of `pi-2026-11-04-d1once.json` that is
   ~1.03x on the Pi (mandelbrot 1.05-1.07x, recip 1.12x, rsqrt 1.08x, softmax
   1.06x; memory-bound kernels less than that); compiling takes ~3.4x the CPU
   time.
