# PIton: compiling Triton for the Raspberry Pi's GPU — project report

This report explains how PIton was built, checked and optimised. It is written for readers who know
what a GPU and a compiler are but have not read this codebase. Every number below comes from a file in
the repository (named next to it) or from the git history. Where the records do not settle a question,
the report says so.

Contents: [1 Goal and result](#1-goal-and-result) · [2 The hardware](#2-the-hardware) ·
[3 The compiler](#3-the-compiler-pipeline) · [4 Verification](#4-verification) ·
[5 Running on the Pi](#5-running-on-the-pi) · [6 The optimisation loop](#6-the-optimisation-loop-2026-11-03-to-2026-11-05) ·
[7 Results](#7-results) · [8 Showcase examples](#8-showcase-examples) ·
[9 Limitations, future work, lessons](#9-limitations-future-work-and-lessons)

---

## 1. Goal and result

*PIton takes ordinary Triton kernels written in Python and runs them on the 12 QPUs of a Raspberry Pi
Zero W. The output is bit-identical to a strict emulator, and the measured speed is reported against
tuned ARM code on the same board.*

[Triton](https://github.com/triton-lang/triton) is a Python language for GPU kernels. A Triton program
works on blocks ("tiles") of values instead of single threads. PIton accepts the same source,
`@triton.jit` kernels using `tl.load`, `tl.store`, `tl.dot`, `tl.sum` and the rest, and compiles it
for the **VideoCore IV 3D block of the BCM2835** in the Pi Zero / Pi 1. That GPU has 12 QPUs (quad
processing units), each a 16-lane fp32 SIMD processor, clocked at 250 MHz. The compiler, an emulator
of the GPU, a bare-metal runtime and the benchmark harness are all in this repository.

```python
import piton as triton            # the same kernel also runs under real Triton
import piton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)               # strict emulator
add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256, device="pi")  # the real Pi, over UART
```

**Headline results** (Pi Zero W: ARM 700 MHz, V3D 250 MHz, SDRAM 400 MHz):

| what | result | source |
|---|---|---|
| 23-kernel performance suite, geometric-mean speedup over the 2026-11-03 baseline, same board | **1.424×** in the default configuration, with every accepted optimisation including scheduler pairing (1.406× without pairing) | `perf/results/pi-2026-11-03-baseline.json`, `pi-2026-11-05-final-sameboard.json`, `ab-2026-11-05-pair.json` |
| Triton tutorial matmul (8×64×8 tiles), square N³ | **9.38 / 10.15 / 9.34 GFLOPS** at 128³ / 512³ / 1024³ (final configuration, same board) | `benchmarks/results/matmul-pi-2026-11-05-final.log` |
| peak fp32 rate (12 QPUs × 16 lanes × one add + one multiply per instruction) | 24 GFLOPS; **23.794 GFLOPS (99.14%)** reached by register-only arithmetic | `benchmarks/results/final-summary-2026-10-30.json` |
| trained 64→128→64→10 digit classifier, 1,797 held-out images | **96.995%** accuracy; full batch **9.320 ms** vs 133.675 ms tuned ARM VFP (**14.3×**) | `benchmarks/results/digits-pi-2026-11-05-final-sameboard.json` |
| 16 Triton primitives against 42 ARM baselines, 58 rows, all checked | exp **35.6×**, softmax **32.5×**, matmul 256³ **12.1×**, copy **6.0×** over the best CPU version | `benchmarks/results/primitives-2026-11-05.log` |
| hardware tests (`PITON_HW=1 pytest tests/test_hw.py`) | **12/12**, outputs bit-identical to the emulator | STATUS.md |

**Timeline** (from `git log --reverse`):

| dates | phase |
|---|---|
| 2026-10-26 | First commit: the whole pipeline (frontend, tile IR, lowering, allocator, scheduler, encoder, emulator, Pi runner, hardware probes; 39 files, about 10,600 lines). VDW DMA kicks serialised on the shared stride register. First Pi matmul benchmark. |
| 2026-10-27 | Review fixes. Deterministic float math. Emulation of the measured float behaviour. Exact 32-bit integer division. Hardware tests. Tuned ARM1176 baselines. Tile sweep. Pushed to GitHub. |
| 2026-10-30 | Scheduler tuning (branch-delay slots, cached A rotations), the finished tile sweep, the compute-peak benchmark, and verified digit-network inference. |
| 2026-11-02/03 | Profiling with the V3D performance counters: the 23-case `perf/` suite, instruction-cache rerolling, pre-allocation scheduling, a launcher fix, the HDMI Mandelbrot demo and the primitives benchmark. |
| 2026-11-03 to 11-05 | The optimisation loop (section 6): every backlog item measured on the Pi and accepted or rejected by one rule. |

The history has commits under three author names: `AdamAlhousiki` (191), `Joe Robertazzi` (6: the
11-03 profiling, perf suite, Mandelbrot and primitives commits) and `PIton Agent` (8, on 2026-11-05).

---

## 2. The hardware

*The QPU is a 16-lane, in-order SIMD core that issues two ALU operations per instruction. It has a
very small instruction cache, and one store engine is shared by all 12 cores. Several limits that
shaped the compiler had to be measured, because the documentation was wrong or silent.*

### 2.1 What a QPU is

| feature | what it means for the compiler |
|---|---|
| **16-lane SIMD**, one instruction every 4 clocks (62.5 M instructions/s per QPU, measured; 12 QPUs scale linearly) | A tile of N elements lives in ⌈N/16⌉ registers. The peak is 12 × 16 × 2 × 62.5 M = 24 GFLOPS. |
| **Two ALUs per 64-bit instruction**: an *add* ALU (fadd, integer ops, compares) and a *mul* ALU (fmul, 24-bit integer multiply, output rotation), plus a signal field (e.g. `ldtmu0`: receive a load) or a small immediate | Peak needs both ALUs busy in every instruction. Pairing an add-ALU op with a mul-ALU op is the scheduler's main job. |
| **Registers**: accumulators r0-r3 (readable by the next instruction), r4 (TMU/SFU results), r5 (per-lane broadcast), and two 32-entry register files A and B, whose values can be read only from the second instruction after the write | **One read port per file.** Two operands of one instruction must come from different files unless one is an accumulator. A small immediate, or a mul-ALU rotation, uses the B port. |
| **TMU** (texture and memory unit) loads: a gather of 16 independent addresses per request, received into r4 by `ldtmu`; two units, TMU0 and TMU1 | Loads are asynchronous (about 12 cycles at best, measured), so the latency must be hidden by issuing requests early. |
| **VPM** (vertex pipe memory), a scratchpad shared by all QPUs, and the **VDW** DMA that writes VPM rows to memory | Every store goes register → VPM row → DMA. One DMA engine serves all 12 QPUs. |
| **SFU**: recip, recipsqrt, exp2, log2, result in r4 three instructions later | Not used: its results are approximate and not reproducible (below). |
| **Synchronisation**: 16 shared 4-bit semaphores and one hardware mutex | Used to share the VPM between QPUs (store slots, section 6). |
| **Uniforms**: a per-program stream of 32-bit values (the kernel arguments); `unif_addr` rewinds it | Arguments needed late are re-read instead of held in registers: a free spill slot. |
| **Branches** on all/any lane flags, with **3 delay slots** | Branches are uniform across lanes. Per-lane decisions use a condition code on each op. |
| **Launch**: a 16-entry user-program request queue (code address + uniforms address per program) | `piton_launch` feeds it from the ARM (section 5). |

### 2.2 What the probes measured

`tools/probes.py`, `tools/float_specials_probe.py`, `perf/vdw_probe.py` and `perf/icache_experiment.py`
run hand-written QPU code on the board. The emulator models every finding (README, "Things we learned
from the hardware"):

| finding | consequence |
|---|---|
| `fadd`/`fsub`/`fmul`/`itof` **round toward zero** on the exact result. Results below 2⁻¹²⁶ and denormal operands become +0. NaN operands act as ±inf, `0 × inf = +0`, and `inf + −inf` returns the second operand. | The emulator reproduces every result bit for bit (sums are truncated exactly using TwoSum). Float comparisons use integer keys so they follow IEEE. |
| SFU `exp2` ≈ 13.8 bits, `log2` ≈ 12.8 bits; recip/rsqrt bits not reproducible (softmax differed from the emulator in the last bit) | exp/log use range reduction and polynomials on the ALUs; recip/rsqrt use an integer seed and Newton steps. Both are deterministic and within about 2 ulp. |
| Only **4** TMU requests may be outstanding per TMU; the guide implies 8, and a 5th overwrites the oldest result | The software pipeliner keeps ≤ 4 per unit, 8 in total over TMU0 and TMU1. |
| The **VDW stride register is one register for all 12 QPUs** (probe `vdw_stride`: 592/1536 rows wrong) | Each DMA kick runs under the hardware mutex, released as soon as the DMA starts. |
| One VDW engine: about **0.18 µs per DMA + 0.85 ns per byte**. 256/512/1024-byte DMAs from 12 QPUs reach 643/804/917 MB/s. Only 64 VPM rows are addressable. | With 4 private rows per QPU, stores cap near 640 MB/s. This motivated store slots (A1). |
| The **instruction cache holds about 500 instructions**. Longer straight-line code misses once per 32-instruction line (3.06% of fetches), and the stalls are not counted as busy. | Large elementwise blocks are rerolled into loops (`piton/reroll.py`). |
| The uncached `0xC` bus alias is >100× slower, and `0x0` hung the TMU | Buffers use the mailbox's default `0x4` (L2-cached) alias. |
| The request queue's completed-program count is **8 bits** | The launcher reads it at least once per 15 issues, so it cannot wrap unseen. |

---

## 3. The compiler pipeline

*Python AST → Triton-style tile IR → optimised tile IR → machine IR on virtual registers → allocated →
scheduled 64-bit instructions → `.qasm` text, machine words and a C launcher. The package is about
15,000 lines of Python; the lowering alone is about 5,000.*

```
Python AST ──frontend.py──▶ tile IR ──passes.py / reroll.py──▶ optimised tile IR
                                                                    │ backend/lower.py
                                                                    ▼
QPU program ◀──sched.py── allocated MIR ◀──regalloc.py── MIR ◀──opt.py, presched.py
(isa.py words, asm.py .qasm, tools/compile.py C)
```

### 3.1 Frontend and tile IR

`piton/frontend.py` works like Triton's `code_generator.py`: it *executes* the kernel's Python AST
symbolically. `tl.constexpr` values are ordinary Python objects, so Python `if`/`for` on them is
resolved at compile time. Values that depend on runtime data are `tl.tensor`s wrapping IR values.
`for` and `if` on runtime values become `scf.for`/`scf.if` with loop-carried variables found from the
AST. `piton/ir.py` is modelled on Triton's TTIR, and `fn.dump()` prints the same textual form. The
vector-add kernel above becomes, in part:

```
%3 = tt.make_range  {start = 0, end = 64} : tensor<64xi32>
%5 = arith.addi %4, %3 : tensor<64xi32>
%7 = arith.cmpi %5, %6 {predicate = "slt"} : tensor<64xi1>
%10 = tt.load %9, %7 {has_mask = True, has_other = False} : tensor<64xf32>
```

`piton/passes.py` runs canonicalisation (folding, CSE, DCE), loop-invariant code motion, broadcast
sinking (index arithmetic is evaluated directly in the broadcast shape, which avoids cross-lane
shuffles), integer **range analysis**, and pruning of unused loop-carried values. Range analysis folds
masks whose outcome is known, such as a matmul's K-mask when K is a constexpr multiple of BLOCK_K. It
also tells the backend when 24-bit multiplies are exact. `kernel[grid](...)` compiles for its grid, so
`program_id` arithmetic is bounded too.

`piton/reroll.py` exists because of the instruction cache. A 1-D elementwise block whose unrolled
program would exceed 450 instructions becomes a loop over 64-element chunks. With D1 (section 6) this
also applies to row reductions such as softmax's max and sum passes.

**Supported Triton** (README, "Supported Triton"): `program_id`, `num_programs`, `arange`,
`zeros/full`, masked `load/store` with `other`, `dot`, `where`, `sum/max/min`,
`exp/exp2/log/log2/sqrt/rsqrt/abs/sigmoid`, `maximum/minimum`, `cdiv`, `static_range`, `.to()`,
`[:, None]` indexing, `reshape/trans`, runtime `for` loops with carried values, runtime `if` including
an early `return`, and inlined `@jit` helpers. Supported types are `float32`, `int32`, `uint32` and
`int1`; the QPU has no fp16 or fp64. Integer `//` and `%` follow C semantics and are exact for all
32-bit values. Block shapes must be powers of two. There is no `break`, and no register spilling.

### 3.2 Lowering to machine IR

`piton/backend/lower.py` turns tile ops into QPU operations on virtual registers. The decisions that
matter most:

* **Layouts.** Element *f* of a block is lane *f* mod 16 of register ⌊*f*/16⌋. Values keep the cheapest
  exact form: a scalar, a *splat*, an **affine** index (`base + Σ coeffᵢ·indexᵢ`, so a `[4, 64]`
  pointer tile costs four lane registers plus one scalar instead of sixteen registers), dense
  registers, one uniform per element, or a **lazy predicate** tree. A mask becomes condition flags only
  where a load, store or select needs it.
* **Loads** are TMU gathers. Masked-off lanes read a hidden zero page, so `other=0` is free. Loads in a
  loop whose addresses depend only on loop-carried pointers are **software pipelined**: each receive
  immediately refills the FIFO with the next request, including the next iteration's first ones.
* **Stores** go through VPM rows and the VDW DMA, as 16-word rows with a stride. 2-D masks become
  runtime row/column counts, and a versioned fast path skips them when the tile is fully in bounds.
* **`tl.dot`** is a register-blocked outer product. For each `k` and row `m`, a rotated
  `mov r5rep, ...` on the mul ALU broadcasts `A[m,k]` to every lane. One *paired* instruction per 16
  columns then multiplies the broadcast by a B row (mul ALU) and accumulates the previous product
  (add ALU). This is the hot loop of the tutorial matmul as compiled today:

  ```
  nop; mov r5rep, r0 >> 8                       # broadcast A[m, k] to all lanes
  fadd ra10, ra10, r1; fmul r0, r5, rb25        # acc += prev product; product = A * B[k, 0:16]
  fadd rb8,  rb8,  r0; fmul r0, r5, ra2         # ...           B[k, 16:32]
  fadd ra11, ra11, r0; fmul r0, r5, rb28        # ...           B[k, 32:48]
  fadd rb9,  rb9,  r0; fmul r0, r5, ra8         # ...           B[k, 48:64]
  add t1s, rb4, r2; mov r5rep, r1               # next TMU request + next broadcast
  ```

  With `BLOCK_N = 64` that is 4 vector multiply-accumulates per 5 instructions. The 32 accumulators
  alternate between files A and B so that every pair reads one register from each.
* **Reductions** across a register's 16 lanes are butterflies: rotations by 8, 4, 2 and 1 on the mul
  ALU, each followed by an add or max. Small tiles, transposes and broadcasts use rotations and masked
  moves.
* **Transcendentals run on the ALUs.** `exp` reduces `x = n·ln2 + r` and evaluates a degree-6
  polynomial, then builds `2ⁿ` in the exponent bits. Reciprocal and rsqrt start from an integer seed
  and take Newton steps. Accuracy is about 22-23 bits, within about 2 ulp, and every result is
  reproducible. Special values (NaN, ±inf, ±0, denormals) are handled explicitly.
* **Integer division** is exact for all 32-bit operands. Since C10 it is branch-free: a float-reciprocal
  estimate plus one correction, with `//` and `%` sharing the estimate and constant divisors taking
  shortcuts.

### 3.3 Register allocation

`piton/backend/regalloc.py` never spills. That is a deliberate limit: a tile that does not fit fails
with `RegAllocError`, and the compiler then retries with less aggressive options (no cached A
rotations, private store rows, the branching division). The allocator:

1. computes live segments and puts rotation sources and short-lived temporaries in r0-r3;
2. assigns the remaining values to file A or B as a **2-colouring of the "must differ" graph**, with
   one edge per pair of register operands that share an instruction, and orients each component to
   balance pressure;
3. assigns first-fit inside each file, re-checks that every instruction encodes, and inserts an
   accumulator copy where two operands still collide.

**Port-aware allocation** (C1, on by default) re-runs step 2 using the pairs the scheduler failed to
form because of read ports, weighted by how often they execute. It keeps the result only when it
schedules shorter.

### 3.4 Scheduling and encoding

`piton/backend/presched.py` reorders each block *before* allocation by critical-path height. Lowering
emits one 16-lane register at a time, so an exp over 256 elements arrives as 16 serial chains. Without
interleaving, allocation reuses the same accumulators for every chain and the false dependences stop
any overlap. The compiler builds both orders and keeps the shorter.

`piton/backend/sched.py` is a list scheduler over a dependence DAG whose edge latencies encode the
pipeline rules: 2 instructions for a register-file read-after-write, 1 for an accumulator, 2 before
rotating a freshly written accumulator, 3 from an SFU write to an r4 read, 3 from `unif_addr` to the
next uniform, and so on. Ready ops are packed into one add op, one mul op and one signal per
instruction. A candidate is accepted only if it actually encodes, which settles read ports, the
write-swap bit and the one-peripheral-access rule. The scheduler also fills the three branch-delay
slots and fixes hazards across block boundaries.

**Pairing** (C11) builds each block with several priority functions and keeps the shortest schedule.
It also lets a register move step over to the mul ALU when an add-only op needs the add ALU. A loop
that streams through the TMU keeps the schedule that waits least for its loads, not the shortest one.

`piton/qpu/isa.py` encodes and decodes 64-bit instructions straight from the Broadcom reference guide.
`piton/qpu/asm.py` prints them in `vc4asm` syntax, chosen so that `vc4asm` assembles the text back to
exactly the same bits.

### 3.5 The launcher and runtime

`kernel[grid](...)` (`piton/runtime/launcher.py`) copies numpy arrays in, builds one uniform list per
program, runs the emulator or the Pi, and copies results back. `python -m piton.tools.compile`
(`emit_c`) is the ahead-of-time path for firmware. It writes a `.qasm`, plus a `.c/.h` pair holding
the machine code and a typed launcher:

```c
matmul_args_t args = { .a_ptr = A.bus, ..., .M = 512, ... };
uint32_t usec = matmul_launch(&args, grid_x, 1, 1);
```

The generated launcher also chooses the coarsening factor per launch (E4, section 6) and sets up the
store-slot semaphores once per boot (A1).

---

## 4. Verification

*Every kernel is checked against numpy in a strict emulator that rejects undefined instruction
sequences. The encoder is cross-checked against an independent assembler, and every hardware test
gives exactly the emulator's bits.*

* **The strict emulator** (`piton/qpu/emulator.py`) executes real machine words for all 12 QPUs. It
  models the TMU FIFOs (4 deep, with the QPU2/QPU3 TMU swap and TMU_NOSWAP), the SFU, VPM, VDR/VDW
  DMA, semaphores, the mutex, the request queue and the float behaviour measured on the board. The
  QPUs run interleaved, so semaphores and the shared VPM see a real ordering of events. In strict mode
  it raises `HazardError` for whatever the reference guide calls undefined: reading a register-file
  location written by the previous instruction, touching r4 too early after an SFU write, overflowing
  a TMU FIFO, starting a DMA while one is still running, or a VPM write after a DMA start without
  `vw_wait`. It also catches two inter-QPU VPM races that shared store slots must rule out, a deadlock
  (every QPU blocked), and a thread end before the last store was waited for. It is functional and only
  approximately timed. Section 6 shows where that mattered.
* **Tests.** On the final branch, `pytest --collect-only tests perf` collects 1,433 tests in 32 test
  files plus the perf suite. The last recorded full runs are 1,276 passed and 33 skipped (perf-loop
  2557a06, 62 minutes) and 1,368 passed and 57 skipped with `PITON_PAIR=1` (4 h 55 min) (perf/NEXT.md).
  They cover tutorial kernels against numpy; small-tile reductions, transposes and dots
  (`test_tiles.py`); IEEE comparisons on NaN/inf/±0; exp/log/div/sqrt accuracy; exact integer
  division, including a numpy model of every division path and tests over every mantissa;
  trained-network inference with propagated float32 error bounds; benchmark-log parsing; and one test
  file per optimisation switch (`test_store_slots_protocol.py`, `test_pair_adversarial.py`,
  `test_tmu_noswap.py`, ...).
* **Independent encoder check.** `tests/test_vc4asm.py` assembles the emitted `.qasm` with
  [vc4asm](https://maazl.de/project/vc4asm/doc/index.html) and requires exactly the words PIton's
  encoder produced. Several switch tests also assemble the Pi builds with `vc4asm -V`.
* **Hardware tests.** `PITON_HW=1 pytest tests/test_hw.py` runs 12 tests on the Pi and compares
  complete outputs with the emulator, bit for bit. They cover vector add, softmax, icache-sized softmax
  and transpose builds, where/row-sum, ragged matmuls with up to 32 concurrent programs, exp/log,
  div/rsqrt/sqrt, float compares on special values, lane permutations, negative-stride stores, and
  integer division including edge pairs. The count grew from 10 to 12 during the loop, and every
  accepted change passed it with its switch on.
* **Review rounds and regressions.** Before the loop, five review rounds found about 30, 10, 17, 13 and
  10 confirmed bugs (STATUS.md). Each was fixed with a regression test (`tests/test_regressions.py`).
  During the loop every change got an adversarial review whose findings became tests (commits such as
  "Store slot review: all-paths protocol check and slots-off differential tests").
* **Application-level checks.** The digit network compares all 28,752 logits of every full run against
  an independent float64 reference, within a propagated bound for round-toward-zero arithmetic. The
  Mandelbrot firmware checks each kernel bit for bit against the emulator's escape counts on every boot.
  Benchmarks pre-fill outputs with NaN and check sampled or complete results.

---

## 5. Running on the Pi

*Programs run bare metal. Each run is a freshly generated C program, built against the course `libpi`,
sent over the UART by the course bootloader, timed on the board, and read back from the serial log. A
USB-controlled power switch recovers the board when a kernel hangs.*

* **Setup.** This is the CS140E/CS240LX bare-metal environment: a `my-install`/`pi-install` bootloader
  and `libpi` from the course repository. The SD card must run the full firmware (`start.elf`, not
  `start_cd.elf`) with `gpu_mem=64` or more. Otherwise the V3D block is powered off and `V3D_IDENT0`
  reads `0xdeadbeef`.
* **Runtime** (`pi/runtime/piton_rt.c`): GPU memory from the mailbox, a launcher for the V3D
  user-program queue, and helpers for clocks and the V3D performance counters. `piton/runtime/pi.py`
  generates `notmain()` for each run. It embeds the QPU code and inputs, launches, and prints the
  requested buffers over the UART. The UART moves about 11 KB/s, so benchmarks generate large inputs
  on the board from a shared LCG (`piton_rt.h`, mirrored in numpy).
* **Timing.** `piton_launch` clears the V3D L2 and slice caches, then reads the microsecond timer
  around issuing every program and waiting for the last completion. The suite reports the best of 5
  launches per boot, and the digit network the median of 5. `perf/profile_pi.py` reads the V3D counters
  for the best launch: *busy* (cycles executing / 12 QPUs × clock × time), *TMU stall*, *idle*, and
  instruction-cache misses. These say whether a kernel is limited by instructions, memory or launch.
* **Automated power switch.** A hung kernel wedges the board, and early on that meant unplugging it.
  The HUGE UART board from CS340LX lab 3 has a CP2102N bridge whose GPIO0 drives a MOSFET on the Pi's
  5 V rail. `tools/pi_power` (macOS, IOKit vendor request) and `tools/pi_power_linux` (libgpiod on
  the cp210x gpiochip) switch it. `piton/runtime/power.py` tells a ready Pi from a wedged one by
  listening for the bootloader's `GET_PROG_INFO` announcements, which repeat about every 0.3 s. It
  power-cycles when they are missing. A launch timeout in `piton_launch` reboots the board itself.
  Together these let risky items (store slots, coarsening, a new mutex-release encoding in C11) run
  unattended.
* **Linux.** The last day's measurements ran partly on a second host (WSL2) with a different UART head.
  That needed the Linux power tool, `PITON_TTY` pointing at a `/dev/serial/by-id/` link, and the
  cs240lx `libpi` built with gcc 13.2 (`-mfloat-abi=softfp`). Under softfp the ARM VFP baseline
  assembly needed a fix (`dd18d2b`). The same compiler timed on both setups agreed within 0.2%
  (geomean; cases 0.97-1.035×). The headline suite numbers above were re-measured on the original board.

---

## 6. The optimisation loop (2026-11-03 to 2026-11-05)

*Each candidate optimisation sat behind a switch and was measured on the Pi against a fixed 23-kernel
suite under one acceptance rule. Eleven changes shipped; the rest were rejected with recorded reasons.
The suite went from 1.000× to 1.424× in three days.*

### 6.1 Method

* **Everything behind a switch.** Each change has a `PITON_*` environment variable (`PITON_STORE_SLOTS`,
  `PITON_PORT_ALLOC`, `PITON_ISSUE`, ...). With the switch off, the compiler produces **bit-identical
  code** to the base (checked over the suite's builds, e.g. "switch off bit-identical to d164de2 (51
  builds)"). Host tests ran with the switch both off and on.
* **A fixed suite.** `perf/suite.py` holds 23 cases: every primitive, the three Triton tutorials,
  the digits dense layer, two Mandelbrot tilings and the compute-peak recurrence. Inputs come from the
  board's LCG. Tile sizes stayed fixed (for example `dense_relu` keeps BK=8 even though the digits app
  moved to BK=4), so the comparison measures compiler changes only.
* **The gate** (`perf/accept.py`): host tests both ways, then `test_hw.py` with the switch on, then
  `perf/ab.py`. `ab.py` compiles every case both ways and skips cases whose code is identical. It boots
  the rest alternately off, on, on, off, takes the best of 5 launches per boot, and checks sampled
  outputs. **Accept** when every result is correct, some target case is **more than 2% faster**, and
  **no case is more than 2% slower**.
* **Noise handling.** Boot-to-boot noise in the baseline was mostly below 0.5% but up to 5% (sqrt), so
  `ab.py` gives any result outside ±2% more off/on pairs. Memory-bound cases needed up to 6 boots each
  way, compared by the median of per-boot bests.
* **Commit only when accepted**, on branch `perf-loop` (never `main`), one item per commit with its
  `perf/results/ab-*.json`, so any change can be reverted on its own. Rules, results and rejections are
  logged in `perf/BACKLOG.md`.

**How the work was organised.** The changes were implemented and adversarially reviewed by AI coding
agents. Each worked in its own git worktree and branch (`opt/<item>` for the change, `int/<item>` or
`try/<item>` to combine it with the current loop head) and tested only against the strict emulator and
the host tests. One coordinator ran every hardware test and Pi A/B, applied the acceptance rule, and
merged accepted items. The project's author set the goals and made the calls the rule did not settle,
such as whether C11's compile-time cost was acceptable.

### 6.2 Every item and its outcome

Speedups are Pi times off → on, best of 5 (`perf/BACKLOG.md`, log rows; `perf/results/ab-*.json`).
Items are in the order they were measured.

| item | change | outcome | Pi effect |
|---|---|---|---|
| infra | power switch + auto-recovery, `ab.py`, `accept.py`, `compare.py` | built | — |
| A4, G1, D3 | BLOCK / UNROLL / reroll-chunk sweeps (`perf/tune.py`) | suite configs kept fixed; digits app BK 8 → 4 | 2-input kernels 6-9% faster at BLOCK ≥ 1024; digits full batch 1.099×; 64-element chunks stay best |
| C1 (+C2) | port-aware register allocation | **accepted** | mandel 1.216× / 1.265×, transpose 1.093×, softmax 1.041×, nothing below 0.993× |
| A1 | shared 16-row VPM store slots under semaphores (1 KiB DMAs) | **accepted** | copy 1.478×, scaled_relu 1.482×, sqrt 1.404×, recip 1.393×, exp 1.243×, add 1.129×; softmax 0.981× |
| A1 rule | slots only where stores ≥ 0.1 B per executed instruction | **accepted** | matmul 1024³ 7.89-8.02 vs 7.64-7.68 GFLOPS; suite code unchanged |
| C10 | branch-free exact integer division | **accepted** | idiv 2.775×, dense_relu 1.146×, matmul 1.03-1.05× |
| E1 v1 | coarsening: 4 program ids per hardware program for any short id | rejected | add 1.214×, fma 1.292×, but idiv 0.940×, rsqrt 0.966× (uneven last round) |
| E1 | coarsen only kernels moving ≥ 4 B per executed instruction | **accepted** | add 1.206×, mul 1.236×, fma 1.306×, dense_relu 1.103× |
| H2 + H4 | fast Mandelbrot: cardioid/bulb lanes, lane-flag exit checks, early exit | **accepted** | mandel 1.239× / 1.162×, escape counts identical |
| C3 + C5 | fill loop delay slots; unroll rerolled loops 2× | rejected | mandel 1.013× / 1.014×, rest ±1% |
| D1 + D2 | reroll reductions + reroll 2-D tiles as transposed loads | D2 rejected | softmax 1.713× but transpose 0.712× |
| D1 | reroll row reductions only | **accepted** | softmax 1.730×, everything else same code |
| A2, A3, D4, E2, E3 | double-buffered slots; uncached store alias; loop alignment; skipping the cache flush; prebuilt uniforms | not built / rejected on analysis | VDW engine already saturated; ARM/L2 coherence; hot loops already fit; outside the timed region |
| G2 | matmul tile × group sweep after the changes (48 runs) | defaults unchanged | 8×64×8 still best; group 2's edge within boot noise |
| A1 runtime | slot semaphores set up once per boot | **accepted** | digits batch 1 1.078×, batch 16 1.063× |
| C7 + C8 | issue efficiency: MIR CSE, cheaper special-value checks, either-ALU ops on the mul ALU, flag groups in pre-scheduling | **accepted** | log 1.261×, log2 1.237×, softmax 1.207×, exp 1.203×, idiv 1.116× |
| E4 rule 1 | coarsening factor chosen per launch, E1's thresholds | replaced by rule 3 | digits batch 1 1.19×, 16 1.34×, but batch 256 0.981× |
| E4 sweep | one build launched with every factor (`perf/coarsen_sweep.py`) | measurement | gains track *bytes stored* per instruction (6.3) |
| E4 rule 2 | 64 ids / 32 programs for every kernel | rejected | row_sum 0.934×, row_max 0.946×, matmul 0.941-0.945× |
| E4 rule 3 | per-kernel thresholds by stored bytes | **accepted** | digits batch 1 1.19×, 16 1.31×, 64 1.08×; dense_relu 0.989× over 6 boots |
| E5 | coarsen only where ids store ≥ 0.4 B per instruction | **accepted** | digits batch 64 1.024× (median of 4 pairs); suite code unchanged |
| B1 + B2 | both TMUs alternating, deeper prefetch, hoisting | dropped | 0.986-1.008× |
| B5 v1 | TMU_NOSWAP in every balanced, uncoarsened program | rejected | matmul 1.09×, matmul_pi 1.13-1.16×, but row_max 0.926×, digits 0.935× |
| B5 v2 | TMU_NOSWAP only for dots | rejected | digits still 0.94-0.95× |
| B5 v3 | TMU_NOSWAP as a per-kernel request (the tutorial matmul) | **accepted** | suite matmul 1.098× / 1.099×; matmul_pi 512³ 8.74-8.94 → 10.17-10.21 GFLOPS |
| H6 | matmul A in column blocks, loads received in use order | rejected | matmul 0.716× / 0.803× |
| C11 | scheduler pairing | **accepted**; on in the default configuration | mandel 1.074× / 1.057×, row_sum 1.044×, softmax 1.026×, suite 1.010×; compiles 3.4-6.4× slower |

Suite geomean over the 11-03 baseline as items landed: 1.165× (C1 + A1), 1.288× (+ C10, E1, H2),
1.328× (+ D1 and the slot rules, 11-04), 1.409× (+ C7, E4, E5, B5, on the second host), 1.406× (the
same code on the original board), and 1.424× with C11.

### 6.3 Five stories

**Store slots and the sparse-store rule (A1).** The probes showed one DMA engine for all 12 QPUs at
about 0.18 µs per DMA. PIton's 4 private VPM rows per QPU limited each DMA to 256 bytes, which held
every elementwise kernel near 640 MB/s. Larger DMAs need more rows than a QPU can own, because only 64
rows are addressable. A1 splits the VPM into four 16-row slots. QPU *q* uses slot *q* & 3, guarded by
one of four hardware semaphores, fills up to 16 rows and issues one 1 KiB DMA. Rerolled loops carry
their stored values in registers so that the whole block leaves in one DMA. Deadlock freedom is argued
in the `lower.py` docstring and checked by the emulator's race detection. Copy went 1.478× faster and
reached about 940 MB/s. A later sweep found the cost: a slot store must **wait for its DMA** before
releasing the slot, while a private-row store lets the program run on. At 1024³, where matmul stores
rarely (0.04 bytes per executed instruction), slots made it slower (7.55-7.66 vs 8.01-8.05 GFLOPS).
The fix is a rule: use slots only where a kernel stores ≥ 0.1 bytes per executed instruction. The
digit network then exposed a third cost: setting up and tearing down the slot semaphores took two
extra one-program launches, about 20 µs per call. Doing that once per boot made batch 1 1.078× faster.

**Coarsening by stored bytes (E1, E4, E5).** Coarsening runs *k* consecutive program ids in one
hardware program, which saves per-program launch cost and lets one id's store DMAs drain while the
next id computes. The first rule (k = 4 for any short id) sped up memory-bound kernels by 21-29% but
slowed compute-bound ones by up to 6%. With 256 programs, the last round leaves some of the 12 QPUs
idle. The accepted rule coarsened only kernels moving ≥ 4 bytes per executed instruction (memory-bound
kernels move 7.7-8.9, compute-bound ones 1.9-2.7). Then the digit network got *slower* at small
batches (batch 1: 0.194 → 0.235 ms). Its layers are compiled once for the largest batch, so a small
launch was coarsened as if it were full and ran on half as many hardware programs. E4 made the factor
a uniform chosen per launch. To set the thresholds, a sweep launched one build of each kernel with
every factor on every grid. It found that what decides is **bytes stored** per instruction, not bytes
moved: `add` (2.8 B stored) and the first digits layer (0.48) gain, while the second (0.26) and third
(0.32) layers and matmul (0.14) lose even on full grids. One threshold for every kernel (rule 2) was
rejected because row_sum and matmul lost 5-7%. The accepted design uses per-kernel thresholds and
coarsens only where ids store ≥ 0.4 B per instruction (E5). Digits batch 16 ended 1.31× faster than
before E4.

**TMU_NOSWAP: a property the compiler cannot see (B5).** Each slice of four QPUs has two TMUs, and by
default QPUs 2 and 3 have TMU0 and TMU1 swapped. TMU_NOSWAP turns the swap off, so the same request
stream from all four QPUs goes to the same physical unit and its cache. Version 1 set it in every
balanced, uncoarsened program: matmul gained 9% and the matmul benchmark 13-16%, but row_max lost 7%
and the digit network 6.5%. Version 2 set it only for dot products: the reductions were fixed, but
digits still lost 5-6%. Timing each digits layer on its own, noswap off vs on, explained it. The
tutorial matmul gained 1.111× and the first layer (two column tiles) 1.041×, but layer 2 (one column
tile) lost (0.867×) and so did layer 3 (0.942×). Noswap pays only when programs running together on a
slice load the same A rows. That depends on the grid order and grouping, not on anything visible in
one program. Version 3 made it a **per-kernel request** (`compile(..., tmu_noswap=True)`), used by the
tutorial matmul, with every other loading program writing 0 explicitly. Matmul at 512³ went from
8.7-8.9 to 10.2 GFLOPS and at 1024³ from 7.9-8.1 to 9.35, with nothing else changed.

**H6: fewer instructions, slower code.** H6 restructured matmul's k loop to keep A in column blocks
and receive each load in use order. In the emulator it executed 3-10% fewer instructions, and programs
were 2.4-4.0% shorter. On the Pi, matmul ran at **0.716×** and **0.803×**. The counters showed why:
the TMU-stall share of QPU time went from 0.22 to 0.47, and busy fell from 0.63 to 0.42. Receiving
each load just before its use leaves no load latency hidden. The emulator does not model that latency
faithfully, so it could not see the problem. The same lesson shaped C11: for a loop streaming through
the TMU, its scheduler keeps the schedule that waits least for its loads, not the shortest one.

**Measurement noise.** The same build's best-of-5 moved by about ±3% between boots on memory-bound
cases. `add` measured 0.977× on one A/B and 1.024× on the recheck. A row_sum "regression" of 0.951×
was 0.993× over 6 boots each way. The baseline's row_sum/row_max (138/139 µs) was a fast boot: every
profile since 11-02 put them at 141-146 µs. That is why the final profile shows them at 0.965×,
although no accepted change was more than 2% slower on them. A matmul 1024³ figure of 8.02 GFLOPS from
one morning was never reproduced, even with every switch off. Re-measuring anything outside ±2%, and
recording every repeat, kept such boots from deciding.

---

## 7. Results

*Memory-bound kernels improved by 1.3-1.5×, the math kernels by about 1.5×, division by 3×, softmax by
2.3× and Mandelbrot by 1.5-1.6×. Matmul gained 1.13-1.19× in the suite and 16-19% in the
large-matrix benchmark.*

### 7.1 The suite, before and after (same board)

Best-of-5 µs; *busy* and *TMU stall* as fractions of QPU time (V3D counters). The last two columns are
the default configuration with scheduler pairing: the C11 A/B's on-times for the 22 cases whose code
pairing changes, and the final profile for `row_max`. They reproduce the 1.424× geomean. Because they
come from other boots, some cases read slightly below the column without pairing.

| case | 11-03 µs | final µs | speedup | busy | TMU stall | with pairing µs | speedup |
|---|---|---|---|---|---|---|---|
| `copy` | 1,651 | 1,108 | 1.490× | 0.16 → 0.19 | 0.02 → 0.04 | 1,117 | 1.478× |
| `add` (tutorial 01) | 1,874 | 1,393 | 1.345× | 0.28 → 0.23 | 0.16 → 0.11 | 1,376 | 1.362× |
| `mul` | 1,868 | 1,392 | 1.342× | 0.28 → 0.23 | 0.16 → 0.10 | 1,380 | 1.354× |
| `fma` | 1,915 | 1,394 | 1.374× | 0.28 → 0.21 | 0.12 → 0.13 | 1,406 | 1.362× |
| `scaled_relu` (`tl.where`) | 1,658 | 1,150 | 1.442× | 0.36 → 0.36 | 0.12 → 0.03 | 1,133 | 1.463× |
| `exp` | 1,725 | 1,119 | 1.542× | 0.78 → 0.80 | 0.04 → 0.00 | 1,129 | 1.528× |
| `exp2` | 1,734 | 1,139 | 1.522× | 0.75 → 0.70 | 0.04 → 0.00 | 1,141 | 1.520× |
| `log` | 1,729 | 1,144 | 1.511× | 0.75 → 0.73 | 0.04 → 0.00 | 1,138 | 1.519× |
| `log2` | 1,705 | 1,144 | 1.490× | 0.75 → 0.72 | 0.04 → 0.00 | 1,136 | 1.501× |
| `recip` | 1,668 | 1,130 | 1.476× | 0.66 → 0.38 | 0.04 → 0.01 | 1,133 | 1.472× |
| `rsqrt` | 1,677 | 1,144 | 1.466× | 0.64 → 0.61 | 0.04 → 0.00 | 1,153 | 1.454× |
| `sqrt` | 1,638 | 1,150 | 1.424× | 0.62 → 0.56 | 0.05 → 0.00 | 1,140 | 1.437× |
| `idiv` (exact i32 `//`, `%`) | 7,827 | 2,540 | 3.081× | 0.72 → 0.61 | 0.14 → 0.01 | 2,503 | 3.127× |
| `row_sum` | 138 | 143 | 0.965× | 0.31 → 0.27 | 0.43 → 0.50 | 136 | 1.015× |
| `row_max` | 139 | 144 | 0.965× | 0.31 → 0.27 | 0.44 → 0.51 | 144 | 0.965× |
| `transpose` | 889 | 761 | 1.168× | 0.41 → 0.41 | 0.00 → 0.00 | 733 | 1.213× |
| `softmax` (tutorial 02) | 903 | 397 | 2.275× | 0.42 → 0.85 | 0.00 → 0.02 | 386 | 2.339× |
| `matmul_8x64x4` (tutorial 03) | 4,376 | 3,846 | 1.138× | 0.52 → 0.57 | 0.35 → 0.31 | 3,858 | 1.134× |
| `matmul_8x64x8` (tutorial 03) | 4,163 | 3,492 | 1.192× | 0.56 → 0.64 | 0.26 → 0.21 | 3,530 | 1.179× |
| `dense_relu` (digits layer 1) | 5,036 | 3,991 | 1.262× | 0.52 → 0.58 | 0.25 → 0.34 | 3,896 | 1.293× |
| `mandel_32x16` | 42,398 | 27,665 | 1.533× | 0.97 → 0.98 | 0.00 → 0.00 | 26,179 | 1.620× |
| `mandel_16x16x2` | 41,139 | 27,990 | 1.470× | 0.98 → 0.98 | 0.00 → 0.00 | 26,052 | 1.579× |
| `compute_peak` | 1,132 | 1,133 | 0.999× | 0.99 → 0.98 | 0.00 → 0.00 | 1,133 | 0.999× |
| **geomean** | | | **1.406×** | | | | **1.424×** |

The 11-03 baseline already included the pre-loop changes of 11-02/03 (rerolling, pre-scheduling, the
launcher fix). Against the first profile of 2026-11-02 (`perf/results/pi-2026-11-02-before.json`) the
final suite is 1.69× faster on geomean, with exp going from 3,041 to 1,119 µs and idiv from 13,065 to
2,540 µs. That comparison is computed here from the two files and was not part of the loop's own
accounting.

### 7.2 Matrix multiplication

`benchmarks/matmul_pi.py --bk 8`: the Triton tutorial kernel with runtime dimensions and strides, inputs
generated on the board, sampled outputs checked against float64.

| N | 2026-10-30 | 2026-11-05, final configuration | change |
|---|---|---|---|
| 128 | — | 0.447 ms, **9.38 GFLOPS** | |
| 512 | 30.569 ms, 8.781 GFLOPS | 26.454 ms, **10.15 GFLOPS** | 1.156× |
| 1024 | 273.409 ms, 7.854 GFLOPS | 229.831 ms, **9.34 GFLOPS** | 1.190× |

(`benchmarks/results/matmul-pi-2026-11-05-final.log`, the original board with every accepted change;
the second board before pairing was on gave 8.21 / 10.15 / 9.30, `matmul-pi-2026-11-05.log`.)

These figures set the scale. The peak is 24 GFLOPS, and a register-resident recurrence reaches 23.794.
At the 10-30 checkpoint the BK=8 loop executed 256 vector multiply-adds in 386 instructions, an
instruction-issue ceiling of 15.9 GFLOPS. In the final suite profile matmul is 64% busy with 21% of QPU
time stalled on the TMU. The remaining gap to peak is the A broadcasts (one mul-ALU slot per 4 MACs),
address arithmetic, and the B stream through the TMU and the 128 KB L2. On 2026-10-30 a 512³ GPU call
was 11.1× faster than blocked VFP assembly on the ARM (`benchmarks/results/cpu-comparison-2026-10-30.log`).

### 7.3 The digit network

A NumPy-trained MLP, 64 → 128 ReLU → 64 ReLU → 10, on the UCI handwritten-digits test split (1,797
images from writers not seen in training). The layers use 4×64×4, 4×64×4 and 8×16×4 tiles with bias
and ReLU fused. Accuracy is **96.995%** (1,743/1,797). Every QPU prediction matches the host reference,
and the maximum logit error is 0.000102. Times are medians of five, including all launch, uniform and
cache-clear overhead; final configuration on the original board
(`benchmarks/results/digits-pi-2026-11-05-final-sameboard.json`):

| batch | QPU | ARM VFP (tuned) | speedup | QPU on 2026-10-30 |
|---|---|---|---|---|
| 1 | 169 µs | 719 µs | 4.25× | 199 µs |
| 16 | 230 µs | 1,452 µs | 6.31× | |
| 64 | 489 µs | 4,524 µs | 9.25× | |
| 256 | 1,478 µs | 18,556 µs | 12.55× | |
| 1,797 | **9.320 ms** | 133.675 ms | **14.34×** | 11.161 ms |

The full batch classifies about 192,811 images/s. The second board, before pairing was on, measured
9.380 ms (14.22×, `digits-pi-2026-11-05-final.json`); pairing (C11) alone measured 1.04× at batch 1 and
about 1.01× at the full batch (`benchmarks/results/digits-pi-2026-11-05-pair-*.log`).

### 7.4 Primitives against the CPU

`pi/examples/primitives` times each Triton primitive alone in one boot against two or three ARM
versions: `arm-naive`, `arm-unroll`, `arm-vfp` (VFP11 vector assembly or hardware `vsqrt`/`vdiv`),
`arm-blocked`, and `arm-fast` (deliberately lower precision, with its error reported). Selected rows,
2026-11-05:

| primitive | QPU | before the loop (11-03) | vs best CPU | best CPU baseline |
|---|---|---|---|---|
| `tl.exp` | 1.61 ms | 2.21 ms | 35.6× | `arm-fast` 57.3 ms |
| softmax | 0.56 ms | 1.10 ms | 32.5× | `arm-fast` 18.2 ms |
| `//` and `%` (exact i32) | 3.17 ms | 8.54 ms | 17.0× | `arm-fast` 53.9 ms |
| `tl.dot` (matmul 256³) | 3.99 ms | 4.57 ms | 12.1× (188× naive) | `arm-vfp` 48.2 ms |
| `x*a+y` | 1.66 ms | 2.36 ms | 8.5× | `arm-vfp` 14.2 ms |
| copy | 1.23 ms | 2.17 ms | 6.0× | `arm-vfp` 7.40 ms |
| `tl.trans` | 0.93 ms | 1.10 ms | 2.5× | `arm-naive` 2.36 ms |

The ARM baselines were rebuilt on the second host between the two dates, and their times moved by
0.93-1.16×. The "vs" columns therefore mix QPU gains with ARM-side changes (README).

### 7.5 Where the time goes now

* **Compute-bound** (busy 0.85-0.98): Mandelbrot, compute_peak and softmax. Only fewer instructions
  help here, which is exactly what pairing's 1.06-1.07× on Mandelbrot came from.
* **Math kernels** (exp, log, sqrt, rsqrt, recip): 1.12-1.15 ms each, within 4% of `copy`'s 1.11 ms,
  with busy 0.38-0.80. They sit at the store-engine floor that `copy` sets, now with 1 KiB DMAs
  instead of 256-byte ones.
* **Store-bound**: `copy` writes at about 940 MB/s, the 1 KiB-DMA rate the probe measured, with the
  QPUs 19% busy. add/mul/fma (21-23% busy) are memory-bound, with two input streams and one output.
* **Load-bound**: row_sum/row_max (TMU stall 0.50) and the dense layer (0.34). Matmul is partly
  load-bound (0.21).

---

## 8. Showcase examples

*New demo kernels written against the final compiler, each checked on the Pi against numpy and an ARM
implementation of the same algorithm.*

**Image filters.** 2-D stencil kernels (for example blur and edge detection) over a full-size image,
each output tile computed from a neighbourhood of masked loads, checked against numpy and timed against
the ARM. Results: TBD.

**Game of Life** (`examples/life.py`, `docs/examples/life.md`). Conway's Life on a torus, one launch
per generation, in three Triton versions: the textbook 9-load stencil, a sliding window that carries
row sums through a runtime loop, and a bit-sliced kernel that packs 32 cells per uint32 and counts
neighbours with and/or/xor and shifts. On the Pi each runs 256 generations of a 512×512 soup, with
every generation of every implementation checked against numpy and against C versions of the same
algorithms on the ARM. The bit-sliced kernel updates **1.63 G cells/s** at 512² (161 µs per
generation, **4.81×** the ARM's bit-sliced C) and **1.84 G cells/s** at 2048² (**5.77×**); the plain
stencils are 4.2× and 8.4× their ARM twins but slower than the ARM's bit-sliced code, which shows how
much the algorithm matters next to the hardware (`benchmarks/results/life-pi-2026-11-05.{json,log}`).

**Attention** (`examples/attention.py`, `docs/examples/attention.md`). One scaled dot-product attention
head, softmax(QKᵀ/√d)·V, written like the forward pass of Triton's fused-attention tutorial: a program
per block of query rows, a loop over key/value blocks with two `tl.dot`s, a transposed load of K and an
online (running max and sum) softmax with `exp2`, optionally causal. On the Pi, for N = 128-512 tokens
and d = 32 or 64, every output matches the ARM reference and float64 numpy (max error 4·10⁻⁶); the best
tile reaches **3.3-3.7 GFLOPS**, **40-58×** a straightforward C implementation on the ARM (d = 32,
N = 512: 10.1 ms vs 526 ms; with K stored pre-transposed 9.1 ms, 57.7×). Register pressure (no
spilling) limits the tiles to 2-4 query rows (`benchmarks/results/attention-pi-2026-11-05.{json,log}`).

---

## 9. Limitations, future work and lessons

*PIton covers enough of Triton for the tutorials and a real network, but it has no spilling and no
fp16, and matmul is still well short of peak. The main lesson is that only the board decides what is
fast.*

### 9.1 Known limitations (STATUS.md, README)

* **No register spilling.** Very large tiles fail with `RegAllocError`. The compiler falls back to
  smaller variants where it has them.
* **Subset limits:** no `break`, block shapes must be powers of two, and there are no fp16/fp64 types.
  A carried Python number that becomes an int tensor in a nested loop and later a float is rejected.
  Masked tile stores assume the index does not wrap past INT_MAX within one tile.
* **Float behaviour:** everything rounds toward zero, and plain `+` and `*` keep the hardware's
  NaN-as-inf behaviour. Division, comparisons and exp/log/sqrt follow IEEE for special values. `ftoi`
  of some huge values is not modelled.
* **Compile time:** with scheduler pairing, compiling takes 3.4-6.4× the CPU time, and the full host
  suite about 5× longer.
* **Transpose** still compiles to a long straight-line program (2-D rerolling, D2, lost on the Pi).
  Its compiled output also still depends on object addresses (perf/NEXT.md).
* **The emulator's timing is approximate.** It does not model TMU latency well enough to rank
  load-scheduling choices (H6).

### 9.2 Future work

From `perf/BACKLOG.md` (items still `todo`) and `perf/NEXT.md`:

* **Matmul:** bulk loads of B through the VPM read DMA (B3), fewer uniforms per program (B4),
  software pipelining of loop bodies (C4), and an autotuner for GROUP_SIZE_M, whose best value depends
  on size (G2).
* **Elementwise:** unmasked full-block loads with one induction variable (C6). About half of the ALU
  ops in these kernels are moves, though the memory bound limits the gain.
* **Opt-in fast math** with fewer Newton steps or shorter polynomials (C9), kernel fusion (H1), and
  8/16-bit outputs (H5).
* **Follow-ups:** a constant-k program for full-grid launches would recover dense_relu's ~1% under E4.
  It is not known whether TMU_NOSWAP outlasts a program (PIton writes it in every loading program, so
  its own code is unaffected). `isa.format_instr` prints condition 7 in a form vc4asm reads as
  condition 5 (no build emits it today).
* **Clocks** (F1/F2: GPU above 250 MHz, SDRAM) were kept out of the loop on purpose, because they need a
  reboot and a stress test.

### 9.3 Lessons learned

1. **Measure the hardware before trusting the manual.** The TMU FIFO is 4 deep, not 8. The VDW stride
   register is shared by all QPUs. The instruction cache holds about 500 instructions. One DMA engine
   serves every core. Each finding came from a probe and changed the compiler.
2. **Make the emulator strict and bit-exact, then make the hardware agree with it.** Running
   transcendentals on the ALUs instead of the SFU cost some speed but bought reproducibility. With it,
   "bit-identical to the emulator" became a usable hardware test, and the AI agents could work against
   the emulator alone.
3. **Fewer instructions is not the same as faster.** H6 and the TMU-aware loop choice in C11 both
   showed that load latency, which the emulator does not time, can outweigh instruction count.
4. **Some performance properties live outside the kernel.** Whether TMU_NOSWAP helps depends on which
   programs run together. Whether coarsening helps depends on the launch size and on how much the ids
   store. Both needed a per-kernel request or a per-launch decision instead of one global rule.
5. **Switches that are bit-identical when off make a fast loop safe.** Each change could be A/B'd in
   one build, reverted as one commit, and combined with the others without re-validating the rest.
6. **Plan for noise and hangs.** A ±3% boot-to-boot spread is larger than many real gains, so the 2%
   rule needed re-measurement and medians over boots. An automated power switch turned a hung kernel
   from an end-of-session event into a retry.
