# PIton presentation guide 

A study sheet for presenting PIton: what it is, how the code is laid out, the IR levels (the "MLIR
hierarchy"), every compiler optimisation, the results, the examples, and the exact commands to
run each one yourself. Numbers come from `README.md`, `docs/REPORT.md`, `STATUS.md` and
`perf/BACKLOG.md`. The IR listings in section 3 were produced by running the compiler on this
branch (`perf-loop`).

---

## 0. The 30-second pitch

PIton is a **Triton compiler for the Raspberry Pi Zero's GPU** (VideoCore IV: 12 QPUs, each a
16-lane fp32 SIMD core at 250 MHz, 24 GFLOPS peak). You write the same `@triton.jit` Python
kernel you would write for an NVIDIA GPU. PIton lowers it through a Triton-style tile IR, then a
machine IR, does register allocation and instruction scheduling, and emits real QPU machine
code. That code runs **bit-identical to our own strict emulator** on the board.

Headline numbers (Pi Zero W, ARM 700 MHz, V3D 250 MHz):

| what | result |
|---|---|
| Triton tutorial matmul | **10.15 GFLOPS** at 512³ (9.38 at 128³, 9.34 at 1024³) |
| peak arithmetic | **23.794 GFLOPS = 99.14%** of the 24 GFLOPS peak |
| handwritten-digit MLP (64→128→64→10) | **96.995%** accuracy, full test set in **9.32 ms**, **14.34×** tuned ARM VFP |
| 23-kernel perf suite after the optimisation loop | **1.424×** geomean faster than the 11-03 baseline |
| primitives vs best ARM code | exp 35.6×, softmax 32.5×, matmul 12.1×, copy 6.0× |
| hardware tests | 12/12, outputs bit-identical to the emulator |

---

## 1. Hardware background (what you're compiling for)

You need this to explain *why* each optimisation exists.

| QPU feature | why the compiler cares |
|---|---|
| 16-lane SIMD, 1 instruction / 4 clocks (62.5 M instr/s per QPU) | A tile of N elements = ⌈N/16⌉ registers. Peak = 12 × 16 × 2 × 62.5 M = 24 GFLOPS. |
| **2 ALUs per 64-bit instruction**: add ALU + mul ALU | Peak needs one add op and one mul op paired in every instruction. Pairing is the scheduler's main job. |
| Registers: accumulators r0-r3, r4 (load/SFU result), r5 (broadcast), regfiles **A** and **B** (32 each) | **One read port per file**: two operands of one instruction can't both come from file A (or both from B). The allocator must respect this. |
| Regfile values readable only 2 instructions after write | Scheduler latency rule. |
| **TMU** loads: 16-address gather, async, ~12 cycles; only **4 outstanding per TMU** (manual says 8) | Software-pipeline loads; never more than 4 in flight per unit. |
| **VPM** scratchpad + **VDW** DMA for stores; **one DMA engine for all 12 QPUs** | Stores are the bottleneck for elementwise kernels. |
| SFU (exp2, log2, recip, rsqrt) | **Not used**: inaccurate (~13 bits) and not reproducible bit-for-bit. Math is done on the ALUs instead. |
| Branches: uniform across lanes, **3 delay slots** | Scheduler fills delay slots; per-lane decisions use condition codes. |
| Instruction cache ≈ **500 instructions** | Long straight-line kernels stall on fetch → "rerolling" pass. |
| Floats **round toward zero**, denormals flush to zero | Emulator reproduces this exactly. |

Several of these were wrong or missing in the Broadcom manual and were measured with probes
(`tools/probes.py`, `tools/float_specials_probe.py`, `perf/vdw_probe.py`,
`perf/icache_experiment.py`). Full table: README "Things we learned from the hardware".

---

## 2. Repository structure

```
piton/                      the compiler (Python, ~15k lines)
  __init__.py, jit.py         @triton.jit decorator, kernel[grid](...) entry
  language/core.py            tl.* builtins and the tensor type (Triton's broadcasting/promotion rules)
  frontend.py                 Python AST → tile IR (symbolic execution, like Triton's code_generator.py)
  ir.py                       tile IR data structures; fn.dump() prints TTIR-style text
  passes.py                   tile-IR optimisations (canonicalize, CSE, DCE, LICM, range analysis, ...)
  reroll.py                   instruction-cache pass: big unrolled blocks → loops over 64-element chunks
  compiler.py                 the driver: runs every stage, tries variants, keeps the best
  backend/
    lower.py                  tile IR → machine IR (layouts, loads, stores, tl.dot, reductions, math) ~5k lines
    mir.py                    machine IR data structures
    opt.py                    MIR cleanups: DCE, copy coalescing, LICM, CSE
    presched.py               reorders independent chains before register allocation
    regalloc.py               register allocation (accumulators + regfiles A/B, read-port aware)
    sched.py                  list scheduler / bundler (add+mul pairing, hazards, delay slots)
  qpu/
    isa.py                    64-bit instruction encode/decode (from the Broadcom guide)
    asm.py                    prints vc4asm-compatible .qasm text
    emulator.py               strict emulator of all 12 QPUs + TMU, VPM, VDW, SFU, mutex, semaphores
  runtime/
    launcher.py               kernel[grid](numpy arrays...) on the emulator or the Pi
    pi.py                     generates a bare-metal C program, boots it over UART, parses results
    power.py                  USB power switch: power-cycles a hung Pi
  tools/compile.py            ahead-of-time compiler → .qasm + .c/.h with a typed *_launch()

pi/runtime/piton_rt.c        bare-metal runtime: mailbox GPU memory, V3D queue launcher, perf counters
pi/examples/                 bare-metal programs (Makefiles): matmul, digits, primitives, mandelbrot,
                             image_filters, life, attention — each with ARM baselines in C/asm
examples/                    Triton kernels + host scripts (*_pi.py builds, boots and checks on the Pi)
tests/                       pytest: emulator vs numpy, vc4asm bit-exactness, hardware (PITON_HW=1)
perf/                        23-case suite, instruction-mix analyser, Pi profiler, A/B + acceptance gate
benchmarks/                  matmul sweeps + every saved hardware result (benchmarks/results/)
tools/                       hardware probes, pi_power (macOS) / pi_power_linux
docs/REPORT.md               the full project report (read this before presenting)
docs/examples/*.md           write-ups for image filters, Game of Life, attention
```

---

## 3. The IR hierarchy ("MLIR hierarchy")

**Important framing for questions:** PIton does **not** link against MLIR or LLVM. It is pure
Python. Its tile IR is *modelled on* Triton's TTIR, which is an MLIR dialect, and it prints in the
same MLIR textual syntax with the same dialect names (`tt.*`, `arith.*`, `scf.*`). Below that,
PIton uses its own QPU-specific machine IR instead of Triton's TTGIR → LLVM IR → PTX path.

### 3.1 Real Triton vs PIton, side by side

```
            Real Triton (NVIDIA)                      PIton (VideoCore IV)
            ────────────────────                      ────────────────────
Python @triton.jit kernel                     Python @triton.jit kernel  (same source)
        │ code_generator.py                          │ frontend.py (symbolic AST execution)
        ▼                                            ▼
TTIR  (MLIR: tt, arith, scf dialects)          tile IR  (same dialects, ir.py)      ← "ttir"
        │ TTIR passes                                │ passes.py, reroll.py
        ▼                                            ▼
TTGIR (adds GPU layouts: blocked, mma...)      optimised tile IR                    ← "ttir_opt"
        │                                            │ backend/lower.py (layouts chosen here)
        ▼                                            ▼
LLVM IR (NVVM)                                 MIR on virtual registers             ← "mir"
        │                                            │ opt.py, presched.py, regalloc.py
        ▼                                            ▼
PTX → SASS (ptxas)                             MIR on physical registers            ← "mir_alloc"
                                                     │ sched.py (bundling), isa.py (encode)
                                                     ▼
                                               QPU machine words / .qasm / C launcher ← "qasm"
```

Every stage can be printed: `CompiledKernel.asm` has keys `ttir`, `ttir_rerolled` (when the
reroll pass fires), `ttir_opt`, `mir`, `mir_alloc`, `qasm`. From the command line, use
`--dump <stage>` on `piton.tools.compile` (section 7.2).

### 3.2 Level 1: tile IR (TTIR) — the dialects

| dialect | ops PIton uses | meaning |
|---|---|---|
| `tt` (Triton) | `tt.func`, `tt.get_program_id`, `tt.make_range`, `tt.splat`, `tt.expand_dims`, `tt.broadcast`, `tt.addptr`, `tt.load`, `tt.store`, `tt.dot`, `tt.reduce`, `tt.trans`, `tt.return` | tile/block operations, pointer arithmetic, memory |
| `arith` | `arith.constant`, `addi/addf`, `muli/mulf`, `divsi/remsi`, `cmpi/cmpf`, `select`, `minsi`, ... | element-wise scalar/tensor math |
| `scf` (structured control flow) | `scf.for` (with loop-carried values and `scf.yield`), `scf.if` | runtime loops and branches |

Types look like MLIR: `i32`, `f32`, `i1`, `!tt.ptr<f32>`, `tensor<64xf32>`,
`tensor<8x4x!tt.ptr<f32>>`.

How Python maps to it (`frontend.py`): `tl.constexpr` values are plain Python, so `if`/`for` on
them are resolved at compile time (unrolled). Anything depending on runtime data is a
`tl.tensor` wrapping an IR value; `for`/`if` on runtime values become `scf.for`/`scf.if`.

**Real output, the tutorial vector add (BLOCK_SIZE=64):**

```mlir
tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>, %output_ptr: !tt.ptr<f32>, %n_elements: i32) {
  %0 = tt.get_program_id  {axis = 0} : i32
  %1 = arith.constant  {value = 64} : i32
  %2 = arith.muli %0, %1 : i32
  %3 = tt.make_range  {start = 0, end = 64} : tensor<64xi32>
  %4 = tt.splat %2 : tensor<64xi32>
  %5 = arith.addi %4, %3 : tensor<64xi32>               // offs = pid*64 + arange
  %6 = tt.splat %n_elements : tensor<64xi32>
  %7 = arith.cmpi %5, %6 {predicate = "slt"} : tensor<64xi1>   // mask = offs < n
  %8 = tt.splat %x_ptr : tensor<64x!tt.ptr<f32>>
  %9 = tt.addptr %8, %5 : tensor<64x!tt.ptr<f32>>
  %10 = tt.load %9, %7 {has_mask = True, has_other = False} : tensor<64xf32>
  ...
  %14 = arith.addf %10, %13 : tensor<64xf32>
  tt.store %16, %14, %7 {has_mask = True}
  tt.return
}
```

**Real output, the matmul K-loop after optimisation** (note the mask has become the constant
`True`, because range analysis proved K=256 is a multiple of BLOCK_K):

```mlir
%69 = arith.constant  {value = True} : i1
%70 = tt.splat %69 : tensor<8x4xi1>
%72, %73, %74 = scf.for %63, %7, %33, %44, %62, %60 : tensor<8x4x!tt.ptr<f32>>, tensor<8x64xf32>, tensor<4x64x!tt.ptr<f32>> {
  ^bb(%75: i32, %76: tensor<8x4x!tt.ptr<f32>>, %77: tensor<8x64xf32>, %78: tensor<4x64x!tt.ptr<f32>>):
  %79 = tt.load %76, %70, %64 {has_mask = True, has_other = True} : tensor<8x4xf32>   // A tile
  %80 = tt.load %78, %71, %65 {has_mask = True, has_other = True} : tensor<4x64xf32>  // B tile
  %81 = tt.dot %79, %80, %77 : tensor<8x64xf32>                                       // acc += A·B
  %82 = tt.addptr %76, %66 : tensor<8x4x!tt.ptr<f32>>
  %83 = tt.addptr %78, %68 : tensor<4x64x!tt.ptr<f32>>
  scf.yield %82, %81, %83        // loop-carried: A pointers, accumulator, B pointers
}
```

### 3.3 Level 2: MIR (machine IR on virtual registers)

`backend/lower.py` turns each tile op into QPU operations on unlimited virtual registers
(`%name123`), organised in basic blocks with loop depth. It already speaks hardware: `unif`
(read next kernel argument), `elem_num` (lane id 0-15), `t0s`/`t1s` (send a TMU load request),
`ldtmu0` (receive), `vpm`/`vw_setup`/`vw_addr` (stores through VPM + DMA), `mutex`, condition
flags (`.setf`, `.ifz`). A 64-element tensor is 4 registers of 16 lanes.

Real excerpt (vector add):

```
entry:  (loop depth 0)
    mov %x_ptr141, unif                 # kernel arguments come from the uniform stream
    mov %n_elements144, unif
    mov %pid0145, unif
    mov %zero_ptr155, unif              # hidden zero page for masked-off lanes
main1:
    shl %s146, %pid0145, 6              # pid * 64
    mov %pat150, elem_num               # lane index
    ...
    min %c154, %pa153, %n_elements144
    sub.setf -, %c154, %n_elements144   # mask → condition flags
    mov.ifz %addr152, %zero_ptr155      # masked lanes read the zero page (other=0 for free)
    mov t0s, %addr152                   # TMU0 load request
    ...
    fadd %e194, %ld189, %ld193          # the actual x + y
    brr.allnz store_part2               # partial tile? take the masked store path
store_full4:                            # fast path: whole tile in bounds
    mov vw_setup, %vws216
    mov vpm, %e194                      # write 4 rows into this QPU's VPM rows
    ...
    mov -, mutex                        # take the hardware mutex (VDW stride is shared!)
    mov vw_addr, %sa223                 # kick the DMA
    mov mutex, 0
```

### 3.4 Level 3: allocated MIR

Same instructions, virtual registers replaced by physical ones: `r0-r3` (accumulators), `ra0-ra31`
(file A), `rb0-rb31` (file B). E.g. `mov %x_ptr141, unif` → `mov rb0, unif`.

### 3.5 Level 4: scheduled QPU code (.qasm)

`sched.py` packs one add-ALU op + one mul-ALU op + a signal into each 64-bit instruction
(`;` separates the two halves). Vector add compiles to ~140 instructions; the matmul to ~1,180.

```
:main1
    shl ra1, r0, 6; mov rb5, elem_num        # add ALU ; mul ALU in one instruction
    ldi rb4, 0x00000040
    add r2, r1, rb4; mov.ifz r3, ra0
    sub r1, ra1, -16; mov t0s, r3
```

The matmul hot loop (from the report) is the best slide for "what the compiler achieves":

```
nop; mov r5rep, r0 >> 8                       # broadcast A[m, k] to all 16 lanes
fadd ra10, ra10, r1; fmul r0, r5, rb25        # acc += prev product ; product = A * B[k, 0:16]
fadd rb8,  rb8,  r0; fmul r0, r5, ra2         #                                 B[k, 16:32]
fadd ra11, ra11, r0; fmul r0, r5, rb28        #                                 B[k, 32:48]
fadd rb9,  rb9,  r0; fmul r0, r5, ra8         #                                 B[k, 48:64]
add t1s, rb4, r2; mov r5rep, r1               # next TMU load + next broadcast
```

4 vector multiply-accumulates per 5 instructions; accumulators alternate between files A and B so
every pair uses one read port from each.

`isa.py` encodes these to 64-bit words; `tests/test_vc4asm.py` checks that the independent
`vc4asm` assembler produces **exactly the same bits** from the `.qasm` text.

---

## 4. Compiler optimisations

### 4.1 Tile-IR passes (`piton/passes.py`, `piton/reroll.py`)

`passes.optimize()` runs: canonicalize → up to 4 rounds of (LICM + broadcast sinking +
canonicalize) → range analysis → canonicalize with ranges → LICM → canonicalize →
rematerialisation across loops → DCE.

| pass | what it does | example payoff |
|---|---|---|
| **canonicalize** | constant folding, algebraic simplification, CSE, DCE | removes redundant index math |
| **CSE** / **DCE** | common-subexpression and dead-code elimination | |
| **LICM** | loop-invariant code motion out of `scf.for` | hoists stride/pointer setup out of the K loop |
| **broadcast sinking** | evaluates index arithmetic directly in the broadcast shape | avoids cross-lane shuffles |
| **range analysis** | integer interval analysis on values, incl. `program_id` (bounded by the launch grid) | folds masks known to be always true (matmul K-mask), enables cheap 24-bit multiplies |
| **prune loop args** | removes unused loop-carried values | fewer live registers |
| **remat across loops** | recomputes cheap values after a loop instead of keeping them live | lower register pressure |
| **drop dead `other`** | removes masked-load fill values no store reads | |
| **reroll** (`reroll.py`) | if a 1-D elementwise block would unroll past ~450 instructions, compile it as a loop over 64-element chunks; also row reductions (D1) | exp 1.76×, log 1.69×, idiv 1.66×; softmax 1.73× (D1) — all from fitting the ~500-instruction I-cache |

### 4.2 Lowering decisions (`backend/lower.py`)

* **Layouts.** Element *f* lives in lane *f* mod 16 of register ⌊*f*/16⌋. Values keep the cheapest
  exact form: scalar, splat, **affine** (`base + Σ coeffᵢ·indexᵢ`, so a `[4,64]` pointer tile is 4
  registers + 1 scalar, not 16 registers), dense, or a lazy predicate tree.
* **Loads** = TMU gathers; masked lanes read a zero page, so `other=0` is free. **Software
  pipelining**: each receive immediately issues the next request (incl. next iteration's), capped
  at 4 in flight per TMU. Worth ~8% on matmul.
* **Stores** = VPM rows + VDW DMA, under the hardware mutex (the stride register is shared). 2-D
  masks become row/column counts; a **versioned fast path** skips masking for full tiles.
* **`tl.dot`** = register-blocked outer product (hot loop above).
* **Reductions** = butterfly: rotate by 8, 4, 2, 1 lanes + add/max.
* **Transcendentals on the ALUs**, not the SFU: exp = range reduction + degree-6 polynomial;
  recip/rsqrt = integer seed + Newton steps. ~22-23 bits, ≤ 2 ulp, reproducible.
* **Exact 32-bit integer `//` and `%`** (C semantics), branch-free since C10.

### 4.3 Machine-level passes

| stage | file | what it does |
|---|---|---|
| MIR cleanup | `backend/opt.py` | DCE, copy coalescing, self-move removal, MIR LICM and CSE |
| pre-scheduling | `backend/presched.py` | interleaves independent dependence chains **before** allocation so the scheduler has pairs to pack; builds both orders, keeps the shorter (2-10% fewer instructions) |
| register allocation | `backend/regalloc.py` | no spilling. Accumulators for short-lived values; regfile A/B chosen by **2-colouring the "must differ" graph** (operands sharing an instruction must be in different files); accumulator copy where it can't. **Port-aware** mode (C1) re-allocates using pairs the scheduler missed. Late kernel args are re-read from the uniform stream (free spill slot). On `RegAllocError` the driver retries with less aggressive options. |
| scheduling | `backend/sched.py` | list scheduler over a dependence DAG whose edge latencies encode the pipeline rules (2-cycle regfile RAW, SFU→r4 3 cycles, etc.); packs add + mul + signal per instruction, only if it actually encodes; fills 3 branch-delay slots; fixes hazards across blocks. **Pairing** (C11): tries several priority functions, keeps shortest; for TMU-streaming loops keeps the one that waits least on loads. |
| encoding | `qpu/isa.py`, `qpu/asm.py` | 64-bit words + `.qasm` text, cross-checked against vc4asm |

### 4.4 The optimisation loop (11-03 → 11-05)

Method: every change behind a `PITON_*` env switch (switch off = bit-identical code); measured on
the Pi over a fixed 23-kernel suite; **accept only if some case is >2% faster and none is >2%
slower** (`perf/accept.py`, `perf/ab.py`). Accepted items:

| item | change | effect on the Pi |
|---|---|---|
| C1 | port-aware register allocation | Mandelbrot 1.22-1.27×, transpose 1.09× |
| A1 | shared 16-row VPM **store slots** under semaphores → 1 KiB DMAs instead of 256 B | copy 1.48×, sqrt 1.40×, exp 1.24× |
| A1 rule | slots only when stores ≥ 0.1 B per executed instruction | fixes matmul 1024³ regression |
| C10 | branch-free exact integer division | idiv 2.78× |
| E1/E4/E5 | **coarsening**: several program ids per hardware program, factor chosen per launch by bytes stored per instruction | add 1.21×, fma 1.31×, digits batch-16 1.31× |
| H2+H4 | Mandelbrot: cardioid/bulb test, early exit | 1.16-1.24× |
| D1 | reroll row reductions | softmax 1.73× |
| C7+C8 | issue efficiency (MIR CSE, cheaper special-value checks, either-ALU ops on mul ALU) | log 1.26×, exp 1.20× |
| B5 v3 | TMU_NOSWAP as per-kernel request (matmul only) | matmul 512³ 8.8 → 10.2 GFLOPS |
| C11 | scheduler pairing (on by default; `PITON_PAIR=0` off) | Mandelbrot 1.06×, compiles 3-6× slower |

Rejected (good discussion material): H6 (fewer instructions but **0.72× slower**: TMU stalls
doubled, which the emulator doesn't time), D2 (2-D reroll: transpose 0.71×), B5 v1/v2
(TMU_NOSWAP everywhere hurt reductions and digits), E1 v1 / E4 rule 2 (one global coarsening rule).

Suite geomean over the 11-03 baseline: 1.165× → 1.288× → 1.328× → 1.406× → **1.424×**.

**Lessons to say out loud:** measure the hardware, not the manual; a strict bit-exact emulator
makes hardware testing meaningful; fewer instructions ≠ faster (H6); some performance properties
depend on the launch, not the kernel (NOSWAP, coarsening).

---

## 5. Verification (how we know it's correct)

* **Strict emulator** (`qpu/emulator.py`): runs the real machine words on 12 interleaved QPUs,
  models measured float quirks bit-for-bit, and raises `HazardError` on anything the manual calls
  undefined (reading a regfile too early, TMU FIFO overflow, DMA races, deadlock...).
* **~1,400 pytest tests** (emulator vs numpy, small tiles, IEEE specials, division over every
  mantissa, one regression test per bug found in 5 review rounds, one test file per optimisation).
* **vc4asm cross-check**: our encoder and an independent assembler agree bit-for-bit.
* **Hardware tests** (`tests/test_hw.py`, 12): full outputs on the Pi == emulator, bit for bit.
* **Apps**: digits checks all 28,752 logits against float64 with a propagated rounding bound;
  Mandelbrot checks escape counts against the emulator every boot; benchmarks pre-fill outputs with
  NaN.

---

## 6. Results

### 6.1 Matrix multiplication (tutorial kernel, 8×64×8 tiles)

| N | 2026-10-30 | 2026-11-05 final | change |
|---|---|---|---|
| 128 | — | **9.38 GFLOPS** | |
| 512 | 8.78 | **10.15** | 1.16× |
| 1024 | 7.85 | **9.34** | 1.19× |

vs ARM (10-30, 512³): 31.7 ms GPU vs 352 ms tuned VFP assembly = **11.1×** (188× vs naive C).
Gap to 24 GFLOPS peak: A broadcasts (1 mul slot per 4 MACs), address math, B stream through
TMU/L2. Matmul is 64% busy, 21% TMU-stalled.

### 6.2 Digit classifier (`examples/digits_pi.py`)

| batch | QPU | ARM VFP | speedup |
|---|---|---|---|
| 1 | 169 µs | 719 µs | 4.25× |
| 16 | 230 µs | 1,452 µs | 6.31× |
| 64 | 489 µs | 4,524 µs | 9.25× |
| 256 | 1,478 µs | 18,556 µs | 12.55× |
| 1,797 | **9.320 ms** | 133.675 ms | **14.34×** |

96.995% accuracy (1,743/1,797), ~192,800 images/s.

### 6.3 Primitives vs CPU (selected; full table in README)

| primitive | QPU | vs best CPU |
|---|---|---|
| `tl.exp` | 1.61 ms | 35.6× |
| softmax | 0.56 ms | 32.5× |
| `//`, `%` | 3.17 ms | 17.0× |
| `tl.dot` 256³ | 3.99 ms | 12.1× |
| copy | 1.23 ms | 6.0× |
| `tl.trans` | 0.93 ms | 2.5× |

Takeaway: memory-bound work tops out at 6-9× (one shared store DMA engine); compute-heavy work
(transcendentals, dot) wins big.

### 6.4 Suite before/after (selected, same board)

idiv 3.08×, softmax 2.27×, mandel 1.53×, exp 1.54×, copy 1.49×, dense_relu 1.26×, matmul 1.19×,
compute_peak 1.00× (already at peak). Geomean **1.424×** with pairing. Full table: REPORT §7.1.

---

## 7. How to run everything yourself

### 7.1 One-time setup (host only, no Pi needed)

```bash
cd ~/Documents/github/PIton
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # numpy + pytest
brew install vc4asm        # optional: enables the encoder cross-check tests
```

### 7.2 See the IR at every level (best live demo, no Pi needed)

```bash
# tile IR before / after optimisation, MIR, allocated MIR, QPU assembly
for s in ttir ttir_opt mir mir_alloc qasm; do
  echo "===== $s"
  .venv/bin/python -m piton.tools.compile examples/kernels.py -n add_kernel --out-name add -o build \
    -s "x_ptr=*fp32,y_ptr=*fp32,output_ptr=*fp32,n_elements=i32" -c "BLOCK_SIZE=64" --dump $s
done

# the tutorial matmul (from the README)
.venv/bin/python -m piton.tools.compile examples/kernels.py -n matmul_kernel --out-name matmul -o build \
  -s "a_ptr=*fp32,b_ptr=*fp32,c_ptr=*fp32,M=i32,N=i32,stride_am=i32,stride_bk=i32,stride_cm=i32" \
  -c "K=256,stride_ak=1,stride_bn=1,stride_cn=1,BLOCK_SIZE_M=8,BLOCK_SIZE_N=64,BLOCK_SIZE_K=4,GROUP_SIZE_M=4" \
  --dump ttir_opt
```

Or from Python:

```python
import sys; sys.path.insert(0, "examples")
from piton import compiler
import kernels
ck = compiler.compile(kernels.add_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32"},
        {"BLOCK_SIZE": 64})
print(ck.asm["ttir"]); print(ck.asm["ttir_opt"]); print(ck.asm["mir"])
print(ck.asm["mir_alloc"]); print(ck.asm["qasm"])
```

`build/` also gets `add.qasm`, `add.c`, `add.h` (the AOT output with `add_launch()`).

### 7.3 Run a kernel on the emulator

```python
import numpy as np, piton as triton, sys; sys.path.insert(0, "examples")
import kernels
n = 1000; x = np.random.rand(n).astype(np.float32); y = np.random.rand(n).astype(np.float32)
out = np.zeros_like(x)
kernels.add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK_SIZE=256)   # add device="pi" for the board
print(np.allclose(out, x + y))
```

### 7.4 Tests

```bash
.venv/bin/python -m pytest -q tests/test_vector_add.py tests/test_kernels.py tests/test_vc4asm.py  # quick
.venv/bin/python -m pytest -q --ignore=tests/test_hw.py      # full host suite (long: ~1 h; ~5 h with pairing tests)
PITON_PAIR=0 .venv/bin/python -m pytest -q --ignore=tests/test_hw.py   # faster compiles, no pairing
.venv/bin/python -m pytest perf -q                           # the 23-case perf suite on the emulator
PITON_HW=1 .venv/bin/python -m pytest tests/test_hw.py       # hardware tests (Pi attached)
```

### 7.5 Examples on the host (emulator; no Pi)

```bash
.venv/bin/python examples/digits_model.py              # host accuracy of the frozen network (96.995%)
.venv/bin/python examples/image_filters.py             # 160x120 previews → build/
.venv/bin/python examples/life.py                      # 8 generations, all kernels
.venv/bin/python examples/life.py --images             # regenerate docs/images life gif/strip
.venv/bin/python examples/attention.py                 # → docs/images/attention-weights.png
.venv/bin/python perf/analyze.py exp -v                # instruction mix / pairing analysis of one kernel
```

`--build-only` / `--dry-run` on the `*_pi.py` scripts compile everything and build firmware without
booting (`--dry-run` also runs the ARM side natively for life/image_filters/attention):

```bash
.venv/bin/python examples/digits_pi.py --build-only
.venv/bin/python examples/image_filters_pi.py --dry-run
.venv/bin/python examples/life_pi.py --dry-run
.venv/bin/python examples/attention_pi.py --dry-run
```

### 7.6 Running on the Pi

**Prerequisites:** CS140E/CS240LX bare-metal setup — `pi-install`/`my-install` bootloader on the
SD card, `CS240LX_2025_PATH` set to the course repo (for `libpi`), SD card on **full firmware**
(`start.elf`, not `start_cd.elf`) with `gpu_mem=64` or more (otherwise V3D is off and
`V3D_IDENT0` reads `0xdeadbeef`). `PITON_TTY` can point at the serial device. A hung board can be
power-cycled with `tools/pi_power` (macOS) if the HUGE UART board's switch is wired up, otherwise
replug it.

| what | command |
|---|---|
| digit network | `.venv/bin/python examples/digits_pi.py --repeats 5` |
| image filters | `.venv/bin/python examples/image_filters_pi.py` (`--dump-images` reads images back) |
| Game of Life | `.venv/bin/python examples/life_pi.py` |
| attention | `.venv/bin/python examples/attention_pi.py` |
| Mandelbrot (HDMI, interactive) | `.venv/bin/python examples/mandelbrot_pi.py` — keys: `i`/`o` zoom, `wasd`/arrows pan, `r` reset, space tour, `q` quit |
| Mandelbrot benchmark only | `.venv/bin/python examples/mandelbrot_pi.py --bench-only` (or `--profile`) |
| matmul benchmark | `PITON_HW=1 .venv/bin/python benchmarks/matmul_pi.py 128 512 1024 --bk 8` |
| matmul vs ARM (bare metal) | `cd pi/examples/matmul && make MM=512 BK=8` |
| every primitive vs every CPU baseline | `cd pi/examples/primitives && make PYTHON=../../../.venv/bin/python` |
| compute peak (99% of 24 GFLOPS) | `PITON_HW=1 .venv/bin/python benchmarks/compute_peak_pi.py --unroll 124` |
| suite profile with V3D counters | `.venv/bin/python perf/profile_pi.py` |
| A/B a compiler switch | `PITON_PRESCHED=0 .venv/bin/python perf/profile_pi.py --label before` |
| hardware probes | `.venv/bin/python tools/probes.py <probe>` |

Saved results for every run are in `benchmarks/results/` and `perf/results/`, so you can show
the numbers without a board.

### 7.7 Useful switches

`PITON_PAIR=0` (no scheduler pairing, compiles 3-6× faster), `PITON_REROLL=0`,
`PITON_PRESCHED=0`, `PITON_PORT_ALLOC`, `PITON_STORE_SLOTS`, `PITON_ISSUE`, `PITON_TMU_NOSWAP`,
`PITON_HW=1` (enable hardware tests/benchmarks), `PITON_TTY`.

---

## 8. Showcase examples (one line each)

| example | shows | Pi result |
|---|---|---|
| **Image filters** (`examples/image_filters.py`) | 2-D stencils, masked borders, Gaussian blur (direct / separable / rolling rows in registers), Sobel, fused blur+Sobel | ~100 Mpixel/s at 640×480, **14-23×** best ARM; Pi images equal the model pixel for pixel |
| **Game of Life** (`examples/life.py`) | integer stencils on a torus, `tl.where` wrap-around, loop-carried rows, bit-sliced uint32 | **1.63-1.84 G cells/s**, 4.8-5.8× same algorithm on ARM |
| **Attention** (`examples/attention.py`) | flash-attention head: two `tl.dot`s, transposed K load, online softmax with `exp2`, causal mask | up to **3.7 GFLOPS**, **40-58×** C on ARM, max error 4·10⁻⁶ |
| **Digits MLP** (`examples/digits_pi.py`) | real trained network, fused bias+ReLU dense layers | 96.995%, 14.34× |
| **Mandelbrot** (`examples/mandelbrot_pi.py`) | interactive 640×480 HDMI render on QPUs | 92.6 fps at 1×, QPUs 97-99% busy |

---

## 9. Limitations (say these before someone asks)

* **No register spilling** — big tiles fail with `RegAllocError` (limits attention tiles to 2-4 rows).
* Triton subset: no `break`, power-of-two block shapes, **fp32/int32/uint32/bool only** (no fp16).
* Floats round toward zero (hardware); plain `+`/`*` keep NaN-as-inf.
* Matmul is ~42% of peak; transpose still a long straight-line program.
* Emulator timing is approximate (missed H6's TMU stalls).
* Pairing makes compiles 3-6× slower.

---

## 10. Likely questions and short answers

* **Do you use MLIR?** No dependency on it. The tile IR mirrors Triton's TTIR (an MLIR dialect) in
  structure and printed syntax; below it we wrote our own machine IR because the QPU has no LLVM
  backend.
* **Why not use the SFU for exp/recip?** It's ~13 bits accurate and its exact bits aren't
  reproducible, which would break the bit-exact emulator check. ALU sequences give ~2 ulp and
  determinism, and after the optimisation loop they run at the store-engine floor anyway.
* **Why is matmul only ~10 of 24 GFLOPS?** One mul slot per 4 MACs goes to the A broadcast, plus
  address math, plus the B stream through TMU/L2 (21% stall time).
* **What was the hardest bug / most surprising finding?** The VDW stride register is shared by all
  12 QPUs (stores corrupted until DMA kicks were put under the mutex); the TMU FIFO is 4 deep, not 8;
  the I-cache is ~500 instructions.
* **How do you know a speedup is real?** Off/on alternating boots, best-of-5 per boot, >2% gain
  with nothing >2% worse, re-measured when outside ±2%; boot noise is up to ±3%.
* **What does `kernel[grid]` actually do on the Pi?** Generates a bare-metal C program embedding
  the QPU code and inputs, builds it against libpi, sends it over UART, the ARM fills the V3D
  user-program queue, results come back over serial.
