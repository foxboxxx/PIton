# PIton — Triton for the Raspberry Pi GPU

PIton compiles [Triton](https://github.com/triton-lang/triton) kernels to machine code for the
**VideoCore IV QPUs** of the Raspberry Pi Zero / Pi 1 (BCM2835: 12 QPUs, 16-wide SIMD, 250 MHz).
The same Python source you would write for an NVIDIA GPU, e.g. the Triton tutorial matmul, is
lowered through a Triton-style tile IR to scheduled QPU instructions, emitted as `vc4asm`
assembly (`.qasm`) and as a C array + launcher for bare-metal programs.

```python
import piton as triton            # or: write the kernel against real `triton`
import piton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)               # QPU emulator
add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256, device="pi")  # real Pi over UART
```

Status:

* The Triton tutorial **vector add**, **fused softmax** and **grouped matmul** (`tl.dot`, masks,
  `%`-wraparound, runtime strides, runtime or constexpr K) compile and pass against numpy in the
  emulator (238 tests, including small-tile reductions/transposes/dots, IEEE comparisons with
  NaN/inf/±0, exp/log/div/sqrt accuracy, exact 32-bit integer division, and regression tests
  for every bug found in review); the `.qasm` we emit assembles with `vc4asm` to
  exactly the bits our encoder produces.
* On a Pi Zero W every hardware test is **bit-identical to the emulator**: vector add, softmax,
  ragged matmuls with up to 32 concurrent programs, exp/log/division/sqrt, float comparisons on
  special values, lane permutations, stores with negative row strides, and exact 32-bit integer
  division (`PITON_HW=1 pytest tests/test_hw.py`, 10 tests). The emulator models the QPU's float
  quirks measured with `tools/float_specials_probe.py` (see the findings table).
* No QPU op whose result we cannot reproduce is used: exp, log, reciprocal and rsqrt run on the
  ALUs (≈22-23 bits, within ~2 ulp) instead of the special function unit.

## Performance (Pi Zero W, 12 QPUs @ 250 MHz, fp32)

Triton tutorial matmul (`examples/kernels.py`), square N×N×N, inputs generated on the board,
`benchmarks/matmul_pi.py`; best of 2-3 runs, sampled results checked against float64:

| N | config (BM, BN, BK) | time | GFLOPS |
|---|---|---|---|
| 128 | 8, 64, 4 | 0.52 ms | 8.1 |
| 256 | 8, 64, 4 | 4.38 ms | 7.7 |
| 512 | 8, 64, 4 | 34.8 ms | 7.7 |
| 512 | **8, 64, 8** | **30.7 ms** | **8.8** |
| 1024 | 8, 64, 4 | 352 ms | 6.1 |
| 1024 | 8, 64, 8 | 270 ms | 8.0 |

(Measured 2026-10-27 after the correctness fixes; with a constant K the loop is unchanged at 198
instructions. With a runtime K, clamping the prefetch index and the overflow-safe K-mask compare
add 4. Exact 32-bit `//` and `%` in the tutorial's program-id arithmetic and `% M` wrap-around
cost ~3% at 128³, where per-program setup is a larger share; the full-range division path is
placed out of line, after the program's end.)

### Speedup over the ARM CPU

The same board's ARM1176JZF-S on the same inputs, bare metal (`pi/examples/matmul`,
`make MM=512 BK=8`). GPU time is the whole `matmul_launch()` call (writing every program's
uniforms, clearing the GPU caches, queueing, waiting); GFLOPS = 2N³ / time:

| N | PIton, 12 QPUs | ARM blocked, VFP vector asm | ARM blocked C | ARM naive i-k-j | vs best ARM | vs ARM peak |
|---|---|---|---|---|---|---|
| 128 | 0.55 ms (7.6) | 6.1 ms (0.69) | 9.2 ms (0.45) | 71 ms (0.059) | **11.0×** | 5.4× |
| 256 | 4.29 ms (7.8) | 46 ms (0.73) | 71 ms (0.48) | 0.75 s (0.045) | **10.7×** | 5.6× |
| 512 | 31.7 ms (8.5) | 352 ms (0.76) | 547 ms (0.49) | 6.1 s (0.044) | **11.1×** | 6.0× |
| 1024 | 272 ms (7.9) | 2.86 s (0.75) | 4.41 s (0.49) | 50 s (0.043) | **10.5×** | 5.6× |

* **The ARM baselines** run with the MMU on so the 16 KB L1 data cache is used (ARM RAM
  write-back cached, GPU memory and peripherals uncached), branch prediction and the I-cache on,
  VFP RunFast mode, `-O3`. *Blocked C*: packed A/B panels, a 4×4 register-blocked micro-kernel,
  K blocks of 64 and column blocks of 32 (best of six (KC, NC) pairs swept at 512³). *Vector
  asm*: the same blocking with an assembly micro-kernel in VFP11 short-vector mode (one `vmla`
  = 4 multiply-accumulates) plus `pld` prefetch; it runs at 1.1–1.2 GFLOPS when its data is in
  L1 (78–84% of peak) and at 0.69–0.76 on the real matrices (49–54%), where the ARM (no L2, no
  hardware prefetch) waits on DRAM. *Naive* is the plain i-k-j loop, unoptimised, for reference
  only (the GPU is 128–191× faster than it).
* **ARM peak** is 2 FLOP/cycle (one single-precision `vmla` per cycle) × 700 MHz = 1.4 GFLOPS:
  the "vs ARM peak" column bounds what any ARM code could do at this clock.
* **Clocks** are what the course firmware sets (read through the mailbox, ARM cross-checked with
  the cycle counter): ARM 700 MHz, core 250 MHz, V3D 250 MHz (firmware max 300), SDRAM 400 MHz;
  128 MB ARM / 128 MB GPU. A stock Pi Zero W clocks the ARM at 1 GHz, which would make the
  ARM up to 1.43× faster (peak 2.0 GFLOPS); that configuration is not measured here.
* **Like for like**: the GPU column uses the faster tile config per size (BK = 8 except BK = 4
  at 128³); with BK = 4 everywhere the speedups over the best ARM are 11.0, 10.1, 9.9 and 8.0×.
  The GPU kernel is compiled for the constant K; the ARM code takes runtime sizes.
* **Method**: inputs are generated once and copied to both sides; best of 3 runs for the GPU and
  the blocked ARM versions (naive: best of 3 up to 256³, one run above); the GPU's C is
  pre-filled with NaN, and every result is compared with blocked C over all N² entries
  (max |error| / max |C| ≤ 3·10⁻⁵, NaN-aware).

An earlier README line put the ARM at ~0.009 GFLOPS: that was a naive dot product through
`volatile` pointers with the data cache off, 80× slower than the blocked baseline above. For
scale the other way, the CS240LX GPU lab's hand-written 1-QPU kernel does 128³ in 116 ms
(≈0.036 GFLOPS).

Where the time goes (512³, measured on the board):

| experiment | GFLOPS |
|---|---|
| theoretical peak: 12 QPUs × 16 lanes × (fmul + fadd) × 62.5 M instr/s | 24 |
| inner loop alone: 128 MACs in 198 instructions (65%) | 15.6 |
| same kernel, all loads cache-resident (strides 0) | 14.1 |
| same kernel, A real but B cache-resident | 12.6–13.1 |
| real data (default `0x4` bus alias) | 8.7 |
| real data through the `0x8` alias | 8.2 |
| real data through the uncached `0xC` alias | < 0.1 |

So the compute side is within 10% of what the instruction schedule allows, and the rest is the
B-matrix stream through the TMU and L2 (each 8×64 tile reads a 128 KB B panel; GROUP_SIZE_M from
4 to 64 does not change it). At 1024³ the panels no longer fit at all and throughput drops to 6.1
with BK=4; BK=8 (fewer, larger loads per iteration) keeps it at 8.0 (that row was measured with
the bare-metal benchmark in `pi/examples/matmul`, same kernel timer).
Software pipelining of the K-loop loads is worth ~8% (7.2 → 7.8 GFLOPS at 512³ with `num_stages=1`).
Accumulation uses the QPU's round-toward-zero adds: at K=1024 the sampled error is ~1e-4 relative.

## Pipeline

```
Python AST ──frontend.py──▶ tile IR (TTIR-like)  ──passes.py──▶ optimised tile IR
   (Triton semantics:          tt.load/store/dot,      CSE, LICM, range analysis (mask folding),
    constexpr, scf.for/if)     scf.for, arith.*        broadcast sinking, rematerialisation
                                                                        │
                                                               backend/lower.py
                                                                        ▼
   QPU program ◀─sched.py── allocated MIR ◀─regalloc.py── machine IR on virtual registers
 (.qasm / words / C)   list scheduling,        accumulators + A/B          affine index math, TMU gathers,
                       add/mul pairing,        register files,             software-pipelined loads,
                       hazards, delay slots    port constraints            VPM/VDW stores, dot micro-kernel
```

| module | what it does |
|---|---|
| `piton/language` | Triton language subset (`tl.*` builtins, `tensor` with Triton's broadcasting/promotion rules) |
| `piton/frontend.py` | executes the kernel's Python AST symbolically, like Triton's `code_generator.py` |
| `piton/ir.py`, `piton/passes.py` | tile IR + optimisation passes; `fn.dump()` prints TTIR-style text |
| `piton/backend/lower.py` | tile IR → machine IR (layouts, loads/stores, `tl.dot`, reductions, division, loops) |
| `piton/backend/regalloc.py` | register allocation for r0-r3 + regfiles A/B with read-port constraints |
| `piton/backend/sched.py` | hazard-aware list scheduler / bundler, branch delay slots |
| `piton/qpu/isa.py`, `asm.py` | instruction encoding/decoding, vc4asm-compatible text |
| `piton/qpu/emulator.py` | V3D emulator (12 QPUs, TMU, SFU, VPM, VDW/VDR, mutex, scheduler queue) with hazard checks |
| `piton/runtime/launcher.py` | `kernel[grid](...)` for numpy arrays (emulator or Pi) |
| `piton/runtime/pi.py` | builds a bare-metal test program, boots it with the course bootloader, reads results back |
| `piton/tools/compile.py` | AOT compiler → `.qasm` + `.c/.h` with a `*_launch()` function |
| `pi/runtime/piton_rt.c` | mailbox GPU memory + V3D user-program queue launcher |
| `tools/probes.py`, `tools/float_specials_probe.py` | hardware probes that pinned down undocumented behaviour |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                 # emulator + vc4asm tests
brew install vc4asm                           # optional: cross-checks our encoder
```

Inspect what the compiler produces:

```bash
python3 -m piton.tools.compile examples/kernels.py -n matmul_kernel --out-name matmul -o build \
  -s "a_ptr=*fp32,b_ptr=*fp32,c_ptr=*fp32,M=i32,N=i32,stride_am=i32,stride_bk=i32,stride_cm=i32" \
  -c "K=256,stride_ak=1,stride_bn=1,stride_cn=1,BLOCK_SIZE_M=8,BLOCK_SIZE_N=64,BLOCK_SIZE_K=4,GROUP_SIZE_M=4" \
  --dump ttir_opt
```

`CompiledKernel.asm` also has `ttir`, `mir` (before register allocation), `mir_alloc` and `qasm`.

### Running on a Pi

The Pi setup is the CS140E/CS240LX one: bare metal, `my-install`/`pi-install` bootloader,
`CS240LX_2025_PATH` pointing at the course repo (for `libpi`). The SD card must run the full
firmware (`start.elf`, **not** `start_cd.elf`) with `gpu_mem=64` or more, otherwise the V3D
block is powered off (`V3D_IDENT0` reads `0xdeadbeef`).

* From Python: `kernel[grid](..., device="pi")` (inputs are embedded in the boot image, so keep
  them small — ~11 KB/s over the UART), or `piton.runtime.pi.bench_matmul_on_pi` which generates
  inputs on the board and checks sampled outputs.
* Bare metal: `pi/examples/matmul` compiles the tutorial matmul AOT and benchmarks it against the ARM:
  `cd pi/examples/matmul && make MM=256`.
* `PITON_HW=1 pytest tests/test_hw.py` runs the hardware tests.

## Supported Triton

`tl.program_id`, `tl.num_programs`, `tl.arange`, `tl.zeros/full`, `tl.load/store` (masks, `other`),
`tl.dot`, `tl.where`, `tl.sum/max/min`, `tl.exp/exp2/log/log2/sqrt/rsqrt/abs/sigmoid`,
`tl.maximum/minimum`, `tl.cdiv`, `tl.static_range`, `tl.multiple_of` (no-op), `.to()`, `[:, None]`
indexing, `tl.reshape/trans`, runtime `for` loops with loop-carried values (including swaps),
runtime `if` (also the `if pid >= n: return` early exit), inlined `@jit` helpers, Python
`min/max` on tensors. Integer `//` and `%` follow C semantics like Triton and are exact for all 32-bit values
(a float estimate refined with the exact remainder; a uniform branch takes a short path when
all operands are below 2²³); integer and float comparisons are exact (no overflow, IEEE
NaN/inf/±0). Types: `float32`, `int32`, `uint32`, `int1`
(the QPU has no fp16/fp64). Block shapes must be powers of two; `tl.dot` needs `BLOCK_N` dividing
16 or a multiple of 16 (fastest). Variables follow Python semantics across runtime loops and ifs:
numbers carried through a loop widen (bool → int32 → uint32 → float32), the loop variable keeps
its last value after the loop. As in Triton, `return` inside any `for` is rejected, a carried
tensor cannot change type, re-binding a tuple inside a runtime loop and using a name assigned
only inside a runtime loop (or one branch of an `if`) are compile errors, and loop bounds must
fit int32. Floats round toward zero and denormals flush to zero, as on the hardware.

Known limits: masked tile stores assume the masked index (`base + arange`) does not wrap past
INT_MAX inside one tile (offsets that large cannot address memory on the Pi); a carried Python
number that first becomes an int tensor in a nested loop and later a float is rejected rather
than promoted; there is no register spilling, so very large tiles fail with `RegAllocError`.

## How it maps to the QPU

* **Layout.** A block of N elements lives in ⌈N/16⌉ 16-lane registers, row-major. Index tensors
  (`arange`, `program_id` arithmetic, pointer offsets) stay symbolic as
  `dense + base + Σ coeffᵢ·indexᵢ`, so a `[4, 64]` pointer tile costs four lane registers
  plus one loop-carried scalar instead of sixteen registers.
* **Loads** are TMU gathers (16 independent addresses per request). Masked-off lanes read a
  hidden zero page, so `other=0` needs no fix-up. Loads in loops whose addresses depend only on
  loop-carried pointers are **software pipelined**: every TMU receive immediately refills the
  FIFO with the next request, including the first ones of the next iteration (clamped on the last).
* **Stores** go through this QPU's 4 VPM rows and the VDW DMA, as 16-word rows with a stride.
  `(row < M) & (col < N)` masks are turned into runtime row/column counts; a versioned fast path
  skips them when the whole tile is in bounds. The VDW stride field holds 16 unsigned bits, so a
  runtime row stride is checked once per store and huge or negative strides fall back to one DMA
  per row. Each DMA kick runs under the hardware mutex.
* **Small tiles** (rows shorter than 16 lanes, transposes, broadcasts, `tl.dot` with N < 16) are
  built with lane rotations and masked moves (`shl.setf -, mask, elem_num` selects lanes), so
  only the source and result registers are live.
* **`tl.dot`** is a register-blocked outer product: for each `k` and row `m`,
  `mov r5rep, a << lane` broadcasts `A[m,k]` (one mul-ALU op), then one *paired* instruction per
  16 columns does `fmul prod, r5, B[k, cb]` + `fadd acc, acc, prev_prod`. With `BLOCK_N = 64` that
  is 4 multiply-accumulates of 16 lanes per 5 instructions, while B rows stream in through the
  pipelined TMU loads. The 32 accumulator registers are split across regfiles A and B by the
  allocator so every pair reads one register from each file.
* **Registers.** 4 accumulators + 2×31 regfile entries. The allocator 2-colours the "these two
  operands share an instruction" graph to respect the one-read-port-per-file rule, and inserts
  accumulator copies when it cannot. Kernel arguments needed only after a loop are dropped and
  re-read from the program's own uniform list (`unif_addr` reset) — a free spill slot.
* **Ranges.** `kernel[grid](...)` compiles for its grid (rounded up to a power of two), so range
  analysis can bound `program_id` arithmetic (narrower compares, 24-bit multiplies);
  `piton.compiler.compile(..., grid=None)` and the AOT tool without `--max-grid` assume any
  int32 program id.
* **Scheduling** packs add/mul ops, signals (`ldtmu`) and small immediates into 64-bit
  instructions under the pipeline rules (2-cycle regfile latency, rotation and SFU delays,
  3 branch delay slots, …) and fixes hazards across block boundaries.

## Things we learned from the hardware

All verified with `tools/probes.py` on a Pi Zero W; the emulator models them:

| finding | consequence |
|---|---|
| `fadd`/`fsub`/`fmul`/`itof` round **toward zero** (not IEEE nearest), on the exact result | emulator reproduces results bit for bit (sums are truncated exactly, using TwoSum) |
| float specials (`tools/float_specials_probe.py`): results ≥ 2¹²⁸ are ±inf, results below 2⁻¹²⁶ and all zero results are +0, denormal operands read as +0, NaN operands act as ±inf, `0 × inf = +0`, `inf + −inf` is the second operand; `fmin`/`fmax` order by sign-magnitude bits (+NaN largest, −NaN smallest, −0 = +0, denormals positive) | emulated exactly; float compares use integer keys so `<`, `==`, … follow IEEE (NaN, ±inf, ±0); division sets NaN for NaN operands, 0/0 and inf/inf; plain `+`/`*` keep the hardware's NaN-as-inf behaviour |
| `ftoi` truncates and keeps the low 32 bits (`1e10` → `0x540be400`, inf/NaN → 0), with exceptions for some huge values | float → int of values ≥ 2³¹ is undefined in Triton; the emulator models the wrap |
| only **4** direct TMU lookups may be outstanding per TMU (the guide implies 8; the 5th overwrites the oldest result) | pipeliner keeps ≤4 per TMU, 8 total using TMU0+TMU1 |
| `r5` written by `mov r5rep, rX << n` can be read by the very next instruction | broadcast + fmul back to back |
| vc4asm `r0 << n` rotates element n into element 0 | used for lane broadcasts |
| a horizontal VDW row longer than 16 words wraps inside the same VPM row | stores are 16-word rows + stride |
| write address 40 is `unif_addr` only in regfile-**A** space (B space is `unif_addr_rel`) | a mul-ALU write needs the write-swap bit — found when argument reloads returned garbage |
| QPU instruction rate 62.5 M/s per QPU, 12 QPUs scale linearly; TMU ≈ 12 cycles latency | timing model |
| the VDW **stride** register is one register for all 12 QPUs: written once, it is overwritten by other QPUs before the next DMA (probe `vdw_stride`: 592/1536 rows wrong) | each DMA kick (setup + stride + address) runs under the hardware mutex, released as soon as the DMA starts (safe per probe); the emulator models the shared register |
| `qpu_num` is unique 0-11; the scheduler hands programs to QPU 11, 10, …, 0 | per-QPU VPM rows `4*qpu_num` never collide |
| SFU accuracy: `exp2` ≈ 13.8 bits, `log2` ≈ 12.8 bits; `recip`/`recipsqrt` + one Newton step ≈ 22 bits, and the exact bits are not reproducible (softmax differed from the emulator in the last bit) | exp/log use range reduction + polynomials, recip/rsqrt an integer seed + 3 Newton steps, all on the ALUs: deterministic and ≈ 2 ulp |
| GPU reads through the uncached `0xC` bus alias are >100× slower; `0x0` hung the TMU | buffers use the mailbox's default `0x4` alias |

## Layout of the repository

```
piton/            compiler, emulator, runtime (Python)
examples/         Triton kernels: tutorials (vector add, softmax, matmul) and test kernels
tests/            pytest: emulator vs numpy, vc4asm bit-exactness, hardware (PITON_HW=1)
tools/probes.py   hardware probes
pi/runtime/       bare-metal runtime (mailbox + V3D queue)
pi/examples/      bare-metal programs using AOT-compiled kernels
```

## Credits

The mailbox / V3D launch sequence follows the CS240LX GPU lab (Rohan Chanani and Asanshay Gupta).
Hardware documentation: Broadcom *VideoCore IV 3D Architecture Reference Guide* (2013).
Assembler cross-checks: [vc4asm](https://maazl.de/project/vc4asm/doc/index.html) by Marcel Müller.
