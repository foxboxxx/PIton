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
  emulator; the `.qasm` we emit assembles with `vc4asm` to exactly the bits our encoder produces.
* On a Pi Zero W, vector add and the tutorial matmul (tile-aligned sizes, and ragged sizes with
  up to two programs) produce results **bit-identical to the emulator**.
* Ragged matmuls with many concurrent programs (up to 32 tested) are bit-identical too, after
  finding that the VDW stride register is shared by all QPUs (see the findings table).

## Performance (Pi Zero W, 12 QPUs @ 250 MHz, fp32)

Triton tutorial matmul (`examples/kernels.py`), square N×N×N, inputs generated on the board,
`benchmarks/matmul_pi.py`; best of 2-3 runs, sampled results checked against float64:

| N | config (BM, BN, BK) | time | GFLOPS |
|---|---|---|---|
| 128 | 8, 64, 4 | 0.49 ms | 8.5 |
| 256 | 8, 64, 4 | 4.33 ms | 7.8 |
| 512 | 8, 64, 4 | 34.5 ms | 7.8 |
| 512 | **8, 64, 8** | **30.9 ms** | **8.7** |
| 1024 | 8, 64, 4 | 352 ms | 6.1 |

For scale: the CS240LX GPU lab's hand-written 1-QPU kernel does 128³ in 116 ms (≈0.036 GFLOPS) and
the ARM1176 ≈0.009 GFLOPS — PIton is ~235× faster than the lab kernel on 128³. Theoretical peak is
24 GFLOPS (12 QPUs × 16 lanes × fmul+fadd × 62.5 M instr/s); the inner loop is at 65% of the ALU
peak, so the remaining gap is TMU/memory bandwidth (1024³ no longer fits the 128 KB L2).
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
| `tools/probes.py` | hand-written hardware probes that pinned down undocumented behaviour |

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
indexing, `tl.reshape/trans` (slow path), runtime `for` loops with loop-carried values, scalar
runtime `if`, inlined `@jit` helpers, Python `min/max` on tensors. Integer `//` and `%` follow C
semantics like Triton. Types: `float32`, `int32`, `uint32`, `int1` (the QPU has no fp16/fp64).
Block shapes must be powers of two; `tl.dot` is fastest when `BLOCK_N` is a multiple of 16.

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
  skips them when the whole tile is in bounds. Store sequences run under the hardware mutex.
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
* **Scheduling** packs add/mul ops, signals (`ldtmu`) and small immediates into 64-bit
  instructions under the pipeline rules (2-cycle regfile latency, rotation and SFU delays,
  3 branch delay slots, …) and fixes hazards across block boundaries.

## Things we learned from the hardware

All verified with `tools/probes.py` on a Pi Zero W; the emulator models them:

| finding | consequence |
|---|---|
| `fadd`/`fsub`/`fmul` round **toward zero** (not IEEE nearest) | emulator reproduces results bit for bit |
| only **4** direct TMU lookups may be outstanding per TMU (the guide implies 8; the 5th overwrites the oldest result) | pipeliner keeps ≤4 per TMU, 8 total using TMU0+TMU1 |
| `r5` written by `mov r5rep, rX << n` can be read by the very next instruction | broadcast + fmul back to back |
| vc4asm `r0 << n` rotates element n into element 0 | used for lane broadcasts |
| a horizontal VDW row longer than 16 words wraps inside the same VPM row | stores are 16-word rows + stride |
| write address 40 is `unif_addr` only in regfile-**A** space (B space is `unif_addr_rel`) | a mul-ALU write needs the write-swap bit — found when argument reloads returned garbage |
| QPU instruction rate 62.5 M/s per QPU, 12 QPUs scale linearly; TMU ≈ 12 cycles latency | timing model |
| the VDW **stride** register is one register for all 12 QPUs: written once, it is overwritten by other QPUs before the next DMA (probe `vdw_stride`: 592/1536 rows wrong) | each DMA kick (setup + stride + address) runs under the hardware mutex, released as soon as the DMA starts (safe per probe); the emulator models the shared register |
| `qpu_num` is unique 0-11; the scheduler hands programs to QPU 11, 10, …, 0 | per-QPU VPM rows `4*qpu_num` never collide |

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
