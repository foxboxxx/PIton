# Checkpoint — where PIton stands (2026-11-05)

## Added 2026-11-04/05: the optimisation loop (`perf/`, `perf/BACKLOG.md`)
- **On by default:** C1, A1 (sparse-store rule, once-per-boot slot setup), C10, E1, H2, D1,
  C7 (`PITON_ISSUE`), E4 (`PITON_COARSEN_RUNTIME`), E5 (`PITON_COARSEN_STORES`), B5
  (`PITON_TMU_NOSWAP`). Suite of 23 kernels: **1.409× geomean** over 2026-11-03, every output
  checked (`perf/results/pi-2026-11-05-final.json`).
- **Matmul** (8×64×8, `benchmarks/matmul_pi.py --bk 8`): **8.21 GFLOPS at 128³, 10.15 at 512³,
  9.30 at 1024³** (`benchmarks/results/matmul-pi-2026-11-05.log`).
- **Digits network**, full batch, median of five: QPU **9.380 ms**, tuned ARM VFP 133.366 ms,
  **14.218×**, 191,578 images/s; all 28,752 logits checked
  (`benchmarks/results/digits-pi-2026-11-05-final.json`).
- **Primitives rerun** (`benchmarks/results/primitives-2026-11-05.log`): all 58 rows checked;
  copy 2.17 → 1.23 ms, exp 2.21 → 1.61 ms, idiv 8.54 → 3.17 ms, softmax 1.10 → 0.56 ms.
- **Tried and rejected on the Pi:** H6 (`PITON_MM`, matmul A in column blocks: 0.72-0.80×, TMU
  stalls double). **C11** (`PITON_PAIR=1`, scheduler pairing) passed its Pi gate on both boards (mandel
  1.06-1.07×, softmax 1.03×, suite geomean 1.010×, nothing below 0.98×) and is on by default
  (`PITON_PAIR=0` turns it off; compiling with it takes 3-6× longer).
- **Same board as the 11-03 baseline:** the final suite profile is **1.406×** (geomean, every case
  checked; `perf/results/pi-2026-11-05-final-sameboard.json`), **1.424×** with scheduler pairing (the default configuration).
- **Setup:** measured on a WSL2 host with a new UART head. `tools/pi_power_linux` power-cycles the
  Pi through the CP2102N's GPIO0, and the firmware is built against the cs240lx libpi with gcc 13.2
  (`-mfloat-abi=softfp`). `pi/examples/primitives/arm_vfp_ops.S` now also works under softfp. The
  same compiler timed on this setup and the earlier one agrees within 0.2% (geomean; cases
  0.97-1.035×). The digits and Pi numbers in the sections below are earlier checkpoints.
  `PITON_HW=1 pytest tests/test_hw.py`: **12 passed**.

## Added 2026-11-02/03: profiling, faster math kernels, Mandelbrot
- **Interactive Mandelbrot** (`examples/mandelbrot.py`, `examples/mandelbrot_pi.py`,
  `pi/examples/mandelbrot`): 640×480 on HDMI, keyboard over UART, every kernel configuration
  checked bit for bit against the emulator on each boot. 92.6 fps at 1×, 23.9 at 10×,
  10.9 at 10,000× (was 75.0 / 16.1 / 7.1); QPUs 97-99% busy.
- **`perf/`**: 23-case suite (emulator tests, instruction-mix analyser, Pi profiler with the V3D
  performance counters) and the hardware experiments behind the findings below.
- **Instruction cache**: straight-line programs over ~500 instructions miss once per cache
  line. `piton/reroll.py` loops large 1-D elementwise blocks over 64-element chunks:
  exp 3,041 → 1,730 µs (1.76×), log 1.69×, idiv 1.66×, recip 1.53×, rsqrt 1.47×, sqrt 1.40×.
- **`piton/backend/presched.py`**: interleaves dependence chains before allocation (2-10% fewer
  instructions; up to 1.13× on the Pi).
- **Launcher fix**: the 8-bit completed-program count could wrap unseen with 256+ short
  programs (the launch timed out); it is now read once per batch, 0.74 µs per program.
- **Primitives rerun** with the new compiler (`benchmarks/results/primitives-2026-11-03.log`):
  all 58 rows checked; exp, log, sqrt, rsqrt and `1/x` now run at copy speed (2.14-2.21 ms).
- `pytest`: **390 passed, 14 skipped** (hardware-only); `PITON_HW=1 pytest tests/test_hw.py`:
  **10 passed** after the compiler and launcher changes.
- The digits timings below are from 2026-10-30; a 2026-11-02 rerun (before
  the compiler changes) is in `benchmarks/results/digits-pi-2026-11-02.{json,log}`, within 1%.

## Done and verified (2026-10-30)
- Full pipeline: Triton AST frontend → tile IR + passes → QPU lowering → register allocation →
  scheduling → `.qasm` / machine words / C launcher (`python -m piton.tools.compile`).
- Emulator (`piton/qpu/emulator.py`) with strict hazard checks; models the measured float
  semantics (exact round-toward-zero, NaN-as-inf, +0 results, fmin/fmax ordering, ftoi wrap),
  4-deep TMU FIFOs, the shared VDW stride register, unif_addr A-space only.
- `pytest`: **355 software tests pass in 38.75 seconds** (`tests/test_tiles.py`: small-tile reductions,
  transposes, broadcasts, small dots, IEEE compares, exp/log/div/sqrt accuracy;
  `tests/test_regressions.py`: one test per bug found in review, incl. exact 32-bit division;
  trained dense-network inference and propagated error bounds, complete benchmark-log parsing,
  and bounded serial cleanup).
- Hardware (Pi Zero W), `PITON_HW=1 pytest tests/test_hw.py`: all 10 tests bit-identical to the
  emulator (vector add, softmax, where/row-sum, ragged matmul with 18 programs, exp/log,
  div/rsqrt/sqrt, float compares on specials, lane permutations, negative-stride stores,
  exact 32-bit integer division). Re-run on the final compiler: **10 passed in 333.13 seconds**.
- Code review: five review rounds (~30 + 10 + 17 + 13 + 10 confirmed bugs) fixed with regression
  tests; deliberate compile errors as in Triton: `return` inside any `for`, re-binding a tuple
  in a runtime loop, using a name assigned only inside a runtime loop / one branch of an if.
- **The unfinished tile sweep is complete:** all seven remaining configurations measured at
  512³ and 1024³; none beat **8×64×8**. Exact repeats, failures/retries and source identities
  are retained in `benchmarks/results/`; `tiling-summary-2026-10-30.json` collects the 14 successes.
- **Arithmetic: 23.794 GFLOPS, 99.14% of the 24 GFLOPS peak**, using compiler-generated
  register-resident recurrences, UNROLL=124. Five repeats: 65588, 65576, 65575, 65581, 65575 µs.
  All 768 outputs checked against an FP64 closed form with tolerance for accumulated RTZ
  rounding. This is a separate workload from matrix multiplication. UNROLL=64 reaches 23.614.
- **Final matmul measurements** (8×64×8, group=4, best of five, sampled final outputs checked):
  generic 512³ **8.781 GFLOPS**, generic 1024³ **7.854**; fixed-dimension 512³ **9.005**.
  Fixed dimensions reduce 512³'s program from 2699 to 831 instructions; they did not help 1024³.
- **Compiler improvements:** retain two A rotation registers when possible, with a fresh
  allocation attempt without the cache if other live values need those registers. Fill branch
  delay slots with bounded dependency/latency checks. The generic 8×64×8 loop shrinks
  **396 → 386 instructions**; 8×64×4 shrinks **198 → 196**. New numerical and hazard regressions pass.
- **Other approaches tested:** packed B, more grouped tile traversal, larger dimension-specialized
  tiles, and several compile-only operand/layout changes. None beat the selected matmul settings.
  Packed input preparation is measured separately and excluded from its GPU timer.
- **CPU comparison rechecked at 512³:** full GPU call 31.731 ms (8.460 GFLOPS), tuned ARM VFP
  352.343 ms (0.762 GFLOPS), **11.104× speedup**. All 262144 matrix outputs checked; max absolute
  error / max reference magnitude = 1.4e-5. Firmware clocks confirmed ARM 700, core/V3D 250,
  SDRAM 400 MHz. See `benchmarks/results/cpu-comparison-2026-10-30.log`.
- **Neural-network application complete and tested on the Pi:** a trained float32 MLP,
  **64→128 ReLU→64 ReLU→10 classes** (16 stored final outputs), recognizes **1,743/1,797
  held-out handwritten digits (96.995%)**. The official UCI 3,823/1,797 split is preserved;
  training and checkpoint selection use only the training split with 572 internal validation
  images. NumPy-only training, frozen weights, compressed data, checksums, metadata and
  CC BY 4.0 attribution are in `examples/digits_model.py` and `examples/data/digits/`.
- **NN correctness:** every one of the 28,752 full-test logits passes an independent float64
  reference with propagated float32 RTZ error bounds; all 1,797 predictions match exactly.
  Six full validations cover GPU/ARM C/ARM VFP before and after timing; 15 additional checks
  cover every implementation/batch pair. Maximum full-batch absolute errors: QPU **0.000102**,
  ARM C **0.000007**, ARM VFP **0.000009**. Strict-emulator tests also run all three trained
  layers on eight real test images. Parser tests reject missing/invalid records and failures.
- **NN performance:** full batch 1,797 (padded to 1,800), median of five: QPU **11.161 ms**,
  tuned ARM VFP **133.775 ms**, **11.986× speedup**, **161,007 images/s**. QPU repeats in µs:
  11210, 11161, 11149, 11185, 11153; ARM VFP: 133744, 133707, 133776, 133785, 133775.
  Logical single-image batch (four computed rows on each side): **199 µs** QPU vs **717 µs**
  ARM VFP, **3.603×**. Timing includes three launches, uniforms/cache clearing and fused
  bias/ReLU; upload, normalization, allocation, validation and serial output are excluded.
  Clocks: ARM 700, core/V3D 250, SDRAM 400 MHz. All five batch sizes (1, 16, 64, 256, 1797)
  were measured on GPU, blocked ARM C and ARM VFP. See
  `benchmarks/results/digits-pi-2026-10-30.{json,log}`.
- **Reliable experimentation:** resumable JSONL sweeps, all repeats and source/code fingerprints,
  finite/error/sample-count guards; compact firmware uniform initialization; bytewise serial
  reading fixes buffered-read timeouts; benchmark build invalidation tracks dimensions, tiles
  and compiler changes. One physical USB reconnect was needed during the tile-sweep phase.
- **Bounded serial cleanup:** the completed NN run's bootloader/serial-driver exit stalled;
  its results were saved when the connection recovered. A USB reconnect was confirmed. Cleanup now waits at
  most five seconds after TERM and one second after KILL, preserving completed output even
  if the serial driver remains blocked. The new cleanup regression tests pass.

## Open items
1. Repository and push access: https://github.com/foxboxxx/PIton (`main`). This checkpoint
   adds the Nov 2 profiling work, compiler passes and Mandelbrot demo to the Oct 30 state.
2. **Known limits** (README "Supported Triton"): masked stores whose tile index wraps past INT_MAX,
   a nested-loop int-then-float carry, no register spilling. Not fixed on purpose: an int argument
   equal to 1 is specialised and then follows Python `//`/`%` rules (as in Triton).
3. **Matrix multiplication still has a separate performance limit:** address work, broadcasts
   and memory traffic prevent it from matching the register-only arithmetic result. The final
   BK=8 loop's 256 vector MACs in 386 instructions imply a 15.917 GFLOPS instruction ceiling.
   128 compile-only shape/layout cases are saved; larger tiles generally exceed register limits.
   A specialized 4×64×32 fits, but the completed sweep favours 8×64×8.
4. **Next performance work** (measured limits, `perf/README.md`): elementwise and math kernels
   are bound by the shared VDW store engine (256-byte DMAs, ~640 MB/s; 1 KiB DMAs reach
   917-966 MB/s but need VPM rows shared between QPUs under semaphores). Mandelbrot is bound by
   register-file read ports (port-aware allocation, ~1.4× estimated). softmax and transpose
   still miss the instruction cache (reroll does not handle reductions or 2-D tiles).
5. `ftoi` of some huge values (e.g. −1.7e38 → 0x80000000) is not modelled; out-of-range
   float→int is undefined in Triton, everything else in `tools/float_specials_probe.py` matches.
6. Plain `+`/`*` keep the hardware's NaN-as-inf behaviour (documented); division, comparisons,
   exp/log/sqrt follow IEEE for NaN/inf.

## Useful commands
```bash
.venv/bin/python -m pytest -q --ignore=tests/test_hw.py     # emulator + vc4asm tests
PITON_HW=1 .venv/bin/python -m pytest tests/test_hw.py      # hardware tests (Pi attached)
.venv/bin/python perf/profile_pi.py                         # every kernel with V3D counters
.venv/bin/python examples/mandelbrot_pi.py                  # interactive Mandelbrot (HDMI)
.venv/bin/python tools/probes.py <probe>                    # hardware probes
PITON_HW=1 .venv/bin/python tools/float_specials_probe.py   # float special values vs emulator
PITON_HW=1 .venv/bin/python benchmarks/matmul_pi.py 128 512 --bk 8
.venv/bin/python examples/digits_model.py                  # frozen host NN accuracy
.venv/bin/python examples/digits_pi.py --build-only        # build NN firmware
.venv/bin/python examples/digits_pi.py --repeats 5          # run and validate the NN on the Pi
.venv/bin/python -m pytest tests/test_nn_kernels.py tests/test_digits_model.py tests/test_digits_pi.py -q
```
