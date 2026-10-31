# Checkpoint — where PIton stands (2026-10-30)

## Done and verified
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
   includes the completed Oct 30 compiler tuning, trained neural network and measurements.
2. **Known limits** (README "Supported Triton"): masked stores whose tile index wraps past INT_MAX,
   a nested-loop int-then-float carry, no register spilling. Not fixed on purpose: an int argument
   equal to 1 is specialised and then follows Python `//`/`%` rules (as in Triton).
3. **Matrix multiplication still has a separate performance limit:** address work, broadcasts
   and memory traffic prevent it from matching the register-only arithmetic result. The final
   BK=8 loop's 256 vector MACs in 386 instructions imply a 15.917 GFLOPS instruction ceiling.
   128 compile-only shape/layout cases are saved; larger tiles generally exceed register limits.
   A specialized 4×64×32 fits, but the completed sweep favours 8×64×8.
4. **Next application, chosen by the user:** a Mandelbrot renderer with zoom controlled over
   UART and optionally GPIO buttons. Small neural-network inference is complete and hardware
   verified; further model/application work can build on the resident dense-layer pipeline.
5. `ftoi` of some huge values (e.g. −1.7e38 → 0x80000000) is not modelled; out-of-range
   float→int is undefined in Triton, everything else in `tools/float_specials_probe.py` matches.
6. Plain `+`/`*` keep the hardware's NaN-as-inf behaviour (documented); division, comparisons,
   exp/log/sqrt follow IEEE for NaN/inf.

## Useful commands
```bash
.venv/bin/python -m pytest -q --ignore=tests/test_hw.py     # emulator + vc4asm tests
PITON_HW=1 .venv/bin/python -m pytest tests/test_hw.py      # hardware tests (Pi attached)
.venv/bin/python tools/probes.py <probe>                    # hardware probes
PITON_HW=1 .venv/bin/python tools/float_specials_probe.py   # float special values vs emulator
PITON_HW=1 .venv/bin/python benchmarks/matmul_pi.py 128 512 --bk 8
.venv/bin/python examples/digits_model.py                  # frozen host NN accuracy
.venv/bin/python examples/digits_pi.py --build-only        # build NN firmware
.venv/bin/python examples/digits_pi.py --repeats 5          # run and validate the NN on the Pi
.venv/bin/python -m pytest tests/test_nn_kernels.py tests/test_digits_model.py tests/test_digits_pi.py -q
```
