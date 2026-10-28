# Checkpoint — where PIton stands (2026-10-27)

## Done and verified
- Full pipeline: Triton AST frontend → tile IR + passes → QPU lowering → register allocation →
  scheduling → `.qasm` / machine words / C launcher (`python -m piton.tools.compile`).
- Emulator (`piton/qpu/emulator.py`) with strict hazard checks; models the measured float
  semantics (exact round-toward-zero, NaN-as-inf, +0 results, fmin/fmax ordering, ftoi wrap),
  4-deep TMU FIFOs, the shared VDW stride register, unif_addr A-space only.
- `pytest`: 238 emulator / vc4asm tests pass (`tests/test_tiles.py`: small-tile reductions,
  transposes, broadcasts, small dots, IEEE compares, exp/log/div/sqrt accuracy;
  `tests/test_regressions.py`: one test per bug found in review, incl. exact 32-bit division).
- Hardware (Pi Zero W), `PITON_HW=1 pytest tests/test_hw.py`: all 10 tests bit-identical to the
  emulator (vector add, softmax, where/row-sum, ragged matmul with 18 programs, exp/log,
  div/rsqrt/sqrt, float compares on specials, lane permutations, negative-stride stores,
  exact 32-bit integer division).
- Code review: five review rounds (~30 + 10 + 17 + 13 + 10 confirmed bugs) fixed with regression
  tests; deliberate compile errors as in Triton: `return` inside any `for`, re-binding a tuple
  in a runtime loop, using a name assigned only inside a runtime loop / one branch of an if.
- Benchmarks re-measured (README table): 8.1 GFLOPS at 128³, 8.8 at 512³ with BK=8, 8.0 at 1024³ with BK=8.
- CPU baseline (README "Speedup over the ARM CPU", `pi/examples/matmul`): 10.5–11.1× over a blocked
  VFP short-vector SGEMM on the same Pi's ARM1176 (caches on, 0.69–0.76 GFLOPS), 5.4–6.0× over
  the ARM's 1.4 GFLOPS peak at the course's 700 MHz.

## Open items
1. **Push access** (deferred by the user). `git push` fails: this machine authenticates as
   `AdamAlhousiki`, which has no write access to `foxboxxx/PIton`. Add that account as a
   collaborator, then `git push -u origin main`.
2. **Known limits** (README "Supported Triton"): masked stores whose tile index wraps past INT_MAX,
   a nested-loop int-then-float carry, no register spilling. Not fixed on purpose: an int argument
   equal to 1 is specialised and then follows Python `//`/`%` rules (as in Triton).
3. **Tile sweep (stopped half way, 2026-10-27).** Compile-only scan: 8×64 is the most efficient
   tile that fits in registers (0.65 multiply-adds per instruction); 16×64, 8×128 and bigger fail
   with RegAllocError (no spilling). Measured on the Pi (GFLOPS at 512³ / 1024³, kernel time):
   8×64×4 7.70 / 6.11; 8×64×8 8.76 / 7.86; 4×64×4 4.88 / 4.13. Not run yet: 4×64×8, 4×64×16,
   4×128×4, 4×128×8, 16×32×4, 8×32×4, 8×32×8. Resume with, per config:
   `PITON_HW=1 .venv/bin/python benchmarks/matmul_pi.py 512 1024 --bm 4 --bn 128 --bk 8`.
   If a tile beats 8×64×8, rerun the CPU comparison with it: `cd pi/examples/matmul && make
   MM=512 BM=.. BN=.. BK=..` (README "Speedup over the ARM CPU").
4. **Inner-loop tuning.** Matmul loop body is 198 instructions for 128 16-lane MACs (65% of the
   ALU peak) at BM=8, BN=64, BK=4; ideal is 160 (80%). Ideas: receive B rows straight into MAC
   operands, reuse one address register per column block, autotune (BM, BN, BK) on the Pi.
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
```
