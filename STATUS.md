# Checkpoint — where PIton stands (2026-10-27)

## Done and verified
- Full pipeline: Triton AST frontend → tile IR + passes → QPU lowering → register allocation →
  scheduling → `.qasm` / machine words / C launcher (`python -m piton.tools.compile`).
- Emulator (`piton/qpu/emulator.py`) with strict hazard checks; models the measured float
  semantics (exact round-toward-zero, NaN-as-inf, +0 results, fmin/fmax ordering, ftoi wrap),
  4-deep TMU FIFOs, the shared VDW stride register, unif_addr A-space only.
- `pytest`: 118 emulator / vc4asm tests pass (`tests/test_tiles.py` covers small-tile reductions,
  transposes, broadcasts, small dots, IEEE compares, exp/log/div/sqrt accuracy).
- Hardware (Pi Zero W), `PITON_HW=1 pytest tests/test_hw.py`: all 9 tests bit-identical to the
  emulator (vector add, softmax, where/row-sum, ragged matmul with 18 programs, exp/log,
  div/rsqrt/sqrt, float compares on specials, lane permutations, negative-stride stores).
- Code review: all ~30 confirmed bugs from the review workflow fixed; their repro scripts pass
  (two now raise deliberate compile errors, as Triton does: `return` inside a runtime `for`,
  re-binding a tuple in a runtime loop).
- Benchmarks re-measured (README table): 8.5 GFLOPS at 128³, 8.8 at 512³ with BK=8, 6.2 at 1024³.

## Open items
1. **Push access** (deferred by the user). `git push` fails: this machine authenticates as
   `AdamAlhousiki`, which has no write access to `foxboxxx/PIton`. Add that account as a
   collaborator, then `git push -u origin main`.
2. **Inner-loop tuning.** Matmul loop body is 198 instructions for 128 16-lane MACs (65% of the
   ALU peak) at BM=8, BN=64, BK=4; ideal is 160 (80%). Ideas: receive B rows straight into MAC
   operands, reuse one address register per column block, autotune (BM, BN, BK) on the Pi.
3. `ftoi` of some huge values (e.g. −1.7e38 → 0x80000000) is not modelled; out-of-range
   float→int is undefined in Triton, everything else in `tools/float_specials_probe.py` matches.
4. Plain `+`/`*` keep the hardware's NaN-as-inf behaviour (documented); division, comparisons,
   exp/log/sqrt follow IEEE for NaN/inf.

## Useful commands
```bash
.venv/bin/python -m pytest -q --ignore=tests/test_hw.py     # emulator + vc4asm tests
PITON_HW=1 .venv/bin/python -m pytest tests/test_hw.py      # hardware tests (Pi attached)
.venv/bin/python tools/probes.py <probe>                    # hardware probes
PITON_HW=1 .venv/bin/python tools/float_specials_probe.py   # float special values vs emulator
PITON_HW=1 .venv/bin/python benchmarks/matmul_pi.py 128 512 --bk 8
```
