# Checkpoint — where PIton stands (2026-10-26)

## Done and verified
- Full pipeline: Triton AST frontend → tile IR + passes → QPU lowering → register allocation →
  scheduling → `.qasm` / machine words / C launcher (`python -m piton.tools.compile`).
- Emulator (`piton/qpu/emulator.py`) with strict hazard checks; matches the Pi bit for bit on
  everything tested (models round-toward-zero floats, 4-deep TMU FIFOs, unif_addr A-space only).
- `pytest`: 17 pass (vector add, softmax ×3, where/relu, row-sum, tutorial matmul ×4 incl.
  runtime K and ragged sizes, vc4asm bit-exact cross-check for vector add + matmul).
- On hardware (Pi Zero W): vector add and tutorial matmul bit-identical to the emulator for
  tile-aligned sizes (64³, 128×64×16) and ragged sizes with ≤2 programs.

## Open items (in order)
1. **Power-cycle the Pi.** A hung probe wedged it (the runner now has a host-side timeout and the
   Pi runtime a 20 s launch watchdog that reboots, so this should not recur).
2. **Push access.** `git push` fails: both HTTPS and SSH authenticate as `AdamAlhousiki`, which has
   no write access to `foxboxxx/PIton`. Add that account as a collaborator, then `git push -u origin main`.
3. **Re-verify concurrency fix on hardware.** Ragged matmuls with 4+ concurrent programs were wrong
   on the Pi (e.g. 16×100×16, 72×100×64, 40×70×36). Hypothesis: VPM-write/VDW DMA path is shared by
   all QPUs. Stores now run under the hardware mutex (emulator enforces it). Re-run:
   `PITON_HW=1 .venv/bin/python -m pytest tests/test_hw.py` plus the ragged cases; also probe
   qpu_num values and unsynchronised-vs-mutex VDW to confirm/refute (fixed probe code is in the
   session log: add a `nop` after `ldi ra2` before the spin loop).
4. **Benchmarks.** `piton.runtime.pi.bench_matmul_on_pi` (LCG inputs on the board, sampled check)
   and `pi/examples/matmul` (`make MM=256`). Baseline: CS240LX lab 1-QPU matmul, 128³ in 116 ms
   (≈36 MFLOPS); ARM ≈ 8.8 MFLOPS. Pre-mutex PIton run: 64³ in 127 µs on 12 QPUs.
5. **Inner-loop tuning.** Matmul loop body is 198 instructions for 128 16-lane MACs (65% of the
   ALU peak) at BM=8, BN=64, BK=4; ideal is 160 (80%). Remaining overhead: TMU address adds and
   receive moves that do not fit the free add slots of the broadcast instructions. Ideas: receive
   B rows straight into MAC operands, reuse one address register per column block, autotune
   (BM, BN, BK) on the Pi.
6. Measure whether holding the mutex across the whole tile store costs much at large K; if so,
   only serialise the DMA part.

## Useful commands
```bash
.venv/bin/python -m pytest -q                       # emulator + vc4asm tests
.venv/bin/python tools/probes.py <probe>            # hardware probes (Pi attached)
PITON_HW=1 .venv/bin/python -m pytest tests/test_hw.py
```
