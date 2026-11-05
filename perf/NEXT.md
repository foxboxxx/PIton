# Next steps for the optimisation loop (2026-11-05: steps 1-4 done)

## State
`perf-loop` (this branch) has every change that passed its Pi gate, all on by default: C1, A1 (+ the
sparse-store rule and once-per-boot slot setup), C10, E1, H2, D1, C7 (`PITON_ISSUE`), E4
(`PITON_COARSEN_RUNTIME`), E5 (`PITON_COARSEN_STORES`), B5 (`PITON_TMU_NOSWAP`). Rules and results:
`perf/BACKLOG.md` (status rows and the log).

Two changes are merged onto this branch's head (d164de2) but not yet gated on the Pi:

| branch | switch | state |
|---|---|---|
| `try/h6` (873e3f3) | `PITON_MM` (H6: matmul k loop, A in column blocks, receives in use order) | host suites pass both ways (1380), switch off bit-identical to d164de2 (51 builds); emulator: matmul_8x64x8 1.096x fewer executed instructions, matmul_8x64x4 1.034x, dense_relu 1.058x, matmul_pi 512/1024 1.08x, digits layers 1.014-1.028x. Ready for the Pi gate |
| `try/c11` (c784e58) | `PITON_PAIR` (C11: scheduler pairing) | merged and reviewed before the merge (opt/c11-pair: 1126 tests both ways, switch off identical); the full suite with `PITON_PAIR=1` on the merged tree was cut short. Run it first, then the Pi gate |

The loop runs in "fast mode": one full host suite per landing, no separate merge review, noise
rechecks only for cases whose code changed.

## Progress 2026-11-05 (WSL2 machine, cs140e setup)
- Pi runner env on this machine: `PITON_TTY=/dev/serial/by-id/usb-Silicon_Labs_CP2102N_...`
  (survives renames), `PITON_BOOTLOADER=~/projects/cs140e/bin/pi-install.linux`,
  `PITON_LIBPI_ROOT=CS140E_2026_PATH=~/projects/cs140e`, and
  `LIB_POST=/usr/lib/gcc/arm-none-eabi/13.2.1/libgcc.a`: cs140e's libpi does not link libgcc, so any
  ARM-side float math failed to link. test_hw 12/12.
- Step 1 (H6) done: REJECTED on the Pi (matmul_8x64x8 0.716x, 8x64x4 0.803x; TMU stalls
  0.22 -> 0.47), BACKLOG H6 + log row. perf-loop is unchanged, so C11 needs no rebase onto it.
- Power: tools/pi_power_linux (libgpiod on the cp210x gpiochip), branch infra/linux-power. On this
  head GPIO0 low powers the Pi off but the bridge stays up (no re-enumeration); power.cycle()
  takes 2.1 s and test_hw passes after it, so hung kernels recover again.
- Step 2 (C11) Pi gate ACCEPT on try/c11-pi (678b041: try/c11 + this branch), NOT landed: compiles
  6.4x slower with pairing on; on-by-default is the user's call. Its PITON_PAIR=1 host suite
  (try/c11 c784e58): 1368 passed, 57 skipped, 0 failed in 4:55:02.
- ARM VFP code (digits' arm_vfp, primitives) needs the cs240lx libpi (-mfpu=vfp, FPU enabled at
  boot); cs140e's builds soft-float. Build against a copy of ~/projects/cs240lx: its makefile
  rebuilds TRACKED libpi.a / objs / libm objects in place.
- Step 3 done: `pi-2026-11-05-final.json`, geomean 1.409x over the 11-03 baseline (BACKLOG log).
- Step 4 done: digits 9.380 ms (14.218x over ARM VFP), matmul 8.21 / 10.15 / 9.30 GFLOPS
  (128 / 512 / 1024), primitives rerun (58 rows checked, after fixing arm_vfp_ops.S for softfp:
  dd18d2b); README.md, STATUS.md, perf/README.md updated. The README primitives table was regenerated
  from the RESULT lines (it reproduces the 11-03 table from the 11-03 log).
- Host suite on perf-loop 2557a06: 1276 passed, 33 skipped (1:02:21), 0 failed.
- Open: C11 on-by-default (compile 6.4x) is the user's call; follow-ups below are unchanged.

## Progress 2026-11-05 (macOS, the 11-03 board)
- perf-loop 386a609 (the WSL2 machine's work) runs here unchanged: test_hw 12/12 (the tty is now
  /dev/cu.usbserial-110; find_tty() finds it).
- C11 gate repeated on this board (try/c11 7f22f86 = c784e58 + 386a609): no hang from the mul-ALU
  mutex release; mandel 1.074x / 1.057x, row_sum 1.044x, softmax 1.026x, idiv 1.019x, suite geomean
  1.010x, digits batch 1 1.04x. Landed in perf-loop, on by default (`PITON_PAIR=0` turns it off).
- Final profile on this board: 1.406x geomean over the 11-03 baseline (1.424x with `PITON_PAIR=1`).
- `PITON_PAIR` is on by default (the best configuration); `PITON_PAIR=0` compiles 3-6x faster.
- Open: the follow-ups below.

## Commands, in order (from the repo root, `.venv/bin/python` = `PY`)

1. H6 Pi gate (in a worktree of `try/h6`: `git worktree add .claude/worktrees/try-h6 try/h6`, cd there):
   - `$PY perf/accept.py PITON_MM --skip-host --target matmul_8x64x8 matmul_8x64x4 dense_relu`
     (test_hw with the switch on + the suite A/B; result in `perf/results/ab-<date>-mm.json`)
   - matmul and digits, alternating twice: `PITON_MM=0|1 PITON_HW=1 $PY benchmarks/matmul_pi.py 512 1024 --bk 8`
     and `PITON_MM=0|1 $PY examples/digits_pi.py --output <file>.json`
   - accept if a target is > 2% faster and nothing is > 2% slower (re-measure a flagged case with
     `$PY perf/ab.py PITON_MM <case> --rounds 6 --max-rounds 6` if its code did not change)
   - to land: default `"1"` in `_mm_options` (piton/compiler.py), tests that meant "off" set `PITON_MM=0`,
     one full suite `$PY -m pytest -q -p no:cacheprovider tests perf`, then on `perf-loop`:
     `git merge --ff-only try/h6`
2. C11 (in a worktree of `try/c11`, after merging the new perf-loop if H6 landed):
   - `PITON_PAIR=1 $PY -m pytest -q -p no:cacheprovider tests perf`
   - hang check first: its builds release the VPM mutex from the mul ALU (waddr 51), which no other build
     does: `PITON_PAIR=1 $PY perf/ab.py PITON_PAIR compute_peak mandel_32x16` (a hang shows as a
     timeout; the runner power-cycles the Pi)
   - `$PY perf/accept.py PITON_PAIR --skip-host --target recip rsqrt sqrt softmax mandel_32x16 mandel_16x16x2 transpose`
     and digits off/on twice; land as above (default `"1"` in `sched.pair_enabled`)
3. Final profile: `$PY perf/profile_pi.py --label final`, then
   `$PY perf/compare.py pi-2026-11-03-baseline.json pi-<date>-final.json`
4. Docs: rerun `examples/digits_pi.py --repeats 5`, `benchmarks/matmul_pi.py 128 512 1024 --bk 8` and the
   primitives firmware (`pi/examples/primitives`, README "Supported Triton" table) and update
   README.md, STATUS.md and perf/README.md with those numbers (README lines ~41-114 still show the
   10-30 numbers: digits 11.161 ms, matmul 9.005 GFLOPS, the primitives table).

## Follow-ups noted in the backlog
- E4: a constant-k program for full-grid launches would recover dense_relu's ~1%.
- B5: it is not known whether TMU_NOSWAP outlasts a program on the Pi (every loading program writes it,
  so it does not matter for PIton's own code).
- isa.format_instr prints condition 7 as `.ifnc`, which vc4asm reads as condition 5 (no build emits
  carry conditions today).
- Compiler output for transpose still depends on object addresses (lower.py permute, regalloc sets).
