# Notes / open questions

## MORNING SUMMARY (2026-11-05 ~01:35)
perf/NEXT.md steps 1-4 are done, all on branch perf-loop (head 2557a06; local commits only).
- H6 rejected on the Pi (matmul 0.72-0.80x). C11 passed its Pi gate but is NOT on by default:
  compiles get 6.4x slower. **Your call** (see the C11 section below).
- Final profile: 1.409x geomean over the 11-03 baseline. Digits 9.380 ms (14.2x over ARM VFP),
  matmul 10.15 GFLOPS at 512^3. Docs updated.
- New: Linux power cycling (tools/pi_power_linux, works on the new head), and an arm_vfp_ops.S fix
  for softfp builds.
- perf-loop full host suite (02:36): **1276 passed, 33 skipped, 0 failed** (1:02:21).
- C11 PITON_PAIR=1 host suite (04:41): **1368 passed, 57 skipped, 0 failed** (4:55:02).
  A duplicate copy the test agent started (perf/pytest_try_c11_run2.log, pid 9726) was still
  running at 04:42. Stopping it was refused by the auto-mode classifier, so kill it if it's still
  around. Delete perf/pytest_try_c11*.log afterwards.
- Everything in perf/NEXT.md is done; the loop is stopped.
- Worktrees made tonight in .claude/worktrees: perf-loop, linux-power, c11-pi, vfp-softfp
  (try-h6 and try-c11 existed). Remove them when you're done with them.

## 2026-11-04 night: Pi not reachable (blocks every Pi gate in perf/NEXT.md)
- The CP2102N USB-UART is not attached to WSL: `usbipd list` shows it only under "Persisted",
  not "Connected", so Windows itself does not see it (new head plugged in?). No /dev/ttyUSB*.
- To fix (admin PowerShell): replug, `usbipd list`, `usbipd bind --busid <id>` if "Not shared",
  `usbipd attach --wsl --busid <id>`.
- Power cycling: tools/pi_power.c is macOS-only (IOKit). On Linux the cp210x driver may expose the
  CP2102N's GPIO0 as /dev/gpiochip*. Check that once the head is attached; otherwise a hung kernel
  waits for a manual reset.
- Runner env for this machine (cs140e instead of the CS240LX repo):
  PITON_TTY=/dev/ttyUSB0 PITON_BOOTLOADER=~/projects/cs140e/bin/pi-install.linux PITON_POWER=0
  PITON_LIBPI_ROOT=~/projects/cs140e CS140E_2026_PATH=~/projects/cs140e
  (arm-none-eabi-gcc 13.2 is now installed from apt.) tests/test_hw.py has not passed with this
  setup yet: the first try failed at the build because CS140E_2026_PATH was unset.
- Meanwhile: running the host-only C11 step (PITON_PAIR=1 full suite in the try-c11 worktree).

## 2026-11-05 ~00:00 update: Pi reachable, power line available
- Head re-attached; tests/test_hw.py: 12 passed (3m38s) with the cs140e setup.
- GPIO: udev rule /etc/udev/rules.d/99-cp210x-gpio.rules (gpiochip* -> dialout 0660) and gpiod
  1.6.3 installed by the user; `gpiodetect` shows gpiochip0 [cp210x] (4 lines). usbipd
  --auto-attach running on Windows. Power cycle via line 0 not yet tested (after H6's Pi runs).
- cs140e's libpi does not link libgcc (Makefile.robust-v2 links $(LIBS) $(LIB_POST) only, and
  defs.mk's LGCC points at a missing cs140e/lib/libgcc.a), so every program that uses float math on
  the ARM side failed to link (__aeabi_fmul etc.). test_hw passes without it; perf/ab.py cases do not.
  Fix with no code change: `export LIB_POST=/usr/lib/gcc/arm-none-eabi/13.2.1/libgcc.a`.
- First H6 accept run (perf/loop/accept-20261104-2353-mm.log in try-h6) said REJECT, but only
  because of that link failure (8 cases "build failed", none ran). Rerun with LIB_POST set.
- PITON_TTY now points at /dev/serial/by-id/usb-Silicon_Labs_CP2102N_..., so a renamed ttyUSB after
  a power cycle is still found.

## 2026-11-05 ~00:30 update
- H6 rerun with libgcc: REJECTED on the Pi (matmul_8x64x8 0.716x, 8x64x4 0.803x; TMU stalls
  0.22 -> 0.47). Recorded on perf-loop (8e5b929).
- Power cycling WORKS on the new head: GPIO0 low powers the Pi off (the bridge stays enumerated),
  power.cycle() 2.1 s, test_hw passes after it. tools/pi_power_linux + power.tool(tty) on
  infra/linux-power (3262ef4, audited 4x), merged into perf-loop (97a0578). PITON_POWER=0 dropped.
- C11: Pi steps run in worktree .claude/worktrees/c11-pi (branch try/c11-pi = try/c11 + perf-loop).
  The PITON_PAIR=1 host suite runs in .claude/worktrees/try-c11 (log perf/pytest_try_c11.log); pairing
  makes compiles ~6x slower (test_bench_matmul_on_pi 6.6 s -> 39.4 s), so it takes a long time.
  A duplicate run (perf/pytest_try_c11_run2.log) is also going; stopping it was refused by the auto
  mode classifier, so it's left alone. Delete those perf/pytest_try_c11*.log files when done.

## 2026-11-05 ~01:05: C11 Pi gate ACCEPT; QUESTION for you: land it on by default?
- C11 (PITON_PAIR=1) on the Pi: hang check fine (no hang; mandel_32x16 1.057x), test_hw 12/12,
  suite A/B ACCEPT: mandel_16x16x2 1.074x, mandel_32x16 1.057x, transpose 1.034x, softmax 1.026x,
  dense_relu 1.031x, row_sum/row_max 1.028x; nothing below 0.982x (exp). Logs in
  .claude/worktrees/c11-pi/perf/loop/accept-20261105-0025-pair.log, perf/results/ab-2026-11-05-pair*.json.
- BUT pairing makes compiles 6.4x slower (the 23 suite cases: 74.6 s -> 481 s; dense_relu 9.2x,
  softmax 8.7 -> 57 s), measured with two other pytest runs on the CPU, so only the ratio counts.
  Turning it on by default puts that on every compile and every suite run. I have NOT flipped the
  default. Options: (a) on by default anyway, (b) opt-in (`PITON_PAIR=1` / per-kernel), (c) make
  the pairing search cheaper first. Your call.

## 2026-11-05 ~01:05: ARM VFP code needs the cs240lx libpi, not cs140e
- cs140e's libpi builds soft-float (no -mfpu=vfp) and never enables the FPU, so digits' arm_vfp
  kernel did not assemble (and VFP code would fault). GPU timings tonight are unaffected (both A/B
  sides used the same ARM build).
- ~/projects/cs240lx is the course repo PIton was written against (CFLAGS -mfpu=vfp
  -mfloat-abi=softfp, FPU enabled in staff-start.S). With it, test_hw 12/12.
- CAUTION, and my mistake, now undone: the first build against ~/projects/cs240lx rebuilt its
  libpi.a, libpi/objs and lib/libm objects, which are TRACKED in git (109 files modified with this
  machine's gcc 13). The repo was clean just before, so I restored exactly those files
  (`git checkout -- lib/libm libpi`; status clean again) and now build against a private copy in the
  session scratchpad (env file pienv240.sh there). Nothing else in cs240lx was touched.
  cs340lx exists too; not used.
