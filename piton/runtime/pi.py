"""Run QPU programs on a real Raspberry Pi (bare metal, CS140E/240LX setup).

For every run we generate a small `notmain()` program that embeds the QPU
code and the input buffers, allocates GPU memory through the mailbox,
launches one QPU program per uniforms list through the V3D request queue,
and dumps the requested buffers over the UART.  The program is built with
the course `libpi` Makefile and sent with the course bootloader.

Environment:
  CS240LX_2025_PATH   course repo containing libpi (default makefile root)
  PITON_TTY           serial device (default: first /dev/cu.usbserial*)
  PITON_BOOTLOADER    bootloader executable (default: my-install, then pi-install)
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

RUNTIME_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "pi", "runtime")


class PiError(Exception):
    pass


def find_tty() -> Optional[str]:
    env = os.environ.get("PITON_TTY")
    if env:
        return env
    devs = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/ttyUSB*"))
    return devs[0] if devs else None


def find_bootloader() -> Optional[str]:
    for name in (os.environ.get("PITON_BOOTLOADER"), "my-install", "pi-install"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def libpi_root() -> Optional[str]:
    for var in ("PITON_LIBPI_ROOT", "CS240LX_2025_PATH", "CS340LX_2026_PATH", "CS140E_2025_PATH"):
        path = os.environ.get(var)
        if path and os.path.exists(os.path.join(path, "libpi", "mk", "Makefile.robust-v2")):
            return path.rstrip("/")
    return None


def pi_available() -> bool:
    return find_tty() is not None and find_bootloader() is not None and libpi_root() is not None


@dataclass
class Buffer:
    name: str
    nbytes: int
    init: Optional[np.ndarray] = None      # uint32 view of initial contents
    lcg_seed: Optional[int] = None         # generate floats on the Pi instead
    column_block: Optional[tuple[int, int]] = None  # (logical columns, packed panel width)

    def __post_init__(self):
        if self.column_block is not None:
            columns, width = self.column_block
            if not isinstance(columns, int) or not isinstance(width, int) or \
                    columns <= 0 or width <= 0 or columns % width:
                raise ValueError("column_block needs positive columns divisible by the panel width")
            if self.nbytes <= 0 or self.nbytes % (4 * columns):
                raise ValueError("column_block buffer size must contain complete float32 rows")
            if self.lcg_seed is None or self.init is not None:
                raise ValueError("column_block is supported only for LCG-generated buffers")


@dataclass
class PiResult:
    outputs: dict
    times_us: list
    log: str


def lcg_floats(n: int, seed: int) -> np.ndarray:
    """Bit-exact twin of piton_lcg_float() in piton_rt.h."""
    out = np.empty(n, dtype=np.float32)
    s = seed & 0xFFFFFFFF
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((s >> 8)) * np.float32(1.0 / 8388608.0) - np.float32(1.0)
    return out


def _c_words(name: str, words) -> str:
    words = [int(w) & 0xFFFFFFFF for w in words]
    lines = []
    for i in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{w:08x}" for w in words[i:i + 8]) + ",")
    return f"static const uint32_t {name}[{max(len(words), 1)}] = {{\n" + "\n".join(lines or ["    0"]) + "\n};\n"


class PiRunner:
    def __init__(self, tty: Optional[str] = None, timeout: float = 120.0, keep: bool = False,
                 verbose: bool = False):
        self.tty = tty or find_tty()
        self.bootloader = find_bootloader()
        self.root = libpi_root()
        self.timeout = timeout
        self.keep = keep
        self.verbose = verbose
        if not (self.tty and self.bootloader and self.root):
            raise PiError("no Raspberry Pi setup found (need a USB serial device, my-install/"
                          "pi-install, and CS240LX_2025_PATH pointing at the course repo)")

    def run(self, code_words, buffers: list, programs: list, outputs: list, repeats: int = 1,
            label: str = "piton", sample: Optional[dict] = None) -> PiResult:
        """Execute `code_words` once per entry of `programs`.

        buffers  -- list[Buffer]
        programs -- list of uniform lists; entries are ints or ("buf", name, byte_offset)
        outputs  -- buffer names to read back after the last repeat
        """
        src = self._generate(code_words, buffers, programs, outputs, repeats, label, sample or {})
        workdir = tempfile.mkdtemp(prefix="piton-pi-")
        try:
            with open(os.path.join(workdir, "main.c"), "w") as f:
                f.write(src)
            for name in ("piton_rt.c", "piton_rt.h"):
                shutil.copy(os.path.join(RUNTIME_DIR, name), workdir)
            with open(os.path.join(workdir, "Makefile"), "w") as f:
                f.write(
                    "PROGS := main.c\nCOMMON_SRC := piton_rt.c\nRUN = 0\n"
                    f"BOOTLOADER = {self.bootloader}\n"
                    f"include {self.root}/libpi/mk/Makefile.robust-v2\n")
            env = dict(os.environ)
            env.setdefault("CS240LX_2025_PATH", self.root + "/")
            proc = subprocess.run(["make", "-s"], cwd=workdir, capture_output=True, text=True, env=env)
            if proc.returncode != 0 or not os.path.exists(os.path.join(workdir, "main.bin")):
                raise PiError(f"build failed:\n{proc.stdout}\n{proc.stderr}")
            log = self._boot(os.path.join(workdir, "main.bin"))
        finally:
            if not self.keep:
                shutil.rmtree(workdir, ignore_errors=True)
            elif self.verbose:
                print("pi workdir:", workdir)
        return self._parse(log, buffers, outputs)

    # -- program generation ----------------------------------------------------

    def _generate(self, code_words, buffers, programs, outputs, repeats, label, sample) -> str:
        code32 = []
        for w in code_words:
            code32 += [w & 0xFFFFFFFF, w >> 32]
        out = ['#include "rpi.h"', '#include "piton_rt.h"', ""]
        out.append(_c_words("qpu_code", code32))
        offset = (len(code32) * 4 + 4095) & ~4095
        layout = {}
        for b in buffers:
            layout[b.name] = offset
            offset += (b.nbytes + 4095) & ~4095
            if b.init is not None:
                out.append(_c_words(f"init_{b.name}", b.init.view(np.uint32).ravel()))
        nunif = max((len(p) for p in programs), default=1)
        unif_off = offset
        offset += 4 * nunif * len(programs)
        total = offset
        out.append(f"static uint32_t unif_ptrs[{len(programs)}];")
        uniform_writes = []

        def column_expr(values, name):
            """Compact constants/progressions, using a data array for other columns."""
            values = [int(v) & 0xFFFFFFFF for v in values]
            if not values:
                return "0u"
            first = values[0]
            step = (values[1] - first) & 0xFFFFFFFF if len(values) > 1 else 0
            if all(v == (first + i * step) & 0xFFFFFFFF for i, v in enumerate(values)):
                return f"(0x{first:08x}u + i * 0x{step:08x}u)" if step else f"0x{first:08x}u"
            out.append(_c_words(name, values))
            return f"{name}[i]"

        for j in range(nunif):
            values, relocations = [], []
            for i, prog in enumerate(programs):
                u = prog[j] if j < len(prog) else 0
                if isinstance(u, tuple) and u[0] == "self":
                    values.append(unif_off + 4 * i * nunif)
                    relocations.append(1)
                elif isinstance(u, tuple):
                    _, name, off = u
                    values.append(layout[name] + off)
                    relocations.append(1)
                else:
                    values.append(u)
                    relocations.append(0)
            expr = column_expr(values, f"unif_col_{j}")
            if relocations and all(relocations):
                expr = f"m.bus + {expr}"
            elif any(relocations):
                reloc = column_expr(relocations, f"unif_reloc_{j}")
                expr = f"{expr} + m.bus * {reloc}"
            uniform_writes.append(f"        w[base + {j}] = {expr};")
        out.append("void notmain(void) {")
        out.append("    piton_init();")
        out.append(f"    piton_buf_t m = piton_alloc({total});")
        out.append("    volatile uint32_t *w = (volatile uint32_t *)m.cpu;")
        out.append(f"    for (unsigned i = 0; i < {len(code32)}; i++) w[i] = qpu_code[i];")
        for b in buffers:
            base = layout[b.name] // 4
            n = b.nbytes // 4
            if b.init is not None:
                out.append(f"    for (unsigned i = 0; i < {n}; i++) w[{base} + i] = init_{b.name}[i];")
            elif b.lcg_seed is not None:
                out.append(f"    {{ uint32_t s = {b.lcg_seed}u; volatile float *f = (volatile float *)&w[{base}];")
                if b.column_block is None:
                    out.append(f"      for (unsigned i = 0; i < {n}; i++) f[i] = piton_lcg_float(&s); }}")
                else:
                    columns, width = b.column_block
                    rows = n // columns
                    out.append("      uint32_t prep_start = timer_get_usec();")
                    out.append(f"      for (unsigned i = 0; i < {n}; i++) {{")
                    out.append(f"          unsigned row = i / {columns}u, col = i % {columns}u;")
                    out.append(f"          unsigned packed = (col / {width}u) * {rows * width}u + "
                               f"row * {width}u + col % {width}u;")
                    out.append("          f[packed] = piton_lcg_float(&s);")
                    out.append("      }")
                    out.append(f'      printk("PITON_PREP {b.name} %d\\n", timer_get_usec() - prep_start); }}')
            else:
                out.append(f"    for (unsigned i = 0; i < {n}; i++) w[{base} + i] = 0;")
        out.append(f"    for (unsigned i = 0; i < {len(programs)}; i++) {{")
        out.append(f"        unsigned base = {unif_off // 4}u + i * {nunif}u;")
        out.extend(uniform_writes)
        out.append("        unif_ptrs[i] = m.bus + 4u * base;")
        out.append("    }")
        out.append(f"    printk(\"PITON_START {label} qpus=%d\\n\", piton_qpu_count());")
        out.append(f"    for (unsigned r = 0; r < {repeats}; r++) {{")
        out.append(f"        uint32_t t = piton_launch(m.bus, unif_ptrs, {len(programs)});")
        out.append("        printk(\"PITON_TIME %d\\n\", t);")
        out.append("    }")
        for name in outputs:
            b = next(x for x in buffers if x.name == name)
            base = layout[name] // 4
            stride = sample.get(name, 1)
            n = (b.nbytes // 4 + stride - 1) // stride
            out.append(f"    printk(\"PITON_BUF {name} {n}\\n\");")
            out.append(f"    for (unsigned i = 0; i < {n}; i += 8) {{")
            out.append(f"        for (unsigned j = i; j < i + 8 && j < {n}; j++) "
                       f"printk(\"%x \", w[{base} + j * {stride}]);")
            out.append("        printk(\"\\n\");")
            out.append("    }")
        out.append("    printk(\"PITON_DONE\\n\");")
        out.append("    piton_free(&m);")
        out.append("}")
        return "\n".join(out) + "\n"

    # -- execution -------------------------------------------------------------

    def _boot(self, binary: str) -> str:
        import select
        proc = subprocess.Popen([self.bootloader, self.tty, binary], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
        chunks = []
        tail = b""
        failed = False
        start = time.monotonic()
        try:
            while True:
                remaining = self.timeout - (time.monotonic() - start)
                if remaining <= 0:
                    log = b"".join(chunks).decode("utf-8", errors="replace")
                    raise PiError(f"timeout after {self.timeout:.0f}s waiting for the Pi "
                                  "(kernel hung?):\n" + "\n".join(log.splitlines()[-40:]))
                ready, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
                if not ready:
                    if proc.poll() is not None:
                        break
                    continue
                # Buffered readline can consume several lines from the pipe at once.
                # select then sees an empty FD while PITON_DONE remains in Python's
                # text buffer. Read ready bytes directly, including partial lines.
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if self.verbose:
                    print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                markers = tail + chunk
                found = [(markers.index(marker), marker)
                         for marker in (b"PITON_DONE", b"PANIC", b"pi exited") if marker in markers]
                if found:
                    _, first_marker = min(found)
                    failed = first_marker != b"PITON_DONE"
                    if not failed:
                        time.sleep(0.2)
                    break
                tail = markers[-32:]
        finally:
            try:
                try:
                    proc.terminate()
                except OSError:
                    pass  # The child may have exited between reading and cleanup.
                try:
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=1)
                    except (OSError, subprocess.TimeoutExpired):
                        # A stuck USB driver can prevent even a killed bootloader
                        # from being reaped. Keep its wait bounded and preserve
                        # the firmware's success/failure result. Reconnecting the
                        # adapter may still be needed before another hardware run.
                        pass
            finally:
                proc.stdout.close()
        log = b"".join(chunks).decode("utf-8", errors="replace")
        if failed or "PITON_DONE" not in log:
            raise PiError("Pi run failed:\n" + "\n".join(log.splitlines()[-40:]))
        return log

    def _parse(self, log: str, buffers, outputs) -> PiResult:
        times = [int(m) for m in re.findall(r"PITON_TIME (\d+)", log)]
        result = {}
        lines = log.splitlines()
        i = 0
        while i < len(lines):
            m = re.match(r".*PITON_BUF (\w+) (\d+)", lines[i])
            if m:
                name, n = m.group(1), int(m.group(2))
                words = []
                i += 1
                while len(words) < n and i < len(lines):
                    words += [int(x, 16) for x in lines[i].split() if re.fullmatch(r"(0x)?[0-9a-fA-F]+", x)]
                    i += 1
                result[name] = np.array(words[:n], dtype=np.uint32)
                continue
            i += 1
        missing = [o for o in outputs if o not in result]
        if missing:
            raise PiError(f"missing outputs {missing} in log")
        return PiResult(result, times, log)


def run_on_pi(ck, grid, values: dict, repeats: int = 1, runner: Optional[PiRunner] = None,
              verbose: bool = False):
    """Launch compiled kernel `ck` on the Pi; arrays in `values` are copied back."""
    from .launcher import program_ids
    from ..qpu.isa import f2u
    runner = runner or PiRunner(verbose=verbose)
    buffers = []
    for name, v in values.items():
        if isinstance(v, np.ndarray):
            buffers.append(Buffer(name, max(v.nbytes, 4), init=np.ascontiguousarray(v).view(np.uint32).ravel()))
    buffers.append(Buffer("__zero", 64))
    buffers.append(Buffer("__scratch", 64))
    programs = []
    for pid, g in program_ids(grid):
        u = []
        for kind, name in ck.uniforms:
            if kind == "arg":
                v = values[name]
                if isinstance(v, np.ndarray):
                    u.append(("buf", name, 0))
                elif isinstance(v, (float, np.floating)):
                    u.append(f2u(float(v)))
                else:
                    u.append(int(v) & 0xFFFFFFFF)
            elif kind == "pid":
                u.append(pid[int(name[3:])])
            elif kind == "nprog":
                u.append(g[int(name[5:])])
            elif kind == "zero":
                u.append(("buf", "__zero", 0))
            elif kind == "scratch":
                u.append(("buf", "__scratch", 0))
            elif kind == "self":
                u.append(("self",))
        programs.append(u)
    outs = [n for n, v in values.items() if isinstance(v, np.ndarray)]
    res = runner.run(ck.words, buffers, programs, outs, repeats=repeats, label=ck.name)
    for name in outs:
        v = values[name]
        v[...] = res.outputs[name][:v.size].view(v.dtype).reshape(v.shape)
    return {"times_us": res.times_us}


def bench_matmul_on_pi(ck, M: int, N: int, K: int, grid, args_after_ptrs: list, repeats: int = 3,
                       sample_stride: int = 97, seeds=(12345, 67890), verbose=False) -> dict:
    """Time a compiled matmul on the Pi with inputs generated on the board.

    args_after_ptrs: runtime scalar arguments in uniform order (M, N, strides, ...)
    by name -> value. Returns timing and error on sampled final outputs. The
    normalized error is max absolute error / max absolute sampled reference;
    it is not the largest elementwise relative error (which blows up near 0).
    """
    from .launcher import program_ids
    if min(M, N, K, repeats, sample_stride) <= 0:
        raise ValueError("matmul dimensions, repeats and sample_stride must be positive")
    runner = PiRunner(verbose=verbose)
    bufs = [Buffer("a", 4 * M * K, lcg_seed=seeds[0]), Buffer("b", 4 * K * N, lcg_seed=seeds[1]),
            Buffer("c", 4 * M * N), Buffer("__zero", 64), Buffer("__scratch", 64)]
    scal = dict(args_after_ptrs)
    programs = []
    for pid, g in program_ids(grid):
        u = []
        for kind, name in ck.uniforms:
            if kind == "arg":
                if name in ("a_ptr", "b_ptr", "c_ptr"):
                    u.append(("buf", name[0], 0))
                else:
                    u.append(int(scal[name]) & 0xFFFFFFFF)
            elif kind == "pid":
                u.append(pid[int(name[3:])])
            elif kind == "nprog":
                u.append(g[int(name[5:])])
            elif kind == "zero":
                u.append(("buf", "__zero", 0))
            elif kind == "scratch":
                u.append(("buf", "__scratch", 0))
            elif kind == "self":
                u.append(("self",))
        programs.append(u)
    res = runner.run(ck.words, bufs, programs, ["c"], repeats=repeats, label=ck.name,
                     sample={"c": sample_stride})
    if len(res.times_us) != repeats or any(t <= 0 or not np.isfinite(t) for t in res.times_us):
        raise PiError(f"expected {repeats} positive finite timings, got {res.times_us}")
    idx = np.arange(0, M * N, sample_stride)
    if len(res.outputs["c"]) != len(idx):
        raise PiError(f"expected {len(idx)} sampled C values, got {len(res.outputs['c'])}")
    got = res.outputs["c"].view(np.float32)
    if not np.isfinite(got).all():
        raise PiError("matmul returned nonfinite sampled output")
    a = lcg_floats(M * K, seeds[0]).reshape(M, K).astype(np.float64)
    b = lcg_floats(K * N, seeds[1]).reshape(K, N).astype(np.float64)
    ref = (a @ b).ravel()
    err = float(np.abs(got - ref[idx]).max())
    ref_scale = float(np.abs(ref[idx]).max())
    normalized_error = err / ref_scale if ref_scale else (0.0 if err == 0 else float("inf"))
    t = min(res.times_us)
    return {"times_us": res.times_us, "best_us": t, "gflops": 2.0 * M * N * K / (t * 1e3),
            "max_err": err, "ref_scale": ref_scale, "normalized_error": normalized_error,
            "sample_count": len(idx), "programs": len(programs)}
