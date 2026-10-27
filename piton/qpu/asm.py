"""A resolved QPU program: instructions + labels -> machine words / qasm text."""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from .isa import Instr, canonicalize, encode, format_instr, u2f


@dataclass
class Program:
    name: str
    instrs: list = field(default_factory=list)       # list[Instr]
    labels: dict = field(default_factory=dict)       # label -> instruction index
    header: list = field(default_factory=list)       # comment lines for the qasm file

    def label_addrs(self) -> dict:
        return {k: 8 * v for k, v in self.labels.items()}

    def canonical(self) -> list:
        return [canonicalize(i) for i in self.instrs]

    def words(self) -> list:
        addrs = self.label_addrs()
        return [encode(ins, 8 * pc, addrs) for pc, ins in enumerate(self.canonical())]

    def binary(self) -> bytes:
        return b"".join(struct.pack("<Q", w) for w in self.words())

    def qasm(self) -> str:
        by_index: dict = {}
        for name, idx in self.labels.items():
            by_index.setdefault(idx, []).append(name)
        out = [f"# {line}" if line else "#" for line in self.header]
        for pc, ins in enumerate(self.canonical()):
            for name in by_index.get(pc, []):
                out.append(f":{name}")
            text = format_instr(ins)
            comment = ins.comment
            if ins.kind == "ldi" and not comment:
                f = u2f(ins.imm)
                if ins.imm > 0xFFFF and abs(f) < 1e7 and abs(f) > 1e-7:
                    comment = f"{f!r} as float"
            out.append(f"    {text:<52} # {comment}" if comment else f"    {text}")
        for name in by_index.get(len(self.instrs), []):
            out.append(f":{name}")
        return "\n".join(out) + "\n"

    def c_array(self, symbol: Optional[str] = None) -> str:
        symbol = symbol or self.name
        words = self.words()
        body = []
        for w in words:
            body.append(f"    0x{w & 0xFFFFFFFF:08x}, 0x{w >> 32:08x},")
        return (f"__attribute__((aligned(8)))\nconst uint32_t {symbol}_code[{2 * len(words)}] = {{\n"
                + "\n".join(body) + "\n};\n")


# ---------------------------------------------------------------------------
# Optional cross-check with the real assembler
# ---------------------------------------------------------------------------


def find_vc4asm() -> Optional[str]:
    return os.environ.get("VC4ASM") or shutil.which("vc4asm")


def vc4asm_assemble(qasm: str, verify: bool = True) -> tuple:
    """Assemble `qasm` with vc4asm. Returns (words, warnings) or raises.

    Raises FileNotFoundError when vc4asm is not installed.
    """
    exe = find_vc4asm()
    if exe is None:
        raise FileNotFoundError("vc4asm not found (brew install vc4asm)")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.qasm")
        out = os.path.join(d, "k.bin")
        with open(src, "w") as f:
            f.write(qasm)
        args = [exe] + (["-V"] if verify else []) + ["-o", out, src]
        proc = subprocess.run(args, capture_output=True, text=True)
        msgs = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            raise RuntimeError(f"vc4asm failed:\n{msgs}")
        data = open(out, "rb").read()
    words = [struct.unpack_from("<Q", data, i)[0] for i in range(0, len(data), 8)]
    warnings = [m for m in msgs.splitlines() if m.strip()]
    return words, warnings
