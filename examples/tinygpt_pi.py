"""TinyGPT resident on the Pi: one bare-metal program, every layer a PIton kernel.

    .venv/bin/python examples/tinygpt_pi.py --size small            # build, boot, check, benchmark
    .venv/bin/python examples/tinygpt_pi.py --size small --chat     # ... then type prompts
    .venv/bin/python examples/tinygpt_pi.py --size small --build-only
    .venv/bin/python examples/tinygpt_pi.py --size tiny --emulate   # same launches, QPU emulator

The host compiles the 15 kernels the model needs (prefill and single-token
decode variants of embed, LayerNorm, the linear layers and causal attention),
packs the weights as fp16 and writes them, a float64 reference, and the
generated launchers into a bare-metal program (pi/examples/tinygpt/main.c).
After the upload the Pi needs nothing from the host: it prefills the prompt
with full-width launches, then decodes one character at a time against a
per-layer KV cache, and the ARM only samples the next character. An ARM C
implementation of the same algorithm runs on the same board as the baseline
and as a second on-device check.

The upload goes through the 115200-baud bootloader (about 11 KB/s), so the
medium model takes ~2.5 minutes to boot; once it is running, prompts cost
nothing extra.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from attention import attention_kernel
from tinygpt import embed_kernel, layernorm_kernel, linear_kernel, linear_tiles
from tinygpt_model import (
    EPS,
    SIZES,
    Config,
    forward,
    layer_names,
    load_model,
    model_path,
)

from piton import compiler
from piton.runtime.launcher import run_emulator

SRC_DIR = ROOT / "pi" / "examples" / "tinygpt"
# The SD-card bootloader lives at 0x200000 and the program loads at 0x8000.
MAX_BINARY = 0x200000 - 0x8000
DECODE_TILE = (4, 16, 8)        # M=1 decode: narrow columns spread a matvec over the QPUs
DEFAULT_PROMPTS = {"tiny": "The QPU ", "small": "def ", "medium": "def "}
F32 = "*fp32"
LINEAR_SIG = {**dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "res_ptr", "y_ptr"), F32), "M": "i32"}
EMBED_SIG = {"ids_ptr": "*i32", "tok_ptr": F32, "pos_ptr": F32, "x_ptr": F32, "T": "i32"}
LN_SIG = {"x_ptr": F32, "g_ptr": F32, "b_ptr": F32, "y_ptr": F32, "M": "i32", "eps": "fp32"}
ATTN_SIG = {**dict.fromkeys(("q_ptr", "k_ptr", "v_ptr", "o_ptr"), F32), "seq": "i32", "sm_scale": "fp32"}
# role -> (N, K, GELU, RESIDUAL) as functions of the config
LINEAR_ROLES = {
    "qkv": lambda c: (c.d_model, c.d_model, False, False),
    "o": lambda c: (c.d_model, c.d_model, False, True),
    "fc1": lambda c: (c.d_mlp, c.d_model, True, False),
    "fc2": lambda c: (c.d_model, c.d_mlp, False, True),
    "head": lambda c: (c.vocab, c.d_model, False, False),
}


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def tiles(cfg: Config) -> dict:
    """Prefill block sizes that fit the register file (d=128 needs BLOCK_M<=2)."""
    return {"embed": 4 if cfg.d_model <= 64 else 1, "ln": 4 if cfg.d_model <= 64 else 2,
                "attn": 4 if cfg.head_dim <= 16 else 2}


def kernel_specs(cfg: Config) -> dict:
    """name -> (kernel, signature, constants, compile grid, tile macros)."""
    t, d, ctx = tiles(cfg), cfg.d_model, cfg.ctx
    specs = {
        "embed_p": (embed_kernel, EMBED_SIG, {"D": d, "BLOCK_M": t["embed"]}, (cdiv(ctx, t["embed"]),),
                    {"BM": t["embed"]}),
        "embed_d": (embed_kernel, EMBED_SIG, {"D": d, "BLOCK_M": 1}, (1,), {}),
        "ln_p": (layernorm_kernel, LN_SIG, {"D": d, "BLOCK_M": t["ln"]}, (cdiv(ctx, t["ln"]),), {"BM": t["ln"]}),
        "ln_d": (layernorm_kernel, LN_SIG, {"D": d, "BLOCK_M": 1}, (1,), {}),
        "attn_p": (attention_kernel, ATTN_SIG,
                   {"HEAD_DIM": cfg.head_dim, "BLOCK_M": t["attn"], "BLOCK_N": 16, "CAUSAL": True, "STRIDE": d},
                   (cdiv(ctx, t["attn"]), cfg.n_head), {"BM": t["attn"]}),
        "attn_d": (attention_kernel, ATTN_SIG,
                   {"HEAD_DIM": cfg.head_dim, "BLOCK_M": 1, "BLOCK_N": 16, "CAUSAL": False, "STRIDE": d},
                   (1, cfg.n_head), {}),
    }
    for role, shape in LINEAR_ROLES.items():
        n, k, gelu, res = shape(cfg)
        variants = [("d", DECODE_TILE, 1)] + ([] if role == "head" else [("p", linear_tiles(n), ctx)])
        for suffix, (bm, bn, bk), rows in variants:
            if k % bk:
                raise ValueError(f"linear {role}: K={k} is not a multiple of BLOCK_K={bk}")
            constants = {"N": n, "K": k, "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
                             "GELU": gelu, "RESIDUAL": res}
            specs[f"lin_{role}_{suffix}"] = (linear_kernel, LINEAR_SIG, constants,
                                             (cdiv(rows, bm) * cdiv(n, bn),), {"BM": bm, "BN": bn})
    return specs


def compile_kernels(cfg: Config, verbose: bool = True) -> dict:
    out = {}
    for name, (kernel, sig, constants, grid, _) in kernel_specs(cfg).items():
        t0 = time.time()
        out[name] = compiler.compile(kernel, sig, constants, grid=grid)
        if verbose:
            print(f"  {name:12s} {len(out[name].words):5d} QPU instructions ({time.time() - t0:.1f}s)",
                  flush=True)
    return out


# ----------------------------------------------------------------------------
# Weights as deployed (fp16 storage, fp32 compute) and the reference

def tensor_order(cfg: Config) -> list[str]:
    names = ["tok_emb", "pos_emb"]
    for layer in range(cfg.n_layer):
        names += layer_names(layer)
    return names + ["lnf_g", "lnf_b", "w_head", "b_head"]


def deploy(params: dict) -> dict:
    """Round every tensor to fp16, as the Pi stores it; back to float32."""
    out = {}
    for k, v in params.items():
        if np.abs(v).max() >= 65504:
            raise ValueError(f"{k} does not fit fp16")
        out[k] = np.asarray(v, np.float16).astype(np.float32)
    return out


def pack(cfg: Config, params: dict) -> tuple[np.ndarray, dict]:
    """All tensors in one fp16 array, each starting on a 64-byte boundary."""
    offsets, chunks, pos = {}, [], 0
    for name in tensor_order(cfg):
        v = np.asarray(params[name], np.float16).ravel()
        offsets[name] = pos
        pad = (-v.size) % 16
        chunks += [v, np.zeros(pad, np.float16)]
        pos += v.size + pad
    return np.concatenate(chunks), offsets


def reference(cfg: Config, params: dict, prompt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy float64 continuation to a full context, and its logits from the
    last prompt position on (row j predicts token P + j)."""
    ids = [int(i) for i in prompt_ids]
    while len(ids) < cfg.ctx:
        ids.append(int(np.argmax(forward(cfg, params, np.array(ids, np.int32))[-1])))
    logits = forward(cfg, params, np.array(ids, np.int32))
    return np.array(ids, np.int32), logits[len(prompt_ids) - 1:]


# ----------------------------------------------------------------------------
# The resident launch sequence on the emulator (mirrors main.c launch by launch)

class EmulatedResident:
    def __init__(self, cfg: Config, params: dict, kernels: dict):
        self.cfg, self.p, self.k = cfg, params, kernels
        self.specs = kernel_specs(cfg)
        c = cfg
        self.x = np.zeros((c.ctx, c.d_model), np.float32)
        self.a, self.q, self.o = (np.zeros_like(self.x) for _ in range(3))
        self.u = np.zeros((c.ctx, c.d_mlp), np.float32)
        self.kc = np.zeros((c.n_layer, c.ctx, c.d_model), np.float32)
        self.vc = np.zeros_like(self.kc)
        self.logits = np.zeros((1, c.vocab), np.float32)
        self.ids = np.zeros(c.ctx, np.int32)
        self.sm_scale = np.float32(c.head_dim ** -0.5)

    def _run(self, name, grid, **values):
        run_emulator(self.k[name], grid, values)

    def _linear(self, role, phase, x, layer_w, layer_b, y, rows, res=None):
        name = f"lin_{role}_{phase}"
        bm, bn = self.specs[name][4]["BM"], self.specs[name][4]["BN"]
        n = LINEAR_ROLES[role](self.cfg)[0]
        values = {"x_ptr": x, "w_ptr": self.p[layer_w], "bias_ptr": self.p[layer_b],
                      "res_ptr": res if res is not None else y, "y_ptr": y, "M": rows}
        self._run(name, (cdiv(rows, bm) * cdiv(n, bn),), **values)

    def _ln(self, phase, x, g, b, y, rows):
        bm = self.specs["ln_p"][4]["BM"] if phase == "p" else 1
        self._run(f"ln_{phase}", (cdiv(rows, bm),), x_ptr=x, g_ptr=self.p[g + "_g"], b_ptr=self.p[g + "_b"],
                  y_ptr=y, M=rows, eps=np.float32(EPS))

    def _block(self, layer, phase, sl, t, rows):
        p = f"h{layer}."
        x, a = self.x[sl], self.a[sl]
        self._ln(phase, x, p + "ln1", None, a, rows)
        kc, vc = self.kc[layer], self.vc[layer]
        self._linear("qkv", phase, a, p + "wq", p + "bq", self.q[sl], rows)
        self._linear("qkv", phase, a, p + "wk", p + "bk", kc[t:t + rows], rows)
        self._linear("qkv", phase, a, p + "wv", p + "bv", vc[t:t + rows], rows)
        if phase == "p":
            bm = self.specs["attn_p"][4]["BM"]
            grid = (cdiv(rows, bm), self.cfg.n_head)
        else:
            grid = (1, self.cfg.n_head)
        self._run(f"attn_{phase}", grid, q_ptr=self.q[sl], k_ptr=kc, v_ptr=vc, o_ptr=self.o[sl],
                  seq=t + rows, sm_scale=self.sm_scale)
        self._linear("o", phase, self.o[sl], p + "wo", p + "bo", x, rows, res=x)
        self._ln(phase, x, p + "ln2", None, a, rows)
        self._linear("fc1", phase, a, p + "w1", p + "b1", self.u[sl], rows)
        self._linear("fc2", phase, self.u[sl], p + "w2", p + "b2", x, rows, res=x)

    def _head(self, row):
        self._ln("d", self.x[row:row + 1], "lnf", None, self.a[:1], 1)
        self._linear("head", "d", self.a[:1], "w_head", "b_head", self.logits, 1)
        return self.logits[0].copy()

    def prefill(self, ids) -> np.ndarray:
        t = len(ids)
        self.ids[:t] = ids
        bm = self.specs["embed_p"][4]["BM"]
        self._run("embed_p", (cdiv(t, bm),), ids_ptr=self.ids, tok_ptr=self.p["tok_emb"],
                  pos_ptr=self.p["pos_emb"], x_ptr=self.x, T=t)
        for layer in range(self.cfg.n_layer):
            self._block(layer, "p", slice(0, t), 0, t)
        return self._head(t - 1)

    def decode(self, token: int, t: int) -> np.ndarray:
        """Token at position t (0-based) with positions < t in the cache."""
        self.ids[t] = token
        self._run("embed_d", (1,), ids_ptr=self.ids[t:t + 1], tok_ptr=self.p["tok_emb"],
                  pos_ptr=self.p["pos_emb"][t:t + 1], x_ptr=self.x[:1], T=1)
        for layer in range(self.cfg.n_layer):
            self._block(layer, "d", slice(0, 1), t, 1)
        return self._head(0)


def emulate_check(cfg, params, kernels, prompt_ids, ref_ids, ref_logits, steps: int) -> float:
    """Teacher-forced prefill + `steps` decodes on the emulator; max |error|."""
    m = EmulatedResident(cfg, params, kernels)
    p = len(prompt_ids)
    rows = [m.prefill(prompt_ids)]
    for j in range(steps):
        rows.append(m.decode(int(ref_ids[p + j]), p + j))
    return float(np.abs(np.array(rows) - ref_logits[:steps + 1]).max())


# ----------------------------------------------------------------------------
# C generation and build

def c_floats(name: str, values) -> str:
    v = np.asarray(values, np.float32).ravel()
    body = ",\n".join(", ".join(float(x).hex() + "f" for x in v[i:i + 8]) for i in range(0, len(v), 8))
    return f"static const float {name}[{max(len(v), 1)}] = {{\n{body}\n}};\n"


def c_ints(name: str, values, ctype: str = "int32_t", per_line: int = 16, fmt=str) -> str:
    v = [int(x) for x in np.asarray(values).ravel()]
    body = ",\n".join(", ".join(fmt(x) for x in v[i:i + per_line]) for i in range(0, len(v), per_line))
    return f"static const {ctype} {name}[{max(len(v), 1)}] = {{\n{body}\n}};\n"


def c_string(s: str) -> str:
    return '"' + "".join("\\n" if c == "\n" else "\\\\" if c == "\\" else '\\"' if c == '"' else c
                         for c in s) + '"'


def write_model_header(path: Path, cfg, tok, packed, offsets, prompt_ids, ref_ids, ref_logits, args):
    lines = ["#pragma once", "#include <stdint.h>", "",
             f"#define TG_VOCAB {cfg.vocab}", f"#define TG_CTX {cfg.ctx}", f"#define TG_D {cfg.d_model}",
             f"#define TG_H {cfg.n_head}", f"#define TG_HD {cfg.head_dim}", f"#define TG_L {cfg.n_layer}",
             f"#define TG_F {cfg.d_mlp}", f"#define TG_NW {packed.size}", f"#define TG_EPS {float(EPS).hex()}f",
             f"#define TG_SM_SCALE {float(np.float32(cfg.head_dim ** -0.5)).hex()}f",
             f"#define TG_PROMPT_LEN {len(prompt_ids)}", f"#define TG_NREF {len(ref_logits)}",
             f"#define TG_NGEN {args.gen}", f"#define TG_CHAT {int(args.chat)}",
             f"#define TG_CHAT_LEN {args.chat_len}", f"#define TG_SEED {args.seed}u",
             f'#define TG_SIZE_NAME "{args.size}"', ""]
    for name in ("tok_emb", "pos_emb", "lnf_g", "lnf_b", "w_head", "b_head"):
        lines.append(f"#define TG_OFF_{name.upper()} {offsets[name]}u")
    for short in ("ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",
                  "ln2_g", "ln2_b", "w1", "b1", "w2", "b2"):
        lines.append(c_ints(f"tg_off_{short}", [offsets[f"h{i}.{short}"] for i in range(cfg.n_layer)],
                            "uint32_t"))
    lines.append(f"static const char tg_chars[] = {c_string(tok.chars)};")
    lines.append(c_ints("tg_ref_ids", ref_ids))
    lines.append(c_floats("tg_ref_logits", ref_logits))
    lines.append(c_ints("tg_weights", packed.view(np.uint16), "uint16_t", 16, lambda x: f"0x{x:04x}"))
    path.write_text("\n".join(lines) + "\n")


def build(args) -> tuple[Path, dict]:
    cfg, tok, params, meta = load_model(model_path(args.size))
    params = deploy(params)
    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPTS[args.size]
    prompt_ids = tok.encode(prompt)
    if not 0 < len(prompt_ids) < cfg.ctx:
        raise ValueError(f"benchmark prompt must be 1..{cfg.ctx - 1} characters of the vocabulary")
    n_params = sum(v.size for v in params.values())
    print(f"TinyGPT-{args.size}: {n_params:,} params, d={cfg.d_model}, {cfg.n_layer} layers x "
          f"{cfg.n_head} heads, ctx {cfg.ctx}, vocab {cfg.vocab}, val loss {meta.get('val_loss')}", flush=True)
    ref_ids, ref_logits = reference(cfg, params, prompt_ids)
    print("compiling kernels:", flush=True)
    kernels = compile_kernels(cfg)
    if args.emulate_steps:
        err = emulate_check(cfg, params, kernels, prompt_ids, ref_ids, ref_logits, args.emulate_steps)
        print(f"emulator: prefill + {args.emulate_steps} decode steps, max |logit error| {err:.2e}", flush=True)
    out = args.build_dir.resolve() / args.size
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from piton.tools.compile import emit_c
    specs = kernel_specs(cfg)
    cfg_lines = ["#pragma once"]
    for name, ck in kernels.items():
        header, source = emit_c(ck, f"tg_{name}")
        (out / f"tg_{name}.h").write_text(header)
        (out / f"tg_{name}.c").write_text(source)
        for macro, value in specs[name][4].items():
            cfg_lines.append(f"#define TG_{name.upper()}_{macro} {value}")
    (out / "tg_kernels.h").write_text("\n".join(cfg_lines + [f'#include "tg_{n}.h"' for n in kernels]) + "\n")
    packed, offsets = pack(cfg, params)
    write_model_header(out / "tg_model.h", cfg, tok, packed, offsets, prompt_ids, ref_ids, ref_logits, args)
    for src in (SRC_DIR / "main.c", SRC_DIR / "Makefile", ROOT / "pi/runtime/piton_rt.c",
                ROOT / "pi/runtime/piton_rt.h"):
        shutil.copyfile(src, out / src.name)
    (out / "kernels.mk").write_text("TG_KERNEL_SRC := " + " ".join(f"tg_{n}.c" for n in kernels) + "\n")
    from piton.runtime.pi import libpi_root
    root = libpi_root()
    if root is None:
        raise RuntimeError("set PITON_LIBPI_ROOT (or CS240LX_2025_PATH) to the course repository with libpi")
    env = {**os.environ, "CS240LX_2025_PATH": root + "/"}
    t0 = time.time()
    made = subprocess.run(["make", "-s", "RUN=0"], check=False, cwd=out, env=env,
                          capture_output=True, text=True)
    (out / "build.log").write_text(made.stdout + made.stderr)
    if made.returncode:
        raise RuntimeError(f"firmware build failed (log in {out / 'build.log'}):\n{made.stdout[-3000:]}"
                           f"\n{made.stderr[-3000:]}")
    binary = out / "main.bin"
    size = binary.stat().st_size
    if size > MAX_BINARY:
        raise RuntimeError(f"{binary} is {size:,} bytes; the bootloader takes at most {MAX_BINARY:,}")
    info = {"size": args.size, "params": n_params, "config": cfg.__dict__, "val_loss": meta.get("val_loss"),
                "binary_bytes": size, "weight_bytes": int(packed.nbytes), "prompt": prompt,
                "kernels": {n: len(ck.words) for n, ck in kernels.items()},
                "upload_seconds_estimate": round(size / 11520, 1)}
    (out / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"built {binary} ({size:,} bytes, ~{size / 11520:.0f} s to upload at 115200 baud; "
          f"build {time.time() - t0:.0f}s)", flush=True)
    return binary, info


# ----------------------------------------------------------------------------
# Running

def parse_log(log: str) -> dict:
    """The checks and timings main.c prints; raises if a check failed or is missing."""
    checks = {impl: {"steps": int(s), "max_abs_error": float(e), "tolerance_failures": int(f)}
              for impl, s, e, f in re.findall(
                  r"^TG_CHECK impl=(\w+) steps=(\d+) max_abs_error=([\d.eE+-]+) failures=(\d+)", log, re.MULTILINE)}
    if set(checks) != {"qpu", "arm"} or any(c["tolerance_failures"] for c in checks.values()):
        raise ValueError(f"on-device check failed or missing: {checks}")
    if not re.search(r"^TG_CHECK_OK\s*$", log, re.MULTILINE):
        raise ValueError("TG_CHECK_OK missing")
    bench = {}
    for impl, n, prefill, decode_us, total in re.findall(
            r"^TG_BENCH impl=(\w+) tokens=(\d+) prefill_us=(\d+) decode_us=(\d+) total_us=(\d+)", log, re.MULTILINE):
        bench[impl] = {"tokens": int(n), "prefill_us": int(prefill), "decode_us": int(decode_us),
                           "total_us": int(total), "tokens_per_second": int(n) * 1e6 / int(total)}
    if set(bench) != {"qpu", "arm"}:
        raise ValueError("benchmark lines missing")
    same = re.search(r"^TG_BENCH_MATCH tokens=(\d+) of=(\d+)", log, re.MULTILINE)
    texts = dict(re.findall(r"^TG_TEXT impl=(\w+) (.*)$", log, re.MULTILINE))
    return {"checks": checks, "bench": bench, "texts": texts,
                "greedy_agreement": [int(same.group(1)), int(same.group(2))] if same else None,
                "speedup": bench["arm"]["total_us"] / bench["qpu"]["total_us"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="tiny", choices=tuple(SIZES))
    ap.add_argument("--prompt", default=None, help="benchmark/check prompt (default per size)")
    ap.add_argument("--gen", type=int, default=None, help="benchmark tokens (default 2 * ctx)")
    ap.add_argument("--chat", action="store_true", help="stay on the Pi and read prompts from the terminal")
    ap.add_argument("--chat-len", type=int, default=200, help="characters generated per prompt in --chat")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--emulate", action="store_true", help="only check the launch sequence on the emulator")
    ap.add_argument("--emulate-steps", type=int, default=0, help="also check N decode steps on the emulator")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "tinygpt")
    ap.add_argument("--tty")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = Config(vocab=1, **SIZES[args.size])
    args.gen = args.gen if args.gen is not None else 2 * cfg.ctx
    if args.emulate:
        args.emulate_steps = args.emulate_steps or 4
        c, tok, params, _ = load_model(model_path(args.size))
        params = deploy(params)
        prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPTS[args.size]
        ids = tok.encode(prompt)
        ref_ids, ref_logits = reference(c, params, ids)
        err = emulate_check(c, params, compile_kernels(c), ids, ref_ids, ref_logits, args.emulate_steps)
        print(f"emulator: prefill + {args.emulate_steps} decode steps, max |logit error| {err:.2e}")
        return
    binary, info = build(args)
    if args.build_only:
        return
    from piton.runtime.pi import PiRunner, find_tty
    if args.chat:
        # only pi-install (cs140e) forwards the terminal to the Pi; my-install does not
        tty = args.tty or find_tty()
        loader = os.environ.get("PITON_BOOTLOADER") or shutil.which("pi-install")
        if not tty or not loader:
            sys.exit("--chat needs the Pi's tty (PITON_TTY or --tty) and pi-install (PITON_BOOTLOADER)")
        print(f"uploading (~{info['upload_seconds_estimate']:.0f} s). Type a prompt and press Enter; "
              "an empty line continues; /help for commands; /quit (or Ctrl-C) to leave.", flush=True)
        os.execvp(loader, [loader, tty, str(binary)])
    runner = PiRunner(tty=args.tty, timeout=args.timeout, verbose=True)
    t0 = time.monotonic()
    log = runner._boot_recovering(str(binary))
    out = args.output or ROOT / f"benchmarks/results/tinygpt-{args.size}-{datetime.now():%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".log").write_text(log)
    result = parse_log(log)
    result.update(info=info, timestamp_utc=datetime.now(UTC).isoformat(),
                  upload_and_run_seconds=round(time.monotonic() - t0, 1))
    out.write_text(json.dumps(result, indent=2) + "\n")
    q, a = result["bench"]["qpu"], result["bench"]["arm"]
    print(f"\nTinyGPT-{args.size} on the Pi: QPU {q['tokens_per_second']:.0f} chars/s, ARM {a['tokens_per_second']:.0f}"
          f" chars/s ({result['speedup']:.2f}x); checks: QPU max err {result['checks']['qpu']['max_abs_error']:.1e},"
          f" ARM {result['checks']['arm']['max_abs_error']:.1e}. Saved {out}")


if __name__ == "__main__":
    main()
