"""TinyGPT resident on the Pi (examples/tinygpt_pi.py): the launch sequence of
pi/examples/tinygpt/main.c on the emulator, the weight packing, and the log
parser. The hardware run itself is examples/tinygpt_pi.py (not under pytest:
it needs the board and takes ~15 s to upload)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import tinygpt_pi as tp
from tinygpt_model import SIZES, Config, load_model


@pytest.fixture(scope="module")
def tiny():
    cfg, tok, params, _ = load_model("tiny")
    return cfg, tok, tp.deploy(params)


def test_resident_launch_sequence_matches_reference(tiny):
    """Prefill, KV-cache decode steps and the head on the last row, launch by
    launch as main.c issues them, against the float64 reference."""
    cfg, tok, params = tiny
    prompt = tok.encode("The QPU ")
    ref_ids, ref_logits = tp.reference(cfg, params, prompt)
    kernels = tp.compile_kernels(cfg, verbose=False)
    err = tp.emulate_check(cfg, params, kernels, prompt, ref_ids, ref_logits, steps=3)
    assert err < 2e-3, err


def test_reference_rows_are_next_token_logits(tiny):
    cfg, tok, params = tiny
    prompt = tok.encode("The ")
    ids, logits = tp.reference(cfg, params, prompt)
    assert len(ids) == cfg.ctx and len(logits) == cfg.ctx - len(prompt) + 1
    # greedy continuation: each reference token is the argmax of the row before it
    np.testing.assert_array_equal(ids[len(prompt):], logits[:-1].argmax(axis=1))


def test_pack_offsets_aligned_and_lossless(tiny):
    cfg, _, params = tiny
    packed, offsets = tp.pack(cfg, params)
    assert packed.dtype == np.float16
    for name in tp.tensor_order(cfg):
        assert offsets[name] % 16 == 0
        v = params[name].ravel()
        np.testing.assert_array_equal(packed[offsets[name]:offsets[name] + v.size].astype(np.float32), v)


def test_deploy_rounds_to_fp16():
    p = tp.deploy({"w": np.array([1.0 + 2 ** -12, 0.1], np.float32)})
    np.testing.assert_array_equal(p["w"], np.array([1.0, np.float16(0.1)], np.float32))
    with pytest.raises(ValueError):
        tp.deploy({"w": np.array([1e5], np.float32)})


@pytest.mark.parametrize("size", list(SIZES))
def test_kernel_specs_cover_every_size(size):
    """Tiles divide the shapes and every launch grid fits in the compiled one."""
    cfg = Config(vocab=96, **SIZES[size])
    specs = tp.kernel_specs(cfg)
    assert len(specs) == 15
    for name, (_, _, constants, grid, macros) in specs.items():
        if name.startswith("lin_"):
            assert constants["K"] % constants["BLOCK_SIZE_K"] == 0
        if name.endswith("_d") and "BM" in macros:
            assert grid[0] == tp.cdiv(constants["N"], macros["BN"])


LOG = """PITON_START
TG_CHECK impl=qpu steps=25 max_abs_error=7.915497e-5 failures=0 max_abs_logit=1.470369e+1
TG_CHECK impl=arm steps=25 max_abs_error=1.716614e-5 failures=0 max_abs_logit=1.470369e+1
TG_CHECK_OK
TG_BENCH impl=qpu tokens=64 prefill_us=680 decode_us=24851 total_us=25531
TG_TEXT impl=qpu The QPU instructions
TG_BENCH impl=arm tokens=64 prefill_us=5205 decode_us=87391 total_us=92596
TG_TEXT impl=arm The QPU instructions
TG_BENCH_MATCH tokens=64 of=64
PITON_DONE
"""


def test_parse_log():
    r = tp.parse_log(LOG)
    assert r["checks"]["qpu"]["max_abs_error"] == pytest.approx(7.915497e-5)
    assert r["bench"]["qpu"]["tokens_per_second"] == pytest.approx(64e6 / 25531)
    assert r["speedup"] == pytest.approx(92596 / 25531)
    assert r["greedy_agreement"] == [64, 64]
    assert r["texts"]["arm"] == "The QPU instructions"


@pytest.mark.parametrize("bad", [
    LOG.replace("failures=0 max_abs_logit=1.470369e+1\nTG_CHECK impl=arm", "failures=3 max_abs_logit=1\nTG_CHECK impl=arm"),
    LOG.replace("TG_CHECK_OK\n", "TG_CHECK_FAILED\n"),
    LOG.replace("TG_BENCH impl=arm", "TG_BENCH_X impl=arm"),
])
def test_parse_log_rejects_failed_runs(bad):
    with pytest.raises(ValueError):
        tp.parse_log(bad)
