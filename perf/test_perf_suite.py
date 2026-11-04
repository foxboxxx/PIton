"""Every suite kernel against its numpy reference in the strict emulator.

Compiler optimisations found with perf/analyze.py must keep all of these
passing; they cover each kernel family the Pi benchmarks measure.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perf.suite import CASES, compile_case, default_check, lcg_floats, lcg_u32  # noqa: E402
from piton.runtime.launcher import run_emulator  # noqa: E402
from piton.runtime import pi as pi_runtime  # noqa: E402


def test_vectorised_lcg_matches_the_runtime_twin():
    np.testing.assert_array_equal(lcg_floats(257, 12345), pi_runtime.lcg_floats(257, 12345))
    s, ref = 99, []
    for _ in range(5):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        ref.append(s)
    np.testing.assert_array_equal(lcg_u32(5, 99), np.array(ref, dtype=np.uint32))


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_matches_reference(case):
    ck, run = compile_case(case, "emu")
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    run_emulator(ck, run.grid, values)
    for name, expected in run.expected.items():
        error = default_check(run, name, values[name], expected)
        assert error is None, f"{case.name}.{name}: {error}"


@pytest.mark.parametrize("name,constant", [("copy", "BLOCK"), ("add", "BLOCK_SIZE"), ("exp", "BLOCK"),
                                           ("idiv", "BLOCK")])
def test_blocks_too_large_for_registers_are_rerolled(name, constant):
    """BLOCK=2048 runs out of registers straight-line; compile() falls back to the chunk loop."""
    from perf.suite import CASES_BY_NAME
    from perf.tune import variant
    ck, run = compile_case(variant(CASES_BY_NAME[name], constant, 2048), "emu")
    assert ck.stats["rerolled"]
    values = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in run.inputs.values.items()}
    run_emulator(ck, run.grid, values)
    for out, expected in run.expected.items():
        assert default_check(run, out, values[out], expected) is None
