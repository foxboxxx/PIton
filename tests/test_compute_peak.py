"""Ensure the peak experiment performs and verifies the arithmetic it counts."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "compute_peak_pi", Path(__file__).resolve().parents[1] / "benchmarks" / "compute_peak_pi.py")
peak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peak)


def test_peak_loop_counts_executed_operations_and_pairs_alus():
    result = peak.run(iterations=3, unroll=16, compile_only=True)
    assert result["vector_adds"] == 64
    assert result["vector_multiplies"] == 64
    assert result["paired_arithmetic_instructions"] >= 60
    assert result["loop_instructions"] < 80


def test_peak_recurrence_returns_all_outputs_in_strict_emulator():
    x = np.linspace(-1, 1, 12 * 64, dtype=np.float32)
    y = np.full_like(x, np.nan)
    peak.compute_kernel[(12,)](x, y, ITERS=3, UNROLL=16)
    alpha = float(np.float32(0.9999))
    power = alpha ** 48
    expected = power * x.astype(np.float64) + alpha * (1 - power) / (1 - alpha)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, expected, rtol=2e-5, atol=1e-5)
