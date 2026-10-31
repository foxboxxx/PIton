"""Float64 MLP reference and conservative per-output VC4 rounding bounds.

The VC4's float32 operations round toward zero.  Its relative error is bounded
by 2**-23 per normal operation, rather than the 2**-24 bound for round-to-nearest.
Each length-K dense output performs K products, K sums, and one bias addition.
ReLU is 1-Lipschitz, so it preserves an absolute error bound between layers.
"""

from __future__ import annotations

import numpy as np


def reference_with_bounds(
    images: np.ndarray, model: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 ideal logits and float64 absolute bounds, both (N, 16).

    Integer inputs are raw 0..16 pixels. Floating inputs must be normalized to
    0..1 and are first represented as float32, matching the QPU input buffer.
    Weights and biases are interpreted as the exact exported float32 values.
    The six final SIMD padding outputs have zero logits and zero error bounds.
    """
    images = np.asarray(images)
    if images.ndim != 2 or images.shape[1] != 64:
        raise ValueError("Expected digit images with shape (batch, 64)")
    if not np.isfinite(images).all():
        raise ValueError("Digit images must contain finite values")
    if np.issubdtype(images.dtype, np.integer):
        if np.any(images < 0) or np.any(images > 16):
            raise ValueError("Raw digit pixels must be in 0..16")
        activation = images.astype(np.float64) / 16.0
    else:
        if np.any(images < 0) or np.any(images > 1):
            raise ValueError("Floating inputs must be normalized to 0..1")
        activation = images.astype(np.float32).astype(np.float64)
    error = np.zeros_like(activation)
    for layer, (k, n) in enumerate(((64, 128), (128, 64), (64, 16)), start=1):
        weight32 = np.asarray(model[f"w{layer}"], dtype=np.float32)
        bias32 = np.asarray(model[f"b{layer}"], dtype=np.float32)
        if weight32.shape != (k, n) or bias32.shape != (n,):
            raise ValueError(f"Incorrect parameter shape in layer {layer}")
        if not np.isfinite(weight32).all() or not np.isfinite(bias32).all():
            raise ValueError(f"Non-finite parameter in layer {layer}")
        weight, bias = weight32.astype(np.float64), bias32.astype(np.float64)
        gamma_numerator = (2 * k + 1) * 2.0**-23
        gamma = gamma_numerator / (1.0 - gamma_numerator)
        absolute_weight = np.abs(weight)
        error = (error @ absolute_weight) * (1.0 + gamma) + gamma * (
            np.abs(activation) @ absolute_weight + np.abs(bias)
        )
        activation = activation @ weight + bias
        if layer < 3:
            activation = np.maximum(activation, 0.0)
    return activation, error
