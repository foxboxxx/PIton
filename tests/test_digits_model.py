"""Frozen real-data model, independent reference, and strict QPU inference."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from digits_model import (DATA_DIR, MODEL_SHAPES, forward, load_data, load_model,
                          normalize, predict)
from digits_reference import reference_with_bounds
from nn_kernels import dense_layer_kernel, dense_tiles
import piton.compiler as pc
from piton.runtime.launcher import run_emulator


@pytest.fixture(scope="module")
def artifacts():
    return load_data(), load_model(), json.loads((DATA_DIR / "metadata.json").read_text())


def test_frozen_artifacts_hashes_shapes_and_nontrivial_weights(artifacts):
    data, model, metadata = artifacts
    for filename, expected_hash in metadata["artifact_sha256"].items():
        assert hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest() == expected_hash
    for name, shape in MODEL_SHAPES.items():
        assert model[name].shape == shape
        assert model[name].dtype == np.float32
        assert np.isfinite(model[name]).all()
    for name in ("w1", "w2", "w3"):
        weights = model[name] if name != "w3" else model[name][:, :10]
        assert np.count_nonzero(weights) == weights.size
        assert np.any(weights < 0) and np.any(weights > 0)
        assert np.unique(weights).size > weights.size // 2
    assert sum(value.nbytes for value in model.values()) == 70464
    np.testing.assert_array_equal(model["w3"][:, 10:], 0)
    np.testing.assert_array_equal(model["b3"][10:], 0)
    for split, count in (("train", 3823), ("test", 1797)):
        assert data[f"{split}_images"].shape == (count, 64)
        assert data[f"{split}_images"].dtype == np.uint8
        assert data[f"{split}_labels"].shape == (count,)
        assert data[f"{split}_labels"].dtype == np.uint8
        assert data[f"{split}_images"].min() == 0
        assert data[f"{split}_images"].max() == 16
        np.testing.assert_array_equal(np.unique(data[f"{split}_labels"]), np.arange(10))
    validation_indices = metadata["training"]["validation_indices"]
    assert len(validation_indices) == len(set(validation_indices)) == 572
    assert min(validation_indices) >= 0 and max(validation_indices) < 3823
    assert metadata["training"]["test_used_for_selection"] is False
    assert metadata["training"]["test_evaluated_after_checkpoint_frozen"] is True


def test_official_test_split_accuracy_and_normalization(artifacts):
    data, model, metadata = artifacts
    images, labels = data["test_images"], data["test_labels"]
    logits = forward(images, model)
    assert logits.shape == (1797, 16) and logits.dtype == np.float32
    assert np.isfinite(logits).all()
    np.testing.assert_array_equal(logits[:, 10:], 0)
    np.testing.assert_array_equal(predict(images, model), predict(normalize(images), model))
    correct = int((predict(images, model) == labels).sum())
    assert correct == metadata["metrics"]["official_test_correct"] == 1743
    assert correct / len(labels) == metadata["metrics"]["official_test_accuracy"]
    # This is the official test split, not a training or validation resample.
    np.testing.assert_array_equal(np.bincount(labels, minlength=10),
                                  [178, 182, 177, 183, 181, 182, 181, 179, 174, 180])


def test_predict_excludes_six_padding_outputs(artifacts):
    data, model, _ = artifacts
    forced = {name: value.copy() for name, value in model.items()}
    forced["w3"].fill(0)
    forced["b3"][:10] = -1
    forced["b3"][7] = -0.5
    logits = forward(data["test_images"][:3], forced)
    np.testing.assert_array_equal(logits.argmax(axis=1), 10)
    np.testing.assert_array_equal(predict(data["test_images"][:3], forced), 7)


def test_float64_reference_and_bounds_cover_numpy_all_test_images(artifacts):
    data, model, _ = artifacts
    reference, bounds = reference_with_bounds(data["test_images"], model)
    assert reference.shape == bounds.shape == (1797, 16)
    assert reference.dtype == bounds.dtype == np.float64
    assert np.isfinite(reference).all() and np.isfinite(bounds).all()
    assert np.all(bounds[:, :10] > 0)
    np.testing.assert_array_equal(reference[:, 10:], 0)
    np.testing.assert_array_equal(bounds[:, 10:], 0)
    errors = np.abs(forward(data["test_images"], model).astype(np.float64) - reference)
    assert np.all(errors <= bounds)
    np.testing.assert_array_equal(np.argmax(reference[:, :10], axis=1),
                                  predict(data["test_images"], model))
    normalized_reference, normalized_bounds = reference_with_bounds(normalize(data["test_images"]), model)
    np.testing.assert_array_equal(normalized_reference, reference)
    np.testing.assert_array_equal(normalized_bounds, bounds)


def _strict_layer(x, weight, bias, relu):
    m, k = x.shape
    n = weight.shape[1]
    bm, bn, bk = dense_tiles(n)
    grid = (((m + bm - 1) // bm) * ((n + bn - 1) // bn),)
    constants = dict(N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                     BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, ADD_BIAS=True, RELU=relu)
    signature = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
    signature["M"] = "i32"
    compiled = pc.compile(dense_layer_kernel, signature, constants, grid=grid, num_stages=2)
    actual = np.full((m, n), np.nan, dtype=np.float32)
    run_emulator(compiled, grid,
                 dict(x_ptr=x, w_ptr=weight, bias_ptr=bias, y_ptr=actual, M=m), strict=True)
    assert np.isfinite(actual).all()
    return actual


def test_trained_model_strict_emulator_error_bound(artifacts):
    data, model, _ = artifacts
    images = data["test_images"][:8]
    expected, bounds = reference_with_bounds(images, model)
    actual = normalize(images)
    for layer in (1, 2, 3):
        actual = _strict_layer(actual, model[f"w{layer}"], model[f"b{layer}"], relu=layer < 3)
    # This independent accumulated bound accounts for all three dense layers,
    # including propagated hidden-layer rounding and ReLU threshold crossings.
    error = np.abs(actual.astype(np.float64) - expected)
    assert np.all(error <= bounds), f"max error/bound ratio: {np.max(error[:, :10] / bounds[:, :10])}"
    np.testing.assert_array_equal(actual[:, 10:], 0)
    np.testing.assert_array_equal(actual[:, :10].argmax(axis=1), expected[:, :10].argmax(axis=1))
