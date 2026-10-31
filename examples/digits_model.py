"""NumPy-only 64 -> 128 -> 64 -> 10 handwritten-digit classifier.

The model's last layer is stored with 16 outputs for QPU SIMD alignment.  Only
the first ten logits are classes.  Integer images are raw 8x8 intensities in
0..16; floating-point inputs to forward/predict are already normalized to 0..1.

Run ``python examples/digits_model.py --train`` to reproduce training.  The
official 3,823/1,797 UCI split is preserved; model selection sees only a
stratified validation subset of the official training set.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "digits"
DATASET_PATH = DATA_DIR / "dataset.npz"
MODEL_PATH = DATA_DIR / "model.npz"
METADATA_PATH = DATA_DIR / "metadata.json"
SEED = 1729
SOURCE_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
SOURCE_HASHES = {
    "optdigits.tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    "optdigits.tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
}
MODEL_SHAPES = {
    "w1": (64, 128), "b1": (128,),
    "w2": (128, 64), "b2": (64,),
    "w3": (64, 16), "b3": (16,),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_data(data: dict[str, np.ndarray]) -> None:
    for split, count in (("train", 3823), ("test", 1797)):
        images, labels = data[f"{split}_images"], data[f"{split}_labels"]
        if images.shape != (count, 64) or labels.shape != (count,):
            raise ValueError(f"Incorrect shape for official {split} split")
        if images.dtype != np.uint8 or labels.dtype != np.uint8:
            raise ValueError("Dataset must contain uint8 images and labels")
        if np.any(images > 16) or np.any(labels > 9):
            raise ValueError("Pixel values must be 0..16 and digit labels 0..9")


def prepare_data(*, download: bool = True) -> dict[str, np.ndarray]:
    """Cache the official UCI files as losslessly compressed uint8 arrays."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, filename in (("train", "optdigits.tra"), ("test", "optdigits.tes")):
        source = DATA_DIR / filename
        if not source.exists():
            if not download:
                raise FileNotFoundError(source)
            raw = urllib.request.urlopen(SOURCE_BASE + filename, timeout=30).read()
            if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[filename]:
                raise ValueError(f"Source checksum mismatch: {filename}")
            source.write_bytes(raw)
        if _sha256(source) != SOURCE_HASHES[filename]:
            raise ValueError(f"Source checksum mismatch: {filename}")
        rows = np.loadtxt(io.BytesIO(source.read_bytes()), delimiter=",", dtype=np.int16)
        if rows.ndim != 2 or rows.shape[1] != 65:
            raise ValueError(f"Expected 64 pixel columns plus label: {filename}")
        if np.any(rows[:, :64] < 0) or np.any(rows[:, :64] > 16):
            raise ValueError(f"Invalid pixel range: {filename}")
        if np.any(rows[:, 64] < 0) or np.any(rows[:, 64] > 9):
            raise ValueError(f"Invalid digit labels: {filename}")
        data[f"{split}_images"] = rows[:, :64].astype(np.uint8)
        data[f"{split}_labels"] = rows[:, 64].astype(np.uint8)
    _validate_data(data)
    np.savez_compressed(DATASET_PATH, **data)
    return data


def load_data() -> dict[str, np.ndarray]:
    """Return train_images/labels and test_images/labels in official row order."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset; run {__file__} --prepare-data")
    with np.load(DATASET_PATH, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    _validate_data(data)
    return data


def normalize(images: np.ndarray) -> np.ndarray:
    """Convert raw 0..16 digit pixels to contiguous float32 0..1 inputs."""
    images = np.asarray(images)
    if images.ndim not in (1, 2) or images.shape[-1] != 64:
        raise ValueError("Expected shape (64,) or (batch, 64)")
    if not np.all(np.isfinite(images)) or np.any(images < 0) or np.any(images > 16):
        raise ValueError("Raw digit pixels must be finite and in 0..16")
    return np.ascontiguousarray(images, dtype=np.float32) / np.float32(16)


def load_model(path: Path | str = MODEL_PATH) -> dict[str, np.ndarray]:
    """Load float32 row-major input-by-output weights and padded final biases."""
    with np.load(path, allow_pickle=False) as stored:
        model = {name: np.ascontiguousarray(stored[name]) for name in stored.files}
    if set(model) != set(MODEL_SHAPES):
        raise ValueError("Unexpected model parameters")
    for name, shape in MODEL_SHAPES.items():
        value = model[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid parameter: {name}")
    if np.any(model["w3"][:, 10:]) or np.any(model["b3"][10:]):
        raise ValueError("Final six SIMD padding outputs must have zero parameters")
    return model


def forward(images: np.ndarray, model: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Return all 16 logits; six trailing outputs are SIMD padding, not classes."""
    model = load_model() if model is None else model
    images = np.asarray(images)
    if np.issubdtype(images.dtype, np.integer):
        x = normalize(images)
    else:
        x = np.ascontiguousarray(images, dtype=np.float32)
        if x.ndim not in (1, 2) or x.shape[-1] != 64:
            raise ValueError("Expected shape (64,) or (batch, 64)")
        if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError("Floating-point digit inputs must be normalized to 0..1")
    h1 = np.maximum(x @ model["w1"] + model["b1"], np.float32(0))
    h2 = np.maximum(h1 @ model["w2"] + model["b2"], np.float32(0))
    return h2 @ model["w3"] + model["b3"]


def predict(images: np.ndarray, model: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Predict digit labels using only the first ten output logits."""
    return np.argmax(forward(images, model)[..., :10], axis=-1)


def _split_training(labels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stratify validation within official training data; never inspect test labels."""
    train, validation = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = int(round(len(indices) * 0.15))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return rng.permutation(train), np.asarray(sorted(validation), dtype=np.int64)


def _loss_accuracy(x: np.ndarray, y: np.ndarray, model: dict) -> tuple[float, float]:
    logits = forward(x, model)[:, :10]
    logits = logits - logits.max(axis=1, keepdims=True)
    log_prob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    loss = float(-log_prob[np.arange(len(y)), y].mean())
    return loss, float((logits.argmax(axis=1) == y).mean())


def train(*, epochs: int = 160, seed: int = SEED) -> dict:
    """Train Adam; choose one frozen checkpoint by validation cross-entropy only."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    data = load_data()
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = _split_training(data["train_labels"], rng)
    x = normalize(data["train_images"])
    y = data["train_labels"]
    model = {}
    for layer, (n_in, n_out) in enumerate(((64, 128), (128, 64), (64, 16)), start=1):
        model[f"w{layer}"] = (rng.standard_normal((n_in, n_out)) * (2 / n_in)**0.5).astype(np.float32)
        model[f"b{layer}"] = np.zeros(n_out, dtype=np.float32)
    model["w3"][:, 10:] = 0
    first_moment = {key: np.zeros_like(value) for key, value in model.items()}
    second_moment = {key: np.zeros_like(value) for key, value in model.items()}
    learning_rate, weight_decay, batch_size = 0.001, 0.0001, 64
    step, best_loss, best_epoch, best_model = 0, float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_indices)
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            xb, yb = x[indices], y[indices]
            h1 = np.maximum(xb @ model["w1"] + model["b1"], 0)
            h2 = np.maximum(h1 @ model["w2"] + model["b2"], 0)
            logits = h2 @ model["w3"][:, :10] + model["b3"][:10]
            logits -= logits.max(axis=1, keepdims=True)
            grad = np.exp(logits)
            grad /= grad.sum(axis=1, keepdims=True)
            grad[np.arange(len(yb)), yb] -= 1
            grad /= len(yb)
            gradients = {}
            gradients["w3"] = np.zeros_like(model["w3"])
            gradients["w3"][:, :10] = h2.T @ grad
            gradients["b3"] = np.zeros_like(model["b3"])
            gradients["b3"][:10] = grad.sum(axis=0)
            dh2 = (grad @ model["w3"][:, :10].T) * (h2 > 0)
            gradients["w2"], gradients["b2"] = h1.T @ dh2, dh2.sum(axis=0)
            dh1 = (dh2 @ model["w2"].T) * (h1 > 0)
            gradients["w1"], gradients["b1"] = xb.T @ dh1, dh1.sum(axis=0)
            step += 1
            for name in model:
                g = gradients[name]
                if name.startswith("w"):
                    g += weight_decay * model[name]
                first_moment[name] *= 0.9
                first_moment[name] += 0.1 * g
                second_moment[name] *= 0.999
                second_moment[name] += 0.001 * g * g
                m = first_moment[name] / (1 - 0.9**step)
                v = second_moment[name] / (1 - 0.999**step)
                model[name] -= learning_rate * m / (np.sqrt(v) + 1e-8)
        validation_loss, validation_accuracy = _loss_accuracy(x[validation_indices], y[validation_indices], model)
        history.append({"epoch": epoch, "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_model = {name: value.copy() for name, value in model.items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}: validation loss={validation_loss:.6f} accuracy={validation_accuracy:.4%}", flush=True)
    # Freeze and save before the first evaluation of the official test split.
    np.savez_compressed(MODEL_PATH, **best_model)
    frozen_model = load_model()
    validation_loss, validation_accuracy = _loss_accuracy(x[validation_indices], y[validation_indices], frozen_model)
    training_loss, training_accuracy = _loss_accuracy(x[train_indices], y[train_indices], frozen_model)
    test_loss, test_accuracy = _loss_accuracy(normalize(data["test_images"]), data["test_labels"], frozen_model)
    correct = int((predict(data["test_images"], frozen_model) == data["test_labels"]).sum())
    metadata = {
        "dataset": "UCI Optical Recognition of Handwritten Digits",
        "dataset_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "doi": "10.24432/C50P49", "license": "CC-BY-4.0",
        "authors": ["E. Alpaydin", "C. Kaynak"], "dataset_year": 1998,
        "source_sha256": SOURCE_HASHES,
        "artifact_sha256": {"dataset.npz": _sha256(DATASET_PATH), "model.npz": _sha256(MODEL_PATH)},
        "architecture": [64, 128, 64, 10], "exported_architecture": [64, 128, 64, 16],
        "hidden_activation": "ReLU", "outputs": "raw logits; argmax only first ten",
        "normalization": "float32(raw uint8 pixel) / 16", "dtype": "float32",
        "weight_layout": "C contiguous (input_features, output_features)",
        "parameter_bytes_with_padding": sum(value.nbytes for value in frozen_model.values()),
        "training": {
            "implementation": "NumPy-only Adam, cross-entropy, He normal initialization",
            "numpy_version": np.__version__, "seed": seed, "epochs": epochs,
            "batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay,
            "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
            "official_train_count": 3823, "fit_count": len(train_indices),
            "validation_count": len(validation_indices), "test_count": 1797,
            "validation_fraction": 0.15, "split": "stratified within official training split only",
            "validation_indices": validation_indices.tolist(),
            "checkpoint_rule": "lowest validation cross-entropy over fixed epoch budget",
            "selected_epoch": best_epoch, "test_used_for_selection": False,
            "test_evaluated_after_checkpoint_frozen": True,
        },
        "metrics": {
            "fit_loss": training_loss, "fit_accuracy": training_accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
            "official_test_loss": test_loss, "official_test_accuracy": test_accuracy,
            "official_test_correct": correct, "official_test_count": 1797,
        },
        "history": history,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Selected epoch {best_epoch}; frozen test: {correct}/1797 = {test_accuracy:.4%}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true", help="Download/check official UCI data and cache uint8 arrays")
    parser.add_argument("--train", action="store_true", help="Reproduce deterministic NumPy model training")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.prepare_data:
        prepare_data()
    if args.train:
        train(epochs=args.epochs, seed=args.seed)
    if not args.prepare_data and not args.train:
        data, model = load_data(), load_model()
        predicted = predict(data["test_images"], model)
        correct = int((predicted == data["test_labels"]).sum())
        print(f"Frozen model test accuracy: {correct}/1797 = {correct / 1797:.4%}")


if __name__ == "__main__":
    main()
