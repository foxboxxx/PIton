"""Build and test a trained digit classifier on the attached bare-metal Pi.

    .venv/bin/python examples/digits_pi.py
    .venv/bin/python examples/digits_pi.py --build-only

The frozen model is trained on the host, not on the Pi. Timings cover resident
three-layer inference, including launch overhead and fused bias/ReLU. Upload,
normalization, allocation, validation, and serial output are outside the timer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from digits_model import DATA_DIR, load_data, load_model, predict
from digits_reference import reference_with_bounds
from nn_kernels import dense_layer_kernel, dense_tiles
from piton import compiler
from piton.runtime.pi import PiRunner, libpi_root
from piton.tools.compile import emit_c


BATCHES = (1, 16, 64, 256, 1797)
IMPLEMENTATIONS = ("gpu", "arm_c", "arm_vfp")
VALIDATIONS = (*IMPLEMENTATIONS, *(name + "_timed" for name in IMPLEMENTATIONS))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c_array(name: str, values: np.ndarray) -> str:
    values = np.asarray(values).ravel()
    floating = values.dtype.kind == "f"
    c_type, width = ("float", 8) if floating else ("uint8_t", 32)
    # Hexadecimal literals preserve every exported float32 bit in the C binary.
    strings = [float(value).hex() + "f" if floating else str(int(value)) for value in values]
    lines = [", ".join(strings[i:i + width]) + "," for i in range(0, len(strings), width)]
    return f"static const {c_type} {name}[{len(values)}] = {{\n" + "\n".join(lines) + "\n};\n"


def build_firmware(build_dir: Path, repeats: int) -> tuple[Path, dict, np.ndarray, np.ndarray]:
    data, model = load_data(), load_model()
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    for filename, expected in metadata["artifact_sha256"].items():
        if sha256(DATA_DIR / filename) != expected:
            raise ValueError(f"Frozen artifact checksum mismatch: {filename}")
    images, labels = data["test_images"], data["test_labels"]
    reference, bounds = reference_with_bounds(images, model)
    predictions = reference[:, :10].argmax(axis=1)
    np.testing.assert_array_equal(predictions, predict(images, model))
    test_count = len(images)
    if test_count != BATCHES[-1]:
        raise ValueError("The benchmark requires the complete official 1,797-image test set")
    padded_count = (test_count + 3) & ~3
    build_dir.mkdir(parents=True, exist_ok=True)
    config = ["#pragma once", f"#define NN_TEST_COUNT {test_count}",
              f"#define NN_PADDED_COUNT {padded_count}", f"#define NN_REPEATS {repeats}"]
    layers = []
    signature = dict.fromkeys(("x_ptr", "w_ptr", "bias_ptr", "y_ptr"), "*fp32")
    signature["M"] = "i32"
    for layer, (k, n) in enumerate(((64, 128), (128, 64), (64, 16)), start=1):
        bm, bn, bk = dense_tiles(n)
        constants = dict(N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                         BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, ADD_BIAS=True, RELU=layer < 3)
        grid = ((padded_count + bm - 1) // bm) * ((n + bn - 1) // bn)
        compiled = compiler.compile(dense_layer_kernel, signature, constants,
                                    num_stages=2, grid=(grid,))
        header, source = emit_c(compiled, f"dense{layer}")
        (build_dir / f"dense{layer}.h").write_text(header)
        (build_dir / f"dense{layer}.c").write_text(source)
        (build_dir / f"dense{layer}.qasm").write_text(compiled.asm["qasm"])
        config += [f"#define NN_BM{layer} {bm}", f"#define NN_BN{layer} {bn}"]
        layers.append(dict(layer=layer, input_features=k, stored_outputs=n,
                           tile=[bm, bn, bk], max_grid=grid,
                           instructions=len(compiled.words), constants=constants,
                           code_sha256=hashlib.sha256(b"".join(
                               int(word).to_bytes(8, "little") for word in compiled.words)).hexdigest()))
        print(f"Layer {layer}: {k} -> {n}, tile {bm}x{bn}x{bk}, "
              f"{len(compiled.words)} QPU instructions", flush=True)
    (build_dir / "nn_config.h").write_text("\n".join(config) + "\n")
    arrays = {"nn_pixels": images, "nn_labels": labels,
              **{f"nn_{name}": value for name, value in model.items()},
              "nn_reference": reference.astype(np.float32),
              "nn_error_bound": bounds.max(axis=1).astype(np.float32)}
    (build_dir / "nn_data.h").write_text(
        "#pragma once\n#include <stdint.h>\n" + "\n".join(
            c_array(name, value) for name, value in arrays.items()))
    sources = [ROOT / "pi/examples/digits" / name for name in ("main.c", "Makefile")]
    sources += [ROOT / "pi/runtime" / name for name in ("piton_rt.c", "piton_rt.h")]
    sources += [ROOT / "pi/examples/matmul" / name
                for name in ("arm_sgemm.c", "arm_sgemm.h", "arm_kernel_vfp.S")]
    for source in sources:
        shutil.copyfile(source, build_dir / source.name)
    course_root = libpi_root()
    if course_root is None:
        raise RuntimeError("Set CS240LX_2025_PATH to the course repository containing libpi")
    env = {**os.environ, "CS240LX_2025_PATH": course_root}
    built = subprocess.run(["make", "-s", "RUN=0"], cwd=build_dir, env=env,
                           capture_output=True, text=True)
    (build_dir / "build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        raise RuntimeError(f"Firmware build failed:\n{built.stdout}\n{built.stderr}")
    binary = build_dir / "main.bin"
    tracked_sources = sources + sorted((ROOT / "piton").rglob("*.py")) + [
        ROOT / "examples" / name for name in
        ("digits_pi.py", "digits_model.py", "digits_reference.py", "nn_kernels.py")]
    manifest = dict(
        architecture=[64, 128, 64, 10], stored_architecture=[64, 128, 64, 16],
        test_count=test_count, padded_count=padded_count, repeats=repeats, layers=layers,
        model_parameters_bytes=sum(value.nbytes for value in model.values()),
        artifact_sha256=metadata["artifact_sha256"], firmware_sha256=sha256(binary),
        firmware_bytes=binary.stat().st_size,
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in tracked_sources},
        numerical_check="All 16 logits per image against FP64 reference; "
                        "1.1 * propagated FP32 RTZ row bound + 2e-5; "
                        "all argmax predictions must match exactly",
        max_allowed_absolute_error=float(1.1 * bounds.max() + 2e-5),
        expected_correct=int((predictions == labels).sum()),
        dataset_source="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
    )
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return binary, manifest, predictions, labels


def parse_log(log: str, expected_predictions: np.ndarray, labels: np.ndarray,
              repeats: int) -> dict:
    """Reject incomplete runs, invalid timings, numeric failures or wrong predictions."""
    if not re.search(r"^NN_CHECK_OK\s*$", log, re.MULTILINE) or "NN_CHECK_FAILED" in log:
        raise ValueError("Pi neural-network validation did not finish successfully")
    if not re.search(r"^PITON_DONE\s*$", log, re.MULTILINE):
        raise ValueError("Pi completion marker is missing")
    times: dict[tuple[str, int], list[int]] = {}
    for impl, batch, padded, elapsed in re.findall(
        r"^NN_TIME impl=(\w+) batch=(\d+) padded=(\d+) us=(\d+)\s*$", log, re.MULTILINE
    ):
        batch, padded, elapsed = int(batch), int(padded), int(elapsed)
        if impl not in IMPLEMENTATIONS or batch not in BATCHES or padded != ((batch + 3) & ~3):
            raise ValueError("Unexpected implementation or batch dimensions in timing output")
        if elapsed <= 0:
            raise ValueError("Inference timing must be positive")
        times.setdefault((impl, batch), []).append(elapsed)
    if sum(len(samples) for samples in times.values()) != len(re.findall(r"^NN_TIME\b", log, re.MULTILINE)):
        raise ValueError("Malformed Pi timing record")
    timings = []
    for batch in BATCHES:
        for impl in IMPLEMENTATIONS:
            samples = times.get((impl, batch), [])
            if len(samples) != repeats:
                raise ValueError(f"Expected {repeats} timings for {impl}, batch {batch}; got {len(samples)}")
            median = statistics.median(samples)
            timings.append(dict(implementation=impl, batch=batch, padded=(batch + 3) & ~3,
                                times_us=samples, min_us=min(samples), median_us=median,
                                images_per_second=batch * 1e6 / median))
    validation = []
    for impl, checked, failures, mismatches, correct, error in re.findall(
        r"^NN_VALIDATE impl=(\w+) checked=(\d+) failures=(\d+) pred_mismatch=(\d+) "
        r"correct=(\d+) max_abs_error=([\deE.+-]+)\s*$", log, re.MULTILINE
    ):
        record = dict(implementation=impl, checked=int(checked), failures=int(failures),
                      prediction_mismatches=int(mismatches), correct=int(correct),
                      max_abs_error=float(error))
        if (impl not in VALIDATIONS or record["checked"] != len(labels) * 16 or
                record["failures"] or record["prediction_mismatches"] or
                record["correct"] != int((expected_predictions == labels).sum()) or
                not np.isfinite(record["max_abs_error"]) or record["max_abs_error"] < 0):
            raise ValueError(f"Invalid Pi numerical validation: {record}")
        validation.append(record)
    if sorted(row["implementation"] for row in validation) != sorted(VALIDATIONS):
        raise ValueError("Expected exactly one initial and final validation for each implementation")
    if len(validation) != len(re.findall(r"^NN_VALIDATE\b", log, re.MULTILINE)):
        raise ValueError("Malformed Pi validation record")
    batch_validation = []
    for impl, batch, checked, failures, mismatches, correct, error in re.findall(
        r"^NN_BATCH_CHECK impl=(\w+) batch=(\d+) checked=(\d+) failures=(\d+) pred_mismatch=(\d+) "
        r"correct=(\d+) max_abs_error=([\deE.+-]+)\s*$", log, re.MULTILINE
    ):
        batch = int(batch)
        record = dict(implementation=impl, batch=batch, checked=int(checked), failures=int(failures),
                      prediction_mismatches=int(mismatches), correct=int(correct),
                      max_abs_error=float(error))
        if (impl not in IMPLEMENTATIONS or batch not in BATCHES or record["checked"] != batch * 16 or
                record["failures"] or record["prediction_mismatches"] or
                record["correct"] != int((expected_predictions[:batch] == labels[:batch]).sum()) or
                not np.isfinite(record["max_abs_error"]) or record["max_abs_error"] < 0):
            raise ValueError(f"Invalid Pi batch validation: {record}")
        batch_validation.append(record)
    expected_pairs = sorted((impl, batch) for impl in IMPLEMENTATIONS for batch in BATCHES)
    if sorted((row["implementation"], row["batch"]) for row in batch_validation) != expected_pairs:
        raise ValueError("Expected exactly one numerical check for each implementation and batch")
    if len(batch_validation) != len(re.findall(r"^NN_BATCH_CHECK\b", log, re.MULTILINE)):
        raise ValueError("Malformed Pi batch validation record")
    prediction_blocks = re.findall(r"^NN_PRED (\d+)\s*\n([\d\s]+?)^NN_PRED_END\s*$",
                                   log, re.MULTILINE)
    if len(prediction_blocks) != 1:
        raise ValueError("Expected exactly one complete Pi prediction block")
    count, block = prediction_blocks[0]
    digits = "".join(block.split())
    if int(count) != len(labels) or len(digits) != len(labels):
        raise ValueError("Incomplete Pi predictions")
    predictions = np.fromiter((int(value) for value in digits), dtype=np.uint8)
    np.testing.assert_array_equal(predictions, expected_predictions)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    correct = int((predictions == labels).sum())
    clocks = re.findall(r"^NN_CLOCK name=(\w+) mhz=(\d+)\s*$", log, re.MULTILINE)
    if sorted(name for name, _ in clocks) != ["arm", "core", "sdram", "v3d"] or any(int(mhz) <= 0 for _, mhz in clocks):
        raise ValueError("Missing or invalid firmware clocks")
    setup = re.findall(r"^NN_SETUP us=(\d+)\s*$", log, re.MULTILINE)
    if len(setup) != 1 or int(setup[0]) <= 0:
        raise ValueError("Missing or invalid setup timing")
    return dict(
        timings=timings, validation=validation, batch_validation=batch_validation,
        predictions=predictions.tolist(), labels=labels.tolist(),
        correct=correct, total=len(labels), accuracy=correct / len(labels),
        confusion_matrix_true_rows_predicted_columns=confusion.tolist(),
        clocks_mhz={name: int(mhz) for name, mhz in clocks}, setup_us=int(setup[0]),
        timing_scope="Median resident three-layer inference; all launch/uniform/cache-clear "
                     "overhead and bias/ReLU included. Excludes upload, allocation, input "
                     "normalization, validation and serial printing. Both implementations "
                     "pad batch 1 to four rows and batch 1797 to 1800 rows.",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/digits")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tty")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/digits-pi-2026-10-30.json")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    binary, manifest, expected, labels = build_firmware(args.build_dir.resolve(), args.repeats)
    print(f"Built {binary.stat().st_size:,}-byte Pi firmware; "
          f"host accuracy {(expected == labels).mean():.2%}", flush=True)
    if args.build_only:
        return
    runner = PiRunner(tty=args.tty, timeout=args.timeout, verbose=args.verbose)
    started = time.monotonic()
    log = runner._boot(str(binary))
    total_seconds = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    log_path.write_text(log)
    # Preserve the raw firmware log even if an incomplete/bad run is rejected.
    result = parse_log(log, expected, labels, args.repeats)
    result.update(manifest=manifest, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  upload_and_run_seconds=total_seconds, raw_log=str(log_path.relative_to(ROOT))
                  if log_path.is_relative_to(ROOT) else str(log_path))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Pi verified: {result['correct']}/{result['total']} correct ({result['accuracy']:.2%}); "
          "all 28,752 logits and all predictions passed", flush=True)
    for batch in BATCHES:
        entries = {row["implementation"]: row for row in result["timings"] if row["batch"] == batch}
        gpu, cpu = entries["gpu"], entries["arm_vfp"]
        print(f"Batch {batch:4d} (padded {gpu['padded']:4d}): QPU {gpu['median_us']/1000:.3f} ms, "
              f"ARM VFP {cpu['median_us']/1000:.3f} ms, "
              f"{cpu['median_us']/gpu['median_us']:.2f}x speedup, "
              f"{gpu['images_per_second']:,.0f} images/s", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
