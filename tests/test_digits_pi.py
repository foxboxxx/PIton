"""Complete serial benchmark records are required before reporting Pi results."""

from pathlib import Path
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from digits_pi import BATCHES, IMPLEMENTATIONS, VALIDATIONS, parse_log


@pytest.fixture
def serial_run():
    expected = (np.arange(1797) % 10).astype(np.uint8)
    labels = expected.copy()
    labels[::37] = (labels[::37] + 1) % 10
    repeats = 5
    correct = int((expected == labels).sum())
    lines = ["PIton neural network", "NN_MODEL test_count=1797 padded_count=1800 repeats=5",
             "NN_CLOCK name=arm mhz=700", "NN_CLOCK name=core mhz=250",
             "NN_CLOCK name=v3d mhz=250", "NN_CLOCK name=sdram mhz=400", "NN_SETUP us=12345"]
    for impl in VALIDATIONS:
        lines.append(f"NN_VALIDATE impl={impl} checked={len(labels) * 16} failures=0 "
                     f"pred_mismatch=0 correct={correct} max_abs_error=0.00021")
    for batch in BATCHES:
        for impl in IMPLEMENTATIONS:
            padded = (batch + 3) & ~3
            for repeat in range(repeats):
                lines.append(f"NN_TIME impl={impl} batch={batch} padded={padded} us={1000 + batch + repeat}")
            prefix_correct = int((expected[:batch] == labels[:batch]).sum())
            lines.append(f"NN_BATCH_CHECK impl={impl} batch={batch} checked={batch * 16} failures=0 "
                         f"pred_mismatch=0 correct={prefix_correct} max_abs_error=0.00017")
    lines += [f"NN_PRED {len(labels)}"]
    digits = "".join(str(int(value)) for value in expected)
    lines += [digits[offset:offset + 32] for offset in range(0, len(digits), 32)]
    lines += ["NN_PRED_END", "NN_CHECK_OK", "PITON_DONE"]
    return "\n".join(lines) + "\n", expected, labels, repeats


def parse(serial_run, log=None):
    original, expected, labels, repeats = serial_run
    return parse_log(original if log is None else log, expected, labels, repeats)


def remove_first(log, prefix):
    return re.sub(r"^" + re.escape(prefix) + r"[^\n]*\n", "", log, count=1, flags=re.MULTILINE)


def replace_first(log, prefix, old, new):
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            assert old in line
            lines[index] = line.replace(old, new, 1)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"Missing fixture record: {prefix}")


def test_complete_serial_run_reports_only_complete_verified_results(serial_run):
    log, expected, labels, repeats = serial_run
    assert sum(line.startswith("NN_VALIDATE ") for line in log.splitlines()) == 6
    assert sum(line.startswith("NN_TIME ") for line in log.splitlines()) == 75
    assert sum(line.startswith("NN_BATCH_CHECK ") for line in log.splitlines()) == 15
    result = parse(serial_run)
    assert result["predictions"] == expected.tolist()
    assert result["labels"] == labels.tolist()
    assert result["correct"] == int((expected == labels).sum())
    assert result["accuracy"] == float((expected == labels).mean())
    assert result["setup_us"] == 12345
    assert len(result["validation"]) == 6
    assert len(result["timings"]) == 15
    assert len(result["batch_validation"]) == 15
    for record in result["batch_validation"]:
        assert record["checked"] == record["batch"] * 16
        assert record["correct"] == int((expected[:record["batch"]] == labels[:record["batch"]]).sum())
    for row in result["timings"]:
        assert len(row["times_us"]) == repeats
        assert row["min_us"] == 1000 + row["batch"]
        assert row["median_us"] == 1002 + row["batch"]
        assert row["padded"] == (row["batch"] + 3) & ~3
    confusion = np.asarray(result["confusion_matrix_true_rows_predicted_columns"])
    assert confusion.shape == (10, 10)
    assert confusion.sum() == 1797
    assert confusion.trace() == result["correct"]


@pytest.mark.parametrize("prefix", [
    "NN_TIME impl=gpu batch=1 ",
    "NN_TIME impl=arm_c batch=256 ",
    "NN_TIME impl=arm_vfp batch=1797 ",
    "NN_VALIDATE impl=gpu ",
    "NN_VALIDATE impl=arm_vfp_timed ",
    "NN_BATCH_CHECK impl=gpu batch=1 ",
    "NN_BATCH_CHECK impl=arm_c batch=64 ",
    "NN_BATCH_CHECK impl=arm_vfp batch=1797 ",
])
def test_reject_missing_required_record(serial_run, prefix):
    with pytest.raises(ValueError):
        parse(serial_run, remove_first(serial_run[0], prefix))


@pytest.mark.parametrize("old,new", [
    ("padded=4", "padded=8"),
    ("us=1001", "us=0"),
    ("us=1001", "us=nan"),
    ("us=1001", "us=inf"),
    ("us=1001", "us=-1001"),
    ("impl=gpu", "impl=unknown"),
    ("batch=1", "batch=2"),
])
def test_reject_invalid_time_record(serial_run, old, new):
    log = replace_first(serial_run[0], "NN_TIME impl=gpu batch=1 ", old, new)
    with pytest.raises(ValueError):
        parse(serial_run, log)


@pytest.mark.parametrize("prefix", ["NN_VALIDATE impl=gpu ", "NN_BATCH_CHECK impl=gpu batch=1797 "])
@pytest.mark.parametrize("old,new", [
    ("checked=28752", "checked=28751"),
    ("failures=0", "failures=1"),
    ("pred_mismatch=0", "pred_mismatch=1"),
    ("max_abs_error=0.000", "max_abs_error=-0.000"),
    ("max_abs_error=0.00021", "max_abs_error=nan"),
    ("max_abs_error=0.00021", "max_abs_error=inf"),
    ("max_abs_error=0.00021", "max_abs_error=1e309"),
])
def test_reject_failed_or_nonfinite_validation(serial_run, prefix, old, new):
    # The two firmware record types differ only in their finite error value.
    if prefix.startswith("NN_BATCH_CHECK"):
        old, new = old.replace("0.00021", "0.00017"), new.replace("0.00021", "0.00017")
    log = replace_first(serial_run[0], prefix, old, new)
    with pytest.raises(ValueError):
        parse(serial_run, log)


@pytest.mark.parametrize("prefix", ["NN_VALIDATE impl=gpu ", "NN_BATCH_CHECK impl=gpu batch=1 "])
def test_reject_incorrect_validation_accuracy(serial_run, prefix):
    log, expected, labels, _ = serial_run
    correct = int((expected == labels).sum()) if prefix.startswith("NN_VALIDATE") else 0
    with pytest.raises(ValueError):
        parse(serial_run, replace_first(log, prefix, f"correct={correct}", f"correct={correct + 1}"))


@pytest.mark.parametrize("prefix", [
    "NN_TIME impl=gpu batch=1 ",
    "NN_VALIDATE impl=arm_c ",
    "NN_BATCH_CHECK impl=arm_vfp batch=16 ",
])
def test_reject_duplicate_record(serial_run, prefix):
    line = next(line for line in serial_run[0].splitlines() if line.startswith(prefix))
    with pytest.raises(ValueError):
        parse(serial_run, serial_run[0] + line + "\n")


@pytest.mark.parametrize("marker", ["NN_CHECK_OK\n", "PITON_DONE\n", "NN_PRED_END\n"])
def test_reject_missing_completion(serial_run, marker):
    with pytest.raises(ValueError):
        parse(serial_run, serial_run[0].replace(marker, ""))


def test_reject_failure_marker_even_when_success_marker_present(serial_run):
    with pytest.raises(ValueError):
        parse(serial_run, serial_run[0] + "NN_CHECK_FAILED\n")


def test_reject_wrong_prediction(serial_run):
    log = serial_run[0].replace("NN_PRED 1797\n0", "NN_PRED 1797\n1", 1)
    with pytest.raises((ValueError, AssertionError)):
        parse(serial_run, log)


def test_reject_truncated_prediction_block(serial_run):
    log = serial_run[0].replace("NN_PRED 1797\n0", "NN_PRED 1797\n", 1)
    with pytest.raises(ValueError):
        parse(serial_run, log)


@pytest.mark.parametrize("extra", [
    "NN_TIME impl=gpu batch=1 padded=4 us=nan",
    "NN_VALIDATE impl=gpu checked=28752 failures=0 pred_mismatch=0 correct=1748 max_abs_error=nan",
    "NN_BATCH_CHECK impl=gpu batch=1 checked=16 failures=0 pred_mismatch=0 correct=0 max_abs_error=inf",
])
def test_reject_malformed_extra_record_in_otherwise_complete_run(serial_run, extra):
    with pytest.raises(ValueError):
        parse(serial_run, serial_run[0] + extra + "\n")


@pytest.mark.parametrize("prefix", ["NN_CLOCK name=arm ", "NN_SETUP "])
def test_reject_missing_benchmark_context(serial_run, prefix):
    with pytest.raises(ValueError):
        parse(serial_run, remove_first(serial_run[0], prefix))


@pytest.mark.parametrize("prefix,old,new", [
    ("NN_CLOCK name=arm ", "mhz=700", "mhz=0"),
    ("NN_SETUP ", "us=12345", "us=0"),
])
def test_reject_invalid_benchmark_context(serial_run, prefix, old, new):
    with pytest.raises(ValueError):
        parse(serial_run, replace_first(serial_run[0], prefix, old, new))
