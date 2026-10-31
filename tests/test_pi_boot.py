"""Serial host regression tests; no board required."""
import os
import subprocess
import sys

import pytest

from piton.runtime.pi import PiError, PiRunner
import piton.runtime.pi as pi_runtime


def runner_for_script(tmp_path, source):
    script = tmp_path / "bootloader.py"
    script.write_text(source)
    runner = object.__new__(PiRunner)
    runner.bootloader = sys.executable
    runner.tty = str(script)
    runner.timeout = 1.0
    runner.verbose = False
    return runner


def test_done_in_same_pipe_write_as_output(tmp_path):
    runner = runner_for_script(tmp_path,
        "import os, time\n"
        "os.write(1, b'PITON_TIME 42\\nPITON_BUF c 2\\n00000000 3f800000\\nPITON_DONE\\n')\n"
        "time.sleep(3)\n")
    log = runner._boot("unused")
    parsed = runner._parse(log, [], ["c"])
    assert parsed.times_us == [42]
    assert parsed.outputs["c"].tolist() == [0, 0x3f800000]


def test_marker_split_across_pipe_reads(tmp_path):
    runner = runner_for_script(tmp_path,
        "import os, time\n"
        "os.write(1, b'PITON_TIME 1\\nPITON_DO')\n"
        "time.sleep(.05)\n"
        "os.write(1, b'NE\\n')\n"
        "time.sleep(3)\n")
    assert "PITON_DONE" in runner._boot("unused")


def test_panic_is_reported_without_waiting_for_process_exit(tmp_path):
    runner = runner_for_script(tmp_path,
        "import os, time\n"
        "os.write(1, b'PANIC: failed launch\\n')\n"
        "time.sleep(3)\n")
    with pytest.raises(PiError, match="failed launch"):
        runner._boot("unused")


def test_panic_preceding_done_in_same_write_is_failure(tmp_path):
    runner = runner_for_script(tmp_path,
        "import os, time\n"
        "os.write(1, b'PANIC: failed launch\\nPITON_DONE\\n')\n"
        "time.sleep(3)\n")
    with pytest.raises(PiError, match="failed launch"):
        runner._boot("unused")


class UnreapableBootloader:
    """A readable completed UART log whose host process cannot be reaped.

    No child is launched or signalled: only the pipe and Popen interface are
    simulated. In particular, this exercises the USB-driver cleanup failure
    without requiring a real adapter or letting a test leave a child behind.
    """

    def __init__(self, payload, *, signal_error=False):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, payload)
        finally:
            os.close(write_fd)
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.signal_error = signal_error
        self.actions = []

    def terminate(self):
        self.actions.append("terminate")
        if self.signal_error:
            raise ProcessLookupError("bootloader already exited")

    def kill(self):
        self.actions.append("kill")
        if self.signal_error:
            raise ProcessLookupError("bootloader already exited")

    def wait(self, timeout=None):
        assert timeout is not None and 0 < timeout <= 5, "cleanup must never wait indefinitely"
        self.actions.append(("wait", timeout))
        raise subprocess.TimeoutExpired("bootloader", timeout)

    def poll(self):
        return None


@pytest.mark.parametrize("signal_error", [False, True])
def test_unreapable_bootloader_preserves_success_and_closes_pipe(tmp_path, monkeypatch, signal_error):
    runner = runner_for_script(tmp_path, "")
    # Run twice to catch cleanup that leaves the following invocation unusable.
    processes = [UnreapableBootloader(b"PITON_TIME 7\nPITON_DONE\n", signal_error=signal_error)
                 for _ in range(2)]
    pending = iter(processes)
    monkeypatch.setattr(pi_runtime.subprocess, "Popen", lambda *args, **kwargs: next(pending))
    monkeypatch.setattr(pi_runtime.time, "sleep", lambda _: None)
    for process in processes:
        assert runner._boot("unused") == "PITON_TIME 7\nPITON_DONE\n"
        assert process.actions == ["terminate", ("wait", 5), "kill", ("wait", 1)]
        assert process.stdout.closed


@pytest.mark.parametrize("payload, message", [
    (b"PANIC: failed launch\nPITON_DONE\n", "failed launch"),
    (b"pi exited\n", "Pi run failed"),
])
def test_unreapable_bootloader_preserves_firmware_failure(tmp_path, monkeypatch, payload, message):
    runner = runner_for_script(tmp_path, "")
    process = UnreapableBootloader(payload)
    monkeypatch.setattr(pi_runtime.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(PiError, match=message):
        runner._boot("unused")
    assert process.actions == ["terminate", ("wait", 5), "kill", ("wait", 1)]
    assert process.stdout.closed
