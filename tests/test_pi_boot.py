"""Serial host regression tests; no board required."""
import os
import shutil
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


def recovering_runner(tmp_path, monkeypatch, failures, ready=True):
    """A runner whose _boot raises the given PiErrors first; power.py faked out."""
    runner = runner_for_script(tmp_path, "")
    calls = {"boots": 0, "ready": 0}

    def boot(binary):
        calls["boots"] += 1
        if failures:
            raise failures.pop(0)
        return "PITON_START x\nPITON_DONE\n"

    def ensure_ready(tty):
        calls["ready"] += 1
        return ready

    monkeypatch.setattr(runner, "_boot", boot)
    monkeypatch.setattr(pi_runtime.power, "tool", lambda tty=None: "pi_power")
    monkeypatch.setattr(pi_runtime.power, "ensure_ready", ensure_ready)
    return runner, calls


def test_boot_failure_before_start_is_retried_after_power_check(tmp_path, monkeypatch):
    runner, calls = recovering_runner(tmp_path, monkeypatch, [PiError("no answer", "")])
    assert "PITON_DONE" in runner._boot_recovering("unused")
    assert calls == {"boots": 2, "ready": 2}


def test_failure_after_program_start_is_not_retried(tmp_path, monkeypatch):
    runner, calls = recovering_runner(tmp_path, monkeypatch, [PiError("hung", "PITON_START x\n")])
    with pytest.raises(PiError, match="hung"):
        runner._boot_recovering("unused")
    assert calls["boots"] == 1


def test_second_boot_failure_is_raised(tmp_path, monkeypatch):
    runner, calls = recovering_runner(tmp_path, monkeypatch, [PiError("a", ""), PiError("b", "")])
    with pytest.raises(PiError, match="b"):
        runner._boot_recovering("unused")
    assert calls["boots"] == 2


def test_unrecoverable_pi_is_reported_without_booting(tmp_path, monkeypatch):
    runner, calls = recovering_runner(tmp_path, monkeypatch, [], ready=False)
    with pytest.raises(PiError, match="not answering"):
        runner._boot_recovering("unused")
    assert calls["boots"] == 0


def test_no_power_switch_means_no_retry(tmp_path, monkeypatch):
    runner, calls = recovering_runner(tmp_path, monkeypatch, [PiError("no answer", "")])
    monkeypatch.setattr(pi_runtime.power, "tool", lambda tty=None: None)
    with pytest.raises(PiError, match="no answer"):
        runner._boot_recovering("unused")
    assert calls == {"boots": 1, "ready": 0}



LINUX_POWER = os.path.join(pi_runtime.power.TOOLS, "pi_power_linux")
# tools/pi_power_linux bounds each gpioset with coreutils `timeout`, which macOS does not ship:
# the tests that expect the script to switch the (fake) bridge need it
needs_timeout = pytest.mark.skipif(shutil.which("timeout") is None,
                                   reason="tools/pi_power_linux needs coreutils timeout (Linux)")


def fake_bridge(tmp_path, chips=(True,), writable=True, gpioset_rc=0, vanish=False, version="v1.6.3"):
    """A sysfs / dev tree with one CP2102N per entry of `chips` (bridge n: ttyUSBn
    and, if chips[n], gpiochipN under interface 1-n:1.0) and a gpioset stub that
    logs its arguments; with vanish=True the stub removes every chip, as a bridge
    browning out would."""
    sys_root, dev, bin_dir = tmp_path / "sys", tmp_path / "dev", tmp_path / "bin"
    (sys_root / "bus" / "usb" / "drivers" / "cp210x").mkdir(parents=True)
    dev.mkdir()
    ifaces = []
    for n, chip in enumerate(chips):
        iface = sys_root / "devices" / "usb1" / f"1-{n}" / f"1-{n}:1.0"
        (iface / f"ttyUSB{n}").mkdir(parents=True)
        (sys_root / "class" / "tty" / f"ttyUSB{n}").mkdir(parents=True)
        os.symlink(iface / f"ttyUSB{n}", sys_root / "class" / "tty" / f"ttyUSB{n}" / "device")
        os.symlink(iface, sys_root / "bus" / "usb" / "drivers" / "cp210x" / f"1-{n}:1.0")
        (dev / f"ttyUSB{n}").write_text("")
        if chip:
            (iface / f"gpiochip{n}").mkdir()
            (dev / f"gpiochip{n}").write_text("")
            if not writable:
                os.chmod(dev / f"gpiochip{n}", 0o444)
        ifaces.append(iface)
    bin_dir.mkdir()
    log = tmp_path / "gpioset.log"
    gone = "".join(f"rm -rf {i}/gpiochip*; " for i in ifaces) if vanish else ""
    (bin_dir / "gpioset").write_text(
        f"#!/bin/sh\n[ \"$1\" = --version ] && echo 'gpioset (libgpiod) {version}' && exit 0\n"
        f"echo \"$@\" >> {log}\n{gone}exit {gpioset_rc}\n")
    os.chmod(bin_dir / "gpioset", 0o755)
    env = dict(os.environ, PITON_SYSFS=str(sys_root), PITON_DEVDIR=str(dev),
               PATH=f"{bin_dir}:{os.environ['PATH']}")
    env.pop("PITON_TTY", None)

    def run(cmd, tty=None):
        e = dict(env, PITON_TTY=str(dev / tty)) if tty else env
        return subprocess.run(["bash", LINUX_POWER, cmd], env=e, capture_output=True, text=True).returncode

    return run, log


@needs_timeout
def test_linux_power_switches_the_bridge_gpio0(tmp_path):
    run, log = fake_bridge(tmp_path)
    assert run("status") == 0 and run("off") == 0 and run("on") == 0
    assert log.read_text().splitlines() == ["gpiochip0 0=0", "gpiochip0 0=1"]


@needs_timeout
def test_linux_power_drives_the_chip_of_the_ttys_bridge(tmp_path):
    run, log = fake_bridge(tmp_path, chips=(True, True, False))
    assert run("status") == 1                                   # two chips and no tty: ambiguous
    assert run("off", tty="ttyUSB1") == 0 and run("on", tty="ttyUSB0") == 0
    assert log.read_text().splitlines() == ["gpiochip1 0=0", "gpiochip0 0=1"]
    assert run("status", tty="ttyUSB2") == 1                    # its bridge has no gpiochip
    assert run("status", tty="ttyUSB9") == 1                    # no such tty


def test_linux_power_without_a_usable_chip_fails_every_command(tmp_path):
    for name, kwargs in (("absent", {"chips": (False,)}), ("read-only", {"writable": False}),
                         ("gpiod-v2", {"version": "v2.1"})):
        if name == "read-only" and os.geteuid() == 0:
            continue                                            # root writes anyway
        run, log = fake_bridge(tmp_path / name, **kwargs)
        assert run("status") == 1, name
        if name != "gpiod-v2":
            assert (run("on"), run("off")) == (1, 1) and not log.exists(), name


@needs_timeout
def test_linux_power_off_succeeds_when_the_bridge_browns_out(tmp_path):
    run, _ = fake_bridge(tmp_path, gpioset_rc=1, vanish=True)
    assert run("off") == 0
    run, _ = fake_bridge(tmp_path / "tty", gpioset_rc=1, vanish=True)
    assert run("off", tty="ttyUSB0") == 0
    run, _ = fake_bridge(tmp_path / "stays", gpioset_rc=1)
    assert run("off") == 1                                      # failed, and the chip is still there


def test_power_passes_the_tty_to_the_switch(tmp_path, monkeypatch):
    power = pi_runtime.power
    tty = tmp_path / "ttyUSB3"
    tty.write_text("")
    seen = []
    monkeypatch.setattr(power, "tool", lambda tty=None: "pi_power")
    monkeypatch.setattr(power.subprocess, "run", lambda cmd, env=None, **kw: (
        seen.append((cmd[1], (env or {}).get("PITON_TTY"))), subprocess.CompletedProcess(cmd, 0))[1])
    monkeypatch.setattr(power, "handshake", lambda tty, seconds: bool(seen))
    monkeypatch.setattr(power.time, "sleep", lambda s: None)
    assert power.cycle(str(tty)) and power.available(str(tty))
    assert seen == [("off", str(tty)), ("on", str(tty)), ("status", str(tty))]


def test_linux_power_tool_needs_a_usable_switch(monkeypatch):
    power = pi_runtime.power
    monkeypatch.delenv("PITON_POWER", raising=False)
    monkeypatch.setattr(power.sys, "platform", "linux")
    monkeypatch.setattr(power.shutil, "which", lambda name: "/usr/bin/" + name)
    status = {"rc": 0}
    monkeypatch.setattr(power.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, status["rc"]))
    assert power.tool() == LINUX_POWER and os.access(LINUX_POWER, os.X_OK)
    status["rc"] = 1                                            # no cp210x chip on this machine
    assert power.tool() is None
    status["rc"] = 0
    monkeypatch.setattr(power.shutil, "which", lambda name: None)  # no libgpiod tools
    assert power.tool() is None
    monkeypatch.setattr(power.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setenv("PITON_POWER", "0")
    assert power.tool() is None


def test_linux_power_script_rejects_unknown_commands():
    proc = subprocess.run(["bash", LINUX_POWER, "bogus"], capture_output=True, text=True)
    assert proc.returncode == 2 and "usage" in proc.stderr


@needs_timeout
def test_power_cycles_the_ttys_bridge_among_two(tmp_path, monkeypatch):
    """The real tool() / cycle() on two bridges: only the given tty's chip moves."""
    power = pi_runtime.power
    _, log = fake_bridge(tmp_path, chips=(True, True))
    monkeypatch.setattr(power.sys, "platform", "linux")
    monkeypatch.delenv("PITON_POWER", raising=False)
    monkeypatch.delenv("PITON_TTY", raising=False)
    monkeypatch.setenv("PITON_SYSFS", str(tmp_path / "sys"))
    monkeypatch.setenv("PITON_DEVDIR", str(tmp_path / "dev"))
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setattr(power, "handshake", lambda tty, seconds: True)
    monkeypatch.setattr(power.time, "sleep", lambda s: None)
    tty = str(tmp_path / "dev" / "ttyUSB1")
    assert power.tool() is None                                 # ambiguous without a tty
    assert power.tool(tty) == LINUX_POWER and power.available(tty)
    assert power.cycle(tty)
    assert log.read_text().splitlines() == ["gpiochip1 0=0", "gpiochip1 0=1"]
