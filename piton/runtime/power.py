"""Power-cycle the Pi through the HUGE UART board (CS340LX lab 3).

The board's CP2102N drives the MOSFET on the Pi's 5 V rail from its GPIO0;
tools/pi_power sends the vendor request.  Low GPIO0 cuts the power.  The bridge
chip itself browns out within ~0.3 s of that and re-enumerates about a second
later with GPIO0 back high, so a cycle always ends with the Pi powered and the
tty briefly gone.

The bootloader announces itself by sending GET_PROG_INFO (0x11112222, little
endian) every ~0.3 s while it waits, so a Pi that is ready for a program can be
told from a wedged one by listening to the UART.

  PITON_POWER=0   never power-cycle (default: whenever tools/pi_power works)
"""
from __future__ import annotations

import os
import select
import subprocess
import sys
import termios
import time
from typing import Optional

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools")
TOOL = os.path.join(TOOLS, "pi_power")
GET_PROG_INFO = bytes.fromhex("22221111")


def tool() -> Optional[str]:
    """tools/pi_power, built on first use (macOS IOKit)."""
    if os.environ.get("PITON_POWER", "1") == "0":
        return None
    src = TOOL + ".c"
    if sys.platform == "darwin" and os.path.exists(src) and (
            not os.path.exists(TOOL) or os.path.getmtime(TOOL) < os.path.getmtime(src)):
        subprocess.run(["clang", "-O2", "-o", TOOL, src, "-framework", "IOKit", "-framework", "CoreFoundation"],
                       capture_output=True)
    return TOOL if os.access(TOOL, os.X_OK) else None


def _power(*args) -> bool:
    path = tool()
    return bool(path) and subprocess.run([path, *args], capture_output=True).returncode == 0


def available() -> bool:
    return _power("status")


def handshake(tty: str, seconds: float) -> bool:
    """True if the bootloader's GET_PROG_INFO arrives on `tty` within `seconds`."""
    try:
        fd = os.open(tty, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        attrs = termios.tcgetattr(fd)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = attrs[5] = termios.B115200
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        data = b""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                try:
                    data += os.read(fd, 4096)
                except (BlockingIOError, OSError):
                    pass
                if GET_PROG_INFO in data:
                    return True
                data = data[-8:]
        return False
    except termios.error:
        return False
    finally:
        os.close(fd)


def cycle(tty: str, timeout: float = 20.0) -> bool:
    """Cut the Pi's power and wait until its bootloader is listening again."""
    if not _power("off"):
        return False
    time.sleep(0.5)
    end = time.monotonic() + timeout
    while not _power("on"):                   # the bridge is re-enumerating
        if time.monotonic() > end:
            return False
        time.sleep(0.25)
    while time.monotonic() < end:
        if os.path.exists(tty) and handshake(tty, 1.0):
            return True
        time.sleep(0.25)
    return False


def ensure_ready(tty: str) -> bool:
    """Make sure the bootloader is waiting for a program, power-cycling if it is not.

    Returns False only when it is not listening and cannot be power-cycled back.
    """
    if handshake(tty, 1.0):
        return True
    return available() and cycle(tty)
