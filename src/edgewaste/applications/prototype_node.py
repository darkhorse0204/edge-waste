# prototype_node.py - serial link to the arduino sensor and actuator node of the 2-day prototype, plus a mock node for testing without hardware
"""The laptop talks to the Arduino (firmware: hardware/arduino/sorter_node/sorter_node.ino) with one-line commands.

`Node.open(port)` opens a real serial port. `MockNode` behaves the same but needs no hardware: it prints what the
tray would do and returns plausible sensor values, so the whole software path can be tested before the parts arrive.
"""
from __future__ import annotations

import json
import random
import time

ACTIONS = {"recycle": "R", "hazardous": "H", "review": "V", "contaminated": "C"}
ACTION_TEXT = {"R": "green led, tray tilts LEFT (auto-sorted bin)", "H": "red led + long beep, tray tilts RIGHT (hazardous bin)",
               "V": "yellow led + 3 short beeps (manual review)", "C": "yellow led + 2 long beeps (contamination reject)"}


class Node:
    """Real Arduino over USB serial."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 5.0):
        try:
            import serial  # pyserial
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pyserial is missing: run  pip install pyserial") from e
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(3.0)                       # arduino and esp32 boards reset when the port opens; the esp32 also prints boot text
        self.ser.reset_input_buffer()
        if not any(self.ping() for _ in range(5)):
            raise SystemExit(f"no answer from the node on {port}: check the cable, the port name, that the sketch is uploaded and that the Serial Monitor is closed")

    def _ask(self, cmd: str, expect_json: bool = False):
        """Send one command and return the reply line. Unknown lines (boot text) are skipped; BTN lines are remembered."""
        self.ser.write((cmd + chr(10)).encode())
        end = time.time() + 6.0
        while time.time() < end:
            raw = self.ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue
            if raw == "BTN":
                self._pending_button = True
                continue
            if expect_json:
                if raw.startswith("{") and raw.endswith("}"):
                    return json.loads(raw)
                continue
            if raw in ("PONG", "OK", "READY"):
                return raw
        return None

    _pending_button = False

    def ping(self) -> bool:
        return self._ask("P") == "PONG"

    def sensors(self) -> tuple[int, int] | None:
        """(moisture_raw, gas_raw), each 0..1023, or None if the node did not answer."""
        d = self._ask("S", expect_json=True)
        return None if d is None else (int(d["m"]), int(d["g"]))

    def act(self, code: str) -> None:
        self._ask(code)

    def angle(self, deg: int) -> None:
        self._ask(f"A{int(deg)}")

    def button(self) -> bool:
        """True once per button press (non-blocking)."""
        if self._pending_button:
            self._pending_button = False
            return True
        if self.ser.in_waiting:
            raw = self.ser.readline().decode(errors="ignore").strip()
            return raw == "BTN"
        return False


class MockNode:
    """Pretends to be the Arduino. `wet=True` makes the sensors look like a contaminated item."""

    def __init__(self, wet: bool = False, seed: int = 0):
        self.wet = wet
        self.rng = random.Random(seed)
        print("[mock node] no hardware: actions are printed, sensor values are invented")

    def sensors(self) -> tuple[int, int]:
        r = self.rng
        if self.wet:
            return int(450 + r.gauss(0, 15)), int(560 + r.gauss(0, 20))
        return int(600 + r.gauss(0, 15)), int(330 + r.gauss(0, 15))

    def act(self, code: str) -> None:
        print(f"[mock node] {code}: {ACTION_TEXT.get(code, '?')}")

    def angle(self, deg: int) -> None:
        print(f"[mock node] tray to {deg} degrees")

    def button(self) -> bool:
        return False

    def ping(self) -> bool:
        return True
