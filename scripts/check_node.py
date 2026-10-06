# check_node.py - bring-up test for the esp32 or arduino node: ping, live sensor readings, tray angles and every action
"""  python scripts/check_node.py --port COM5              # ping, then print sensor readings every second (Ctrl+C to stop)
     python scripts/check_node.py --port COM5 --angles    # move the tray to 60, 90, 120 degrees (finds ANGLE_LEFT / ANGLE_RIGHT)
     python scripts/check_node.py --port COM5 --actions   # run R, H, V, C once each (tray, leds, buzzer)
     python scripts/check_node.py --mock                  # no hardware"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edgewaste.applications.prototype_node import ACTION_TEXT, MockNode, Node  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM5")
ap.add_argument("--mock", action="store_true")
ap.add_argument("--angles", action="store_true")
ap.add_argument("--actions", action="store_true")
a = ap.parse_args()
node = MockNode() if a.mock else Node(a.port)
print("node answers: OK")
if a.angles:
    for deg in (90, 60, 90, 120, 90):
        print(f"tray -> {deg} degrees"); node.angle(deg); time.sleep(1.2)
elif a.actions:
    for c in "RHVC":
        input(f"press Enter to run {c}: {ACTION_TEXT[c]} ")
        node.act(c)
else:
    print("moisture channel = humidity x10 (0 means the DHT22 gave no reading); gas = 0..1023. Breathe on the sensors or hold a damp tissue near them to see the numbers move.")
    try:
        while True:
            r = node.sensors()
            print(f"  m={r[0]:4d}  g={r[1]:4d}" if r else "  no reply")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
