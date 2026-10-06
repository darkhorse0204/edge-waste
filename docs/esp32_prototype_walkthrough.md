<!-- esp32_prototype_walkthrough.md - step-by-step build of the prototype with an esp32, dht22, mq-135 and one servo -->

# ESP32 prototype walkthrough (ESP32 + DHT22 + MQ-135 + servo)

You have no moisture sensor, so the **DHT22 humidity (RH %)** is used as the moisture channel. It senses the air next to the item, so it responds more slowly
and more weakly than a probe on the item. Expect a modest signal: put the DHT22 close to the item (3–4 cm) under a small cardboard hood and wait about 10 seconds
after placing the item before pressing the button. The software treats it exactly like the moisture channel (the firmware sends humidity × 10).

Everything else (camera, classifier, decision logic, calibration) is in `docs/hardware_prototype_plan.md`; this page is the fast path for your parts.

## 0. What you need on the desk

ESP32 DevKit + USB cable (data cable, not charge-only), DHT22 (module or bare), MQ-135 module, SG90 servo, breadboard, jumper wires,
**two 10 kΩ resistors** (for the MQ-135, see below), cardboard and tape. Optional: 3 LEDs with 220 Ω, a buzzer (the firmware works without them; the BOOT button is the trigger).

## 1. Install the tools (15 minutes, once)

1. **Arduino IDE 2.x** from arduino.cc.
2. **USB driver.** Plug the ESP32 in. If Device Manager shows no "COM" port, install the driver for the chip printed on the board (CP2102 → Silicon Labs CP210x, CH340 → WCH CH340).
3. **ESP32 board package.** Arduino IDE → File → Preferences → "Additional boards manager URLs", paste
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json` → OK. Tools → Board → Boards Manager → search "esp32" → install **esp32 by Espressif Systems**.
4. **Libraries** (Tools → Manage Libraries): install **ESP32Servo**, **DHT sensor library** (Adafruit) and, when asked, **Adafruit Unified Sensor**.
5. On the laptop: `pip install pyserial` (already done on this machine).

## 2. Wiring (power the ESP32 from USB, nothing else)

| Part | Part pin | ESP32 pin | Note |
|---|---|---|---|
| DHT22 | VCC / + | **3V3** | |
| | GND / − | GND | |
| | DATA / OUT | **GPIO4** | bare 4-pin DHT22: add a 10 kΩ resistor between DATA and 3V3 (pin 1 is VCC, 2 DATA, 3 not used, 4 GND, seen from the front) |
| MQ-135 | VCC | **VIN** (the 5 V pin) | the heater needs 5 V |
| | GND | GND | |
| | AO | → 10 kΩ → **GPIO34**, and a second 10 kΩ from GPIO34 to GND | **voltage divider: AO can reach 5 V, the ESP32 pin tolerates only 3.3 V** |
| Servo SG90 | red | **VIN** (5 V) | |
| | brown or black | GND | |
| | orange or yellow | **GPIO18** | |
| Trigger | BOOT button on the board (GPIO0) | – | press after start-up to capture an item |
| Optional | green / red / yellow LED | GPIO25 / 26 / 27 through 220 Ω to GND | |
| Optional | active buzzer + | GPIO14, buzzer − to GND | |

Join all grounds. Do not connect the MQ-135 AO straight to the ESP32. If you have 10 kΩ + 20 kΩ instead, put the 10 kΩ on the AO side and the 20 kΩ to GND and set
`GAS_DIVIDER = 1.5` in the sketch.

## 3. Upload the firmware (5 minutes)

1. Open `hardware/esp32/sorter_node_esp32/sorter_node_esp32.ino` in the Arduino IDE.
2. Tools → Board → **ESP32 Dev Module**; Tools → Port → your COM port; Tools → Upload Speed 115200 if uploads fail.
3. Click Upload. If it stays on "Connecting…", hold the **BOOT** button until the upload starts, then release.
4. Tools → Serial Monitor, 115200 baud, line ending "Newline". You should see `READY`. Type `P` → `PONG`; type `S` → `{"m":...,"g":...,"t":...,"h":...}`.
   - `m` near 300–700 means a humidity of 30–70% (it is humidity × 10). `m` = 0 means the DHT22 gives no reading: check the DATA wire and the pull-up.
   - `g` should be a number between about 50 and 600 and rise when you hold a spirit swab or vinegar near the MQ-135.
5. **Close the Serial Monitor** (it blocks the port for Python).

## 4. Bring-up tests from Python (10 minutes)

```
python scripts/check_node.py --port COM5               # live readings: breathe on the DHT22, hold a damp tissue near it, smell source near the MQ-135
python scripts/check_node.py --port COM5 --angles      # tray angles 60 / 90 / 120: change ANGLE_LEFT, ANGLE_RIGHT in the sketch and upload again until the tray tilts about 25-30 degrees each way
python scripts/check_node.py --port COM5 --actions     # R, H, V, C: servo, LEDs, buzzer
```

The MQ-135 needs warm-up: keep it powered for **30 minutes at least** before trusting numbers (24 hours is better). Start this now and build the frame meanwhile.

## 5. Build the frame (1–2 hours)

Shoebox base; the servo on a bridge in the middle with its axis running front to back; a 12 × 14 cm cardboard tray with white paper and clear tape glued on the servo horn
(lips on the front and back edges only); a cup under each side (left: auto-sorted, right: hazardous, mark it red); a small cardboard hood on one side of the tray with the
**DHT22 and MQ-135 inside it, 3–4 cm from where the item sits**; phone camera on a stack of books 25–30 cm straight above; lamp from the side; white background.
Use light, safe props only (bottle, cup, cardboard, can, cloth, a dead AA battery, a broken earphone).

## 6. Camera

Phone: install **IP Webcam** (Android) or DroidCam, start the server, note the address, and use `--camera http://<phone-ip>:8080/video`
(phone and laptop on the same Wi-Fi or hotspot). Laptop camera: `--camera 0`.

## 7. Software test without any hardware (do this first if you can)

```
python scripts/prototype_sorter.py --mock-node --image data/processed/battery/garbage12__battery__e6790782bd.jpg
```

Expected: hazardous decision.

## 8. Calibrate the contamination index (about 90 minutes, same sitting as the demo, sensors warm)

```
python scripts/calibrate_prototype_oci.py collect --port COM5
> air           clean air, nothing near the sensors
> smell         spirit swab or vinegar a few cm from the MQ-135
> dry           room humidity with a dry item under the hood (wait 10 s)
> wet           a soaked tissue next to the DHT22 under the hood (wait 15 s; gives the highest humidity you will use)
> sample porous 0        cardboard or paper, clean, under the hood, wait 10 s, then type the line
> sample porous 1        the same kind of item with 1 teaspoon of curd, ketchup or juice
> sample rigid 0 / rigid 1   plastic cup or lid, clean / with residue
> quit
```

Collect 2 categories × residue 0, 0.5, 1, 2, 4 teaspoons × 4 repeats = 40 samples (at least 8 clean and 8 contaminated). Between samples let humidity and gas return to the baseline
(fan with a card, wait a minute). Then `python scripts/calibrate_prototype_oci.py fit` prints the AUC, thresholds and sensitivity and writes `runs/prototype/oci_calibration.json`.
Humidity moves slowly and weakly, so the combined model may lean on the gas sensor. That is a result worth reporting.

## 9. Run the prototype

```
python scripts/prototype_sorter.py --port COM5 --camera 0
```

Place an item, wait about 10 seconds under the hood, press the **BOOT button** (or SPACE in the window). Decisions go to `runs/prototype/log.csv`.
Green LED + tilt left = auto-sorted; red LED + long beep + tilt right = hazardous; yellow = manual review or contamination reject.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| No COM port | data cable, driver (CP210x or CH340), another USB port |
| Upload stuck on "Connecting…" | hold BOOT while uploading |
| `no answer from the node` in Python | wrong COM port, Serial Monitor still open, sketch not uploaded; press EN once |
| ESP32 resets when the servo moves | brown-out: add a 470 µF capacitor across VIN and GND, or power the servo from a separate 5 V supply with a common ground |
| `m` always 0 | DHT22 DATA on GPIO4? 10 kΩ pull-up on bare sensors? wait 3 s after start-up |
| `g` stuck at 0 or 1023 | divider wired wrongly or MQ-135 not on VIN 5 V; check that AO goes through the resistors |
| Gas reading drifts | longer warm-up, shield from draughts, keep the room still |
| Everything goes to manual review | plain white background, side light, item fills the green box; `--review-budget 0.2` for a demo |
| Item does not slide off | smoother tray, larger angle, lighter item |

## 11. Safety

USB power only. Keep liquids away from the ESP32. Use dead or small cells only, never damaged lithium cells.
