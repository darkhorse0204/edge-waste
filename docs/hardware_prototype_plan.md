<!-- hardware_prototype_plan.md - two-day, minimum-cost bench prototype: parts, wiring, build steps, calibration, schedule and tests -->

# Two-day prototype: camera + sensors + tilt tray (minimum cost)

**Goal.** A working bench unit that takes one item, photographs it, reads a moisture and a gas sensor,
decides with the project's own software (hybrid classifier, uncertainty, conformal family sets, contamination
index, decision engine) and physically sorts it into one of two bins or asks a person (led + buzzer).

**How it stays cheap and fast.** The laptop is the "edge computer" (no Raspberry Pi), a tilting tray replaces the
conveyor, one servo does all the sorting, and the phone is the camera. The software already exists in this repository;
only an Arduino sketch, one wiring job and a small card-board frame are new.

| Full design (report) | 2-day prototype | Why this is acceptable |
|---|---|---|
| Raspberry Pi 5, ONNX models | laptop with the same PyTorch models | same decisions, only slower to port; Pi is the next stage |
| Conveyor belt and servo diverter, 9 bins | tilt tray: left bin, right bin, level = review | demonstrates the routing logic; family bins are collapsed to "auto-sorted" |
| Calibrated sensors, 5 residue levels | 40-sample mini calibration the same evening | gives real data and a real threshold, small sample |
| Federated learning on 2+ units | not built | simulated result already in the report |

## 1. Bill of materials (India, approximate, check stock and price first)

Prices are planning estimates (about ±30%). Buy locally or borrow from a VIT lab: delivery will not arrive in two days.

| # | Part | Qty | Approx. cost (₹) | Notes |
|---|---|---|---|---|
| 1 | Arduino Uno clone (or Nano) with USB cable | 1 | 400–600 | any 5 V Arduino with a 10-bit ADC; avoid ESP32 here (noisy ADC, 3.3 V) |
| 2 | MQ-135 gas sensor module | 1 | 150–250 | needs 5 V; heater draws about 150 mA; warm it up |
| 3 | Capacitive soil-moisture sensor v1.2 | 1 | 100–200 | the capacitive type, not the two-prong resistive one (it corrodes) |
| 4 | SG90 micro servo (MG90S if available) | 1 | 120–250 | tilts the tray |
| 5 | Breadboard (400 points) and jumper wires (M–M, M–F) | 1 set | 150–250 | |
| 6 | LEDs green, red, yellow; three 220 Ω resistors; active buzzer 5 V; push button; 470 µF capacitor | 1 bag | 60–100 | |
| 7 | Cardboard, ice-cream sticks, hot glue or tape, white chart paper, 2–3 small plastic boxes or cups | – | 0–150 | frame, tray and bins |
| 8 | Camera: your phone with a free IP-webcam app, or a USB webcam | 1 | 0 or 500–800 | phone is enough |
| | **Total** | | **about 1,000–1,800** (+500–800 if you buy a webcam) | |

Skip: conveyor, Raspberry Pi, load cell, metal detector, relays, cloud service.

## 2. Wiring

```
Arduino Uno
  A0  <- MQ-135 AO          5V -> MQ-135 VCC      GND -> MQ-135 GND
  A1  <- moisture AOUT      5V -> moisture VCC    GND -> moisture GND
  D9  -> servo signal (orange)   servo red -> 5V   servo brown -> GND   (470 uF across 5V and GND next to the servo)
  D4  -> 220 ohm -> green LED -> GND
  D5  -> 220 ohm -> red LED   -> GND
  D6  -> 220 ohm -> yellow LED-> GND
  D7  -> buzzer +   ;  buzzer - -> GND
  D2  -> push button -> GND   (the internal pull-up is used)
```

If the Arduino resets while the servo moves, power the servo from a separate 5 V source (phone charger on a cut USB
cable, or 4 × AA) and join the grounds.

## 3. Mechanics (about 1 hour)

1. **Frame.** A shoebox (about 25 × 20 × 10 cm) is the base. The servo is fixed on a small bridge in the middle of the box,
   with its axis running front to back, so that the tray tilts left and right.
2. **Tray.** A 12 × 14 cm piece of stiff cardboard covered with white chart paper and clear tape (smooth, so items slide).
   Glue a 1.5 cm lip along the front and back edges only; the left and right edges stay open. Glue it to the servo horn.
3. **Bins.** A cup or box under the left edge (auto-sorted) and under the right edge, marked in red (hazardous).
4. **Moisture sensor.** Tape the blade flat in the tray centre, sensing end up, under one layer of clear tape.
5. **Gas sensor.** On the box top, 3 cm from the tray centre, with a small cardboard hood on the side facing the room to cut draughts.
6. **Camera.** The phone on a stack of books, 25–30 cm straight above the tray; desk lamp from the side; plain white background.
7. **Push button** on the front edge of the box.

Use safe, light props (under 150 g): a plastic water bottle, a paper cup, a cardboard piece, an aluminium can, a cloth, a dead AA battery,
a broken earphone or charger for e-waste, an unused face mask for medical waste. Never use damaged lithium cells or real medical waste.

## 4. Software set-up (about 30 minutes)

1. Install the Arduino IDE. Open `hardware/arduino/sorter_node/sorter_node.ino`, choose the board and port, upload.
2. Serial Monitor at 115200 baud: type `P` (answer `PONG`), `S` (answer `{"m":...,"g":...}`), then `A60`, `A90`, `A120` to see the tray angles.
   Put the best left and right angles into `ANGLE_LEFT` and `ANGLE_RIGHT` in the sketch and upload again. **Close the Serial Monitor afterwards**, because it holds the port.
3. `pip install pyserial` (already installed on this laptop). Find the port name in Device Manager (for example `COM5`).
4. Smoke test with no hardware at all:
   `python scripts/prototype_sorter.py --mock-node --image data/processed/battery/garbage12__battery__e6790782bd.jpg`
   (expected: hazardous bin).
5. Phone camera: install "IP Webcam" (Android) or DroidCam, start the server, and use `--camera http://<phone-ip>:8080/video`
   (or `--camera 1` for DroidCam over USB).

## 5. Mini calibration of the contamination index (about 90 minutes, same sitting as the demo)

The MQ-135 drifts and needs warm-up, so keep it powered for at least 30 minutes (24 hours is better) and calibrate and demonstrate in the same session.

```
python scripts/calibrate_prototype_oci.py collect --port COM5
> air                      (clean air, after warm-up)
> smell                    (spirit swab or vinegar a few cm from the sensor)
> dry                      (blade on a dry clean item)
> wet                      (blade on a soaked item)
> sample porous 0          (cardboard or paper, no residue)   ... repeat
> sample porous 1          (1 teaspoon of curd, ketchup or juice)
> sample rigid 0           (plastic cup or lid)  ...
```

Plan: 2 categories (porous, rigid) × residue 0, 0.5, 1, 2, 4 teaspoons × 4 repeats = 40 samples. An item with at least one teaspoon is labelled contaminated.
Wipe or change the item between samples and let the gas reading return to the baseline. Then:

```
python scripts/calibrate_prototype_oci.py fit      # writes runs/prototype/oci_calibration.json, prints AUC, thresholds, sensitivity
```

The fit chooses the threshold with the lowest false-positive rate at 95% sensitivity (falls back to 90% or 85% if the data cannot reach it).
Expect modest separation with a crude setup; report whatever it gives.

## 6. Run the prototype

```
python scripts/prototype_sorter.py --port COM5 --camera 0
```

Put an item on the tray, press the button (or SPACE in the window). Every decision is written to `runs/prototype/log.csv`
(item, top class, confidence, uncertainty, family set, raw sensors, index, route) and a snapshot is saved in `runs/prototype/snaps/`.

| Outcome | Tray / signal | When |
|---|---|---|
| auto-sorted | green led, tilt left | one family in the set, low uncertainty, not contaminated |
| hazardous | red led, long beep, tilt right | hazardous is the only family in the set and uncertainty is low |
| manual review | yellow led, 3 short beeps | unsure, mixed or empty set, or a hazard that is unsure |
| contamination reject | yellow led, 2 long beeps | index above its threshold on a recyclable |

Thresholds are calibrated at start-up on the validation set of the verified first model (never on the test set).

## 7. Schedule (about 20 working hours over two days)

| When | Hours | Task | Done when |
|---|---|---|---|
| Today (evening) | 2 | buy or borrow parts; install Arduino IDE; run the mock smoke test | mock run prints a hazardous decision for the battery image |
| Day 1 morning | 4 | breadboard wiring, upload sketch, check `P`, `S`, servo angles, LEDs, buzzer, button | all five checks work from the Serial Monitor |
| Day 1 midday | 3 | box, tray, bins, camera stand, lighting | tray tilts both ways and items slide into the bins |
| Day 1 afternoon | 3 | camera plus classifier with real props: try 10 items, keep the ones it handles | at least 7 of 10 props classified in the right family |
| Day 1 evening | 2 | warm the gas sensor, mini calibration, `fit` | `oci_calibration.json` exists |
| Day 2 morning | 3 | full run: 30 trials (10 clean recyclables, 5 hazards, 5 unfamiliar items, 10 with residue); read `log.csv` | results table filled |
| Day 2 afternoon | 3 | fix the weakest part, record a one-minute video, take photos | video and photos saved |

**Hard cut-offs.** If the gas sensor is not usable by Day 1 evening, run with moisture only (the dedicated single-channel model is part of the design and the
sorter switches to it by itself when one sensor reading is out of range). If the servo cannot move the tray, drop the tilt and show the decision on the LEDs and buzzer.

## 8. Acceptance tests and what to report

| Test | Pass |
|---|---|
| Node answers `P` and `S`; servo reaches three angles; LEDs and buzzer work | yes or no |
| 10 recyclable props, plain background | count routed to the left bin |
| 5 hazard props | count routed to the right bin (a hazard must never go left) |
| 5 unfamiliar items (hand, spoon, key) | count sent to review |
| 10 items with residue | count flagged as contaminated, and false alarms on 10 clean items |
| Latency from button to tray movement | seconds |

Report the counts as they are. With this much hardware, results depend on lighting, props and sensor warm-up. The honest wording is
"laboratory bench prototype with real sensors and an actuator, small sample"; a Technology Readiness Level of 4 can be argued if the tests pass, and your guide decides.

## 9. Typical problems

| Symptom | Fix |
|---|---|
| No answer from the node | wrong port, Serial Monitor still open, sketch not uploaded |
| Arduino resets when the servo moves | add the 470 µF capacitor or a separate 5 V supply |
| Gas readings drift or jump | longer warm-up, hood against draughts, keep the room still, repeat `air` |
| Moisture reading barely changes | blade must touch the wet area; capacitive sensors read high when dry and low when wet (either direction works) |
| Everything goes to manual review | the scene looks unlike the training photos: plain white background, light from the side, item fills the green box; raise `--review-budget` to 0.2 for a demo |
| Item will not slide off | smoother tray surface, larger angle, lighter items |

## 10. Safety

Use dead or small cells only, never open or damage a battery, keep liquids away from the Arduino, and keep the mains lamp away from the wet test area.
