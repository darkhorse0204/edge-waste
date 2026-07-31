# Master Work Plan
## Edge-Deployed Waste Classification with Sensor-Augmented Vision

*Granular, execution-ready roadmap. Phases 1–4 are the mandatory MVP path (independently trained vision classifier + moisture/gas sensors fused into a deterministic, logistic-regression-derived Organic Contamination Index). Phases 5–6 are explicit extensions, sequenced after Phase 4 is validated. Hardware baseline: Raspberry Pi 5 (4GB) + ESP32 sensor hub. No Jetson/TensorRT, no metal sensor/load cell, and no cross-attention fusion inside Phases 1–4.*

---

# PHASE 1: Environment Setup & Core Sensor Bring-Up (MVP Foundation)

## Sub-Phase 1.1 — Development & Training Environment Setup

### 1.1.1 Objectives & Technical Scope
Stand up a reproducible software environment for model training (laptop or free-tier cloud GPU — this sub-phase needs zero hardware and can run in parallel with component shipping). Deliverable is a working Python environment that can import every library the project will use, verified with a smoke-test script.

### 1.1.2 System Requirement & Readiness Checklist
- A laptop/desktop with Python 3.10 or 3.11 (Ultralytics and timm both officially support 3.10–3.12; avoid 3.13 until library support catches up)
- ~15 GB free disk (public datasets + model weights + venv)
- GPU optional for this sub-phase but strongly recommended for Sub-Phase 3.1 — if no local GPU, have a Google Colab account ready (free T4 tier is sufficient for YOLO26-n / ConvNeXt-V2-Tiny fine-tuning)
- Git installed
- No hardware dependency for this sub-phase

### 1.1.3 Download Links & Resource Directory
- Python: https://www.python.org/downloads/
- Ultralytics (YOLO26/YOLO11): https://github.com/ultralytics/ultralytics — docs: https://docs.ultralytics.com/models/yolo26
- timm (ConvNeXt-V2 weights): https://github.com/huggingface/pytorch-image-models — model card: `convnextv2_tiny.fcmae_ft_in22k_in1k`
- PyTorch install matrix (pick the correct CUDA/CPU build): https://pytorch.org/get-started/locally/
- scikit-learn: https://scikit-learn.org/stable/install.html
- SHAP: https://github.com/shap/shap
- pytorch-grad-cam: https://github.com/jacobgil/pytorch-grad-cam
- OpenCV-Python: https://pypi.org/project/opencv-python/
- pyserial (ESP32 ↔ Pi communication, used from Sub-Phase 1.2 onward): https://github.com/pyserial/pyserial
- Flower (only imported/tested here, used in Phase 6): https://github.com/adap/flower — docs: https://flower.ai/docs/framework/

### 1.1.4 Step-by-Step Implementation Protocol
1. Install Python 3.11 and confirm: `python3 --version`
2. Create an isolated environment: `python3 -m venv waste-mvp-env && source waste-mvp-env/bin/activate` (Windows: `waste-mvp-env\Scripts\activate`)
3. Upgrade pip: `pip install --upgrade pip`
4. Install PyTorch using the exact command from the PyTorch install matrix for your OS/CUDA version (CPU-only example): `pip install torch torchvision`
5. Install the core stack in one command:
   ```
   pip install ultralytics timm scikit-learn shap grad-cam opencv-python pandas numpy matplotlib pyserial flwr
   ```
6. Freeze the environment for reproducibility (cite this in your report's methodology): `pip freeze > requirements.txt`
7. Initialize a Git repository at the project root: `git init && git add requirements.txt && git commit -m "Initial environment"`
8. Write and run a smoke-test script `env_check.py`:
   ```python
   import torch, timm, sklearn, shap, cv2, serial, flwr
   from ultralytics import YOLO
   print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
   print("timm:", timm.__version__)
   print("All imports OK")
   ```
9. Run: `python env_check.py` — resolve any import errors before proceeding.

### 1.1.5 Expected Deliverables & Verification Criteria
- `requirements.txt` committed to Git
- `env_check.py` runs with no exceptions and prints `All imports OK`
- **Verification:** re-create the environment from `requirements.txt` alone on a second machine (or a fresh venv) and confirm the smoke test still passes — this is your reproducibility check, worth stating explicitly in your report's methodology section.

---

## Sub-Phase 1.2 — ESP32 + Core Sensor Bring-Up (Moisture, MQ-135, DHT11)

### 1.2.1 Objectives & Technical Scope
Wire the moisture sensor, MQ-135 gas sensor, and DHT11 to the ESP32; confirm each produces a stable, sane reading; perform the **mandatory** per-unit MQ-135 R0 clean-air calibration (this cannot be skipped or approximated from a datasheet — see the OCI protocol document, Part 1.2 and Part 8, item 5). Output of this sub-phase is a single serial data stream of `{moisture_raw, gas_raw, temp, humidity}` readable from a PC over USB.

### 1.2.2 System Requirement & Readiness Checklist
**Hardware:**
- ESP32 NodeMCU (30-pin, CP2102 USB-UART)
- Capacitive soil moisture sensor v1.2/v2.0
- MQ-135 gas sensor module (with onboard potentiometer for load-resistor sensitivity trim)
- DHT11 temperature/humidity sensor
- Breadboard + jumper wires (M-M, M-F)
- Micro-USB (or USB-C, depending on board revision) cable

**Software:**
- Arduino IDE 2.x (https://www.arduino.cc/en/software) **or** PlatformIO in VS Code (https://platformio.org/) — either works; Arduino IDE is simpler for a first setup
- ESP32 board support package: https://github.com/espressif/arduino-esp32 (install via Arduino IDE → Boards Manager → search "esp32" → install "esp32 by Espressif Systems")
- DHT sensor library: https://github.com/adafruit/DHT-sensor-library (Arduino Library Manager → "DHT sensor library" by Adafruit) + its dependency "Adafruit Unified Sensor" (installs automatically as a prompt)
- Reference MQ-135 calibration logic (for the Rs/R0 math, not a plug-in library — write your own per the OCI protocol, but this is a good sanity-check reference): https://github.com/GeorgK/MQ135

**Pin assignments (ESP32 30-pin NodeMCU):**
| Signal | ESP32 Pin | Notes |
|---|---|---|
| Moisture sensor AOUT | GPIO34 (ADC1_CH6) | ADC1 channel — remains usable with WiFi active, unlike ADC2 |
| Moisture sensor VCC | 3V3 or VIN (5V) | Check your specific module's rated voltage on silkscreen; 3.3–5.5V modules can use either |
| Moisture sensor GND | GND | — |
| MQ-135 AOUT | GPIO35 (ADC1_CH7) | ADC1 channel |
| MQ-135 VCC | VIN (5V) | MQ-135's internal heater needs 5V — do not power from 3V3 |
| MQ-135 GND | GND | — |
| DHT11 DATA | GPIO4 | Add a 10 kΩ pull-up resistor between DATA and 3V3 if your DHT11 breakout doesn't already include one on-board |
| DHT11 VCC | 3V3 | — |
| DHT11 GND | GND | — |

### 1.2.3 Download Links & Resource Directory
- Arduino IDE: https://www.arduino.cc/en/software
- ESP32 Arduino core (board manager URL to add in Preferences): `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
- Adafruit DHT library: https://github.com/adafruit/DHT-sensor-library
- MQ135 reference calibration math: https://github.com/GeorgK/MQ135
- pyserial (Python-side reading, already installed in 1.1): https://github.com/pyserial/pyserial

### 1.2.4 Step-by-Step Implementation Protocol
1. Wire all three sensors per the pin table above. Double-check MQ-135 is on VIN (5V), not 3V3 — under-powering the heater gives unstable, meaningless readings.
2. Install Arduino IDE, add the ESP32 board manager URL (File → Preferences → Additional Board Manager URLs), then install the ESP32 package via Tools → Board → Boards Manager.
3. Select your board: Tools → Board → ESP32 Arduino → "ESP32 Dev Module" (adjust if your specific board variant differs).
4. Install the Adafruit DHT sensor library via Library Manager (accept the prompt to also install "Adafruit Unified Sensor").
5. Write a bring-up sketch that reads all three sensors and streams CSV over serial every 2 seconds (DHT11's own sensing frequency limit):
   ```cpp
   #include "DHT.h"
   #define DHTPIN 4
   #define DHTTYPE DHT11
   DHT dht(DHTPIN, DHTTYPE);

   const int MOISTURE_PIN = 34;
   const int GAS_PIN = 35;

   void setup() {
     Serial.begin(115200);
     dht.begin();
     analogReadResolution(12); // ESP32 ADC is 12-bit (0-4095)
   }

   void loop() {
     int moisture_raw = analogRead(MOISTURE_PIN);
     int gas_raw = analogRead(GAS_PIN);
     float temp = dht.readTemperature();
     float humidity = dht.readHumidity();
     Serial.print(moisture_raw); Serial.print(",");
     Serial.print(gas_raw); Serial.print(",");
     Serial.print(temp); Serial.print(",");
     Serial.println(humidity);
     delay(2000);
   }
   ```
6. Upload the sketch; open the Arduino Serial Monitor at 115200 baud and confirm you see four comma-separated values updating every 2 seconds, with no `nan` values from the DHT11 (a `nan` usually means a wiring or timing issue — check the pull-up resistor).
7. **MQ-135 burn-in (mandatory, do this in parallel with everything else — start it and forget it for 24–48 hours):** power the MQ-135 continuously (leave the ESP32 running and connected) for 24–48 hours before treating any gas reading as meaningful. This lets the internal heater element and sensing layer reach stable operating characteristics — readings taken before burn-in will not calibrate correctly and should be discarded.
8. **R0 calibration (after burn-in, in genuinely clean outdoor/well-ventilated air, away from cooking, exhaust, or cleaning products):**
   - Record `gas_raw` for at least 50 samples over ~2 minutes in clean air, and average them.
   - Convert to voltage: `V_RL = gas_raw * (3.3 / 4095)` (ESP32 ADC reference is 3.3V, 12-bit resolution).
   - Compute sensor resistance: `Rs = ((Vc - V_RL) / V_RL) * RL`, where `Vc = 5.0` (the MQ-135 circuit supply voltage) and `RL` is your module's onboard load resistor value (commonly 20 kΩ — check your specific module's datasheet/silkscreen, as this varies by manufacturer).
   - Compute `R0 = Rs_clean_air / 3.6` (3.6 is the widely used clean-air Rs/R0 ratio for MQ-135 from the datasheet curve and the reference GeorgK/MQ135 implementation).
   - **Hard-code this R0 value as a constant** in all subsequent code — it is specific to your physical sensor unit and does not need to be recomputed unless you replace the sensor.
9. Write a minimal Python serial reader to confirm PC-side ingestion works:
   ```python
   import serial
   ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)  # Windows: 'COM3' or similar
   for _ in range(10):
       print(ser.readline().decode().strip())
   ```

### 1.2.5 Expected Deliverables & Verification Criteria
- Wired breadboard rig with all three sensors producing live readings on the Serial Monitor
- A documented, hard-coded `R0` constant for your specific MQ-135 unit, with the raw clean-air Rs samples saved (e.g., to a CSV) as evidence for your report's methodology section
- Python script successfully reads the same live stream over USB serial
- **Verification:** disconnect and reconnect the MQ-135 sensor, re-run the clean-air check, and confirm the raw `gas_raw` value returns to within ~5% of its pre-disconnection reading — this confirms your wiring/connections are stable and not contributing noise that would corrupt calibration later.

---

## Sub-Phase 1.3 — Raspberry Pi 5 Edge Device Setup

### 1.3.1 Objectives & Technical Scope
Flash and configure Raspberry Pi OS on the Pi 5, enable headless (no-monitor) access, confirm the camera interface works, and confirm the Pi can read the ESP32's serial stream over USB. This sub-phase produces the actual edge device the final system will run on — but no model or sensor fusion logic runs here yet, purely device bring-up.

### 1.3.2 System Requirement & Readiness Checklist
- Raspberry Pi 5 (4GB)
- Official 27W USB-C power supply (a phone charger will undervoltage the Pi 5 and cause instability/throttling)
- MicroSD card, 32GB+, Class 10/A2, plus a card reader for your PC
- USB webcam (or Raspberry Pi Camera Module 3, if you chose that option)
- A second PC on the same Wi-Fi network for SSH access (no monitor/keyboard needed for the Pi itself)
- The ESP32 from Sub-Phase 1.2, with a spare USB cable to plug into the Pi

### 1.3.3 Download Links & Resource Directory
- Raspberry Pi Imager (flashing tool): https://www.raspberrypi.com/software/
- Raspberry Pi OS documentation: https://www.raspberrypi.com/documentation/computers/getting-started.html
- Picamera2 library docs (only needed if using the official Camera Module rather than a USB webcam): https://github.com/raspberrypi/picamera2
- gpiozero (Pi GPIO library, for later actuation phases — install now to save time): https://gpiozero.readthedocs.io/

### 1.3.4 Step-by-Step Implementation Protocol
1. Download and install Raspberry Pi Imager on your PC.
2. Insert the microSD card into your PC's card reader, open Imager, choose OS: "Raspberry Pi OS (64-bit)".
3. Click the gear/settings icon (OS customization) before writing: set hostname (e.g., `wastebin1`), enable SSH with password authentication (or paste your public SSH key), set username/password, and pre-configure your Wi-Fi SSID/password so the Pi joins your network on first boot without needing a monitor.
4. Write the image to the microSD card and wait for verification to complete.
5. Insert the microSD card into the Pi 5, connect the official 27W USB-C supply last (after camera/USB devices are connected), and power on.
6. From your PC, find the Pi's IP (check your router's connected-devices list, or use `ping wastebin1.local` if mDNS/Bonjour is available), then connect: `ssh <username>@<pi-ip-address>`.
7. Update the system: `sudo apt update && sudo apt full-upgrade -y`
8. Enable the camera interface: `sudo raspi-config` → Interface Options → Camera → Enable (if using the official Camera Module; USB webcams need no special enabling, they appear as `/dev/video0` automatically).
9. Install Python and core packages on the Pi itself (this is a separate environment from your training laptop):
   ```
   sudo apt install -y python3-pip python3-venv python3-opencv
   python3 -m venv ~/waste-edge-env
   source ~/waste-edge-env/bin/activate
   pip install --upgrade pip
   pip install pyserial gpiozero numpy
   ```
10. Test the camera: for a USB webcam, run a short OpenCV test script:
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite('test_capture.jpg', frame)
    cap.release()
    print("Captured:", ret)
    ```
    Copy `test_capture.jpg` back to your PC (`scp <user>@<pi-ip>:~/test_capture.jpg .`) and visually confirm it's a real, in-focus image.
11. Plug the ESP32 (from Sub-Phase 1.2, already burning in) into one of the Pi 5's USB ports. Confirm it enumerates: `ls /dev/ttyUSB*` (or `/dev/ttyACM*` depending on the USB-serial chip). Run the same minimal pyserial read-test from Sub-Phase 1.2, now executed on the Pi instead of your laptop.

### 1.3.5 Expected Deliverables & Verification Criteria
- Pi 5 boots headlessly, reachable over SSH on your local network
- `test_capture.jpg` confirms the camera pipeline works end-to-end (capture → save → visually inspect)
- ESP32 enumerates as a serial device on the Pi and streams live sensor data, readable via the same pyserial pattern validated in Sub-Phase 1.2
- **Verification:** reboot the Pi with the ESP32 already plugged in, and confirm the serial device path (`/dev/ttyUSB0` or similar) is consistent across reboots — if it isn't, note this now and plan to reference devices by a persistent udev rule or by-id path later, rather than a raw `/dev/ttyUSB0` path that can shift if you plug in other USB devices.

---

# PHASE 2: Data Engineering — Vision Dataset & Real Calibration Data

## Sub-Phase 2.1 — Public Vision Dataset Acquisition & Unified Taxonomy Remapping

### 2.1.1 Objectives & Technical Scope
Acquire TrashNet, TACO, TrashBox, and ZeroWaste-f; remap all four onto a single unified label taxonomy (`paper, cardboard, plastic, glass, metal, organic, other`); produce one combined, cleaned image directory ready for Sub-Phase 3.1 training. No custom images are required for this sub-phase — everything here is public data.

### 2.1.2 System Requirement & Readiness Checklist
- Environment from Sub-Phase 1.1 (Python, `pip install`-able packages)
- Additional packages: `pip install pillow pycocotools kaggle` (pycocotools needed to parse TACO/ZeroWaste's COCO-format annotations; kaggle CLI optional, only if you choose to source any dataset mirror via Kaggle)
- ~10–15 GB free disk space (TrashBox and ZeroWaste-f are the largest of the four)
- A GitHub account is not required for downloading, but useful for cloning conveniently

### 2.1.3 Download Links & Resource Directory
- **TrashNet:** repository https://github.com/garythung/trashnet (README links to the `dataset-original.zip` Google Drive mirror, since the raw data exceeds GitHub's size limits)
- **TACO:** repository https://github.com/pedropro/TACO — project site http://tacodataset.org/ — paper https://arxiv.org/abs/2003.06975. Clone, then run its own downloader: `python3 download.py` (fetches images via the URLs in `data/annotations.json` — some source images may 404 over time since TACO hosts image URLs, not the images themselves, in the base repo; the tool skips broken links and reports what it recovered)
- **TrashBox:** repository https://github.com/nikhilvenkatkumsetty/TrashBox (17,785 images, 7 classes: glass, plastic, metal, e-waste, cardboard, paper, medical waste)
- **ZeroWaste-f:** code repository https://github.com/dbash/zerowaste — project page http://ai.bu.edu/zerowaste/ — data archive on Zenodo https://zenodo.org/record/6412647 — paper https://arxiv.org/abs/2106.02740. Note: the Zenodo archive is large (tens of GB across splits); download only the `zerowaste-f` split folder, not the full `zerowaste-aug` semi-supervised augmentation set, which you don't need for this project.

### 2.1.4 Step-by-Step Implementation Protocol
1. Create the working directory structure:
   ```
   mkdir -p data/raw/{trashnet,taco,trashbox,zerowaste} data/unified/{train,val,test}
   ```
2. **TrashNet:** clone the repo, follow its README to fetch `dataset-original.zip` from the linked Google Drive mirror, unzip into `data/raw/trashnet/`. Confirm 2,527 images across 6 class folders.
3. **TACO:** `git clone https://github.com/pedropro/TACO.git && cd TACO && pip install -r requirements.txt && python3 download.py`. This populates images referenced in `data/annotations.json`. Note the console output for any failed/broken URLs — TACO's images are hosted externally (originally Flickr), so a nonzero failure count is normal; proceed with what downloads successfully.
4. **TrashBox:** clone `https://github.com/nikhilvenkatkumsetty/TrashBox.git` — follow the repo's own instructions for locating/extracting the image archive (check the README for the current hosting location, as dataset repos occasionally move large files to an external link after initial release).
5. **ZeroWaste-f:** download the `zerowaste-f` folder specifically from the Zenodo record (https://zenodo.org/record/6412647); the record's file listing separates splits — do not pull the entire multi-split archive unless you also intend to use the semi-supervised augmentation data. Use the password documented in the Zenodo record listing if the archive is encrypted.
6. Write a `taxonomy_map.py` mapping every source-dataset class name to your unified 7-class taxonomy, e.g.:
   ```python
   UNIFIED_CLASSES = ["paper", "cardboard", "plastic", "glass", "metal", "organic", "other"]

   TRASHNET_MAP = {
       "paper": "paper", "cardboard": "cardboard", "plastic": "plastic",
       "glass": "glass", "metal": "metal", "trash": "other"
   }
   TRASHBOX_MAP = {
       "paper": "paper", "cardboard": "cardboard", "plastic": "plastic",
       "glass": "glass", "metal": "metal", "e-waste": "other", "medical waste": "other"
   }
   # TACO and ZeroWaste-f use COCO-style category IDs — map category name strings
   # (loaded via pycocotools) to UNIFIED_CLASSES the same way.
   ```
7. Write a `build_unified_dataset.py` script that: (a) walks each raw dataset's images, (b) looks up each image's remapped unified class, (c) copies/symlinks the image into `data/unified/<split>/<unified_class>/`, and (d) performs an 80/10/10 train/val/test split **stratified per source dataset** (i.e., split TrashNet's images 80/10/10 separately from TrashBox's, then merge — this guarantees your val/test sets contain examples from every source dataset, which lets you later report cross-dataset generalization, not just a pooled-random split that could accidentally put an entire source dataset only in train).
8. Run the script; log final per-class, per-source-dataset image counts to a `dataset_manifest.csv` for your report's dataset-description table.

### 2.1.5 Expected Deliverables & Verification Criteria
- `data/unified/{train,val,test}/<class>/` populated with remapped images from all four sources
- `dataset_manifest.csv` showing per-class and per-source counts
- **Verification:** open 10 random images per unified class and visually confirm the remapping is sane (e.g., no TACO "cigarette" images ended up in "organic" by taxonomy-mapping error) — a manual spot check here catches remapping bugs before they cost you a wasted training run.

---

## Sub-Phase 2.2 — Physical Staged Calibration Data Collection Protocol

### 2.2.1 Objectives & Technical Scope
Collect the real, physically staged moisture/gas/residue-mass calibration dataset that Sub-Phase 3.3's logistic regression will be fit on. This is **not synthetic** — synthetic sensor data cannot substitute for a real per-unit R0-calibrated MQ-135 response or a real capacitive moisture response, since both are specific to your physical hardware and environment (see the OCI protocol document, Part 1 and Part 4).

### 2.2.2 System Requirement & Readiness Checklist
- Fully bring-up-complete rig from Sub-Phase 1.2 (post burn-in, R0 already computed)
- Digital scale, 0.1–1g resolution
- A consistent standardized contaminant (one specific food paste/puree brand+recipe, kept identical across all trials, plus plain water for the liquid-contamination condition)
- Reference items per category: at least 5 genuinely clean/dry paper plates or cardboard pieces (porous/flat category) and 5 genuinely clean/dry plastic containers or bottles (rigid/enclosed category)
- A DHT11 already logging ambient temperature/humidity alongside every trial (from Sub-Phase 1.2)
- A spreadsheet or CSV template ready before you start (do not design your logging schema mid-collection)

### 2.2.3 Download Links & Resource Directory
- No external downloads — this sub-phase is physical data collection. Reference document: your own `OCI-Formula-Design-and-Calibration-Protocol.md`, Part 4, for the exact experimental design (2 categories × 5 severity levels × 8 replicates ≈ 160 samples) and Part 4.3 for the ground-truth labeling method.

### 2.2.4 Step-by-Step Implementation Protocol
1. Build the logging CSV schema before collecting anything, with columns: `trial_id, category, severity_level, residue_mass_g, item_reference_area_or_weight, moisture_raw, gas_raw, ambient_temp_c, ambient_humidity_pct, timestamp`.
2. Establish fixed calibration anchors first (do this once, not per-trial): measure `moisture_raw` on a completely dry reference item (`M_dry`) and on a fully saturated reference item (`M_wet`); measure `gas_raw` in clean air (baseline) and near your most heavily contaminated staged sample (saturated reference). Record all four anchor values in a separate `calibration_anchors.csv`.
3. For each of the 2 categories × 5 severity levels (0g, 2g, 5g, 10g, 20g of standardized contaminant) × 8 replicates:
   - Prepare the item, weigh out the exact residue mass on the digital scale, apply it consistently (same application method/spread each time).
   - Let the item sit for a fixed, consistent dwell time before measuring (e.g., 60 seconds) so gas off-gassing has a consistent window to develop — pick one dwell time and use it for every single trial without exception.
   - Record `moisture_raw` and `gas_raw` from the live serial stream (average over ~10 consecutive readings to reduce single-sample noise), plus the current DHT11 temp/humidity.
   - Log the row to your CSV immediately, not from memory afterward.
4. Randomize the order of severity levels within each session (don't do all "0g" trials first, then all "20g" trials in sequence) to avoid conflating a real severity effect with sensor drift over the session.
5. Clean the sensor rig thoroughly between trials (wipe the moisture sensor probe, ventilate the enclosure to let gas readings return toward baseline) — verify `gas_raw` has returned to within a reasonable band of your clean-air baseline before starting the next trial.
6. If pursuing the optional real-world secondary validation set (Part 4.3 of the OCI protocol): collect ~20–30 naturally contaminated items, apply the same sensor-reading procedure, but instead of a residue-mass ground truth, have 2–3 independent raters score each on the 0–4 ordinal rubric with reference photos, and compute inter-rater agreement (Cohen's/Fleiss' kappa) with `sklearn.metrics` or the `statsmodels` package.

### 2.2.5 Expected Deliverables & Verification Criteria
- `calibration_data.csv` with ~160 rows (staged) + optionally ~20–30 rows (real-world secondary set)
- `calibration_anchors.csv` with the four fixed reference readings
- **Verification:** plot `gas_raw` and `moisture_raw` against `residue_mass_g` (a simple scatter plot) before moving to Sub-Phase 2.3 — you should see a visually plausible upward trend for both. If either sensor shows a flat, noise-dominated relationship with no visible trend at all, stop and re-check wiring/dwell-time/application-consistency before investing further time in the pipeline downstream.

---

## Sub-Phase 2.3 — Two-Track Preprocessing Pipeline

### 2.3.1 Objectives & Technical Scope
Build two independent, non-interacting preprocessing pipelines: (a) an image pipeline feeding the vision classifier (Sub-Phase 3.1), and (b) a tabular feature-engineering pipeline feeding the OCI (Sub-Phases 3.2–3.4). Keeping these independent is a direct consequence of the decision-level (not learned-fusion) architecture — there is no shared preprocessing step between the two.

### 2.3.2 System Requirement & Readiness Checklist
- Completed Sub-Phase 2.1 (`data/unified/`) and Sub-Phase 2.2 (`calibration_data.csv`, `calibration_anchors.csv`)
- `pip install albumentations` for image augmentation (optional but recommended for a small combined dataset)

### 2.3.3 Download Links & Resource Directory
- Albumentations: https://github.com/albumentations-team/albumentations
- No other new external resources — this sub-phase is your own pipeline code, consuming the outputs of 2.1 and 2.2.

### 2.3.4 Step-by-Step Implementation Protocol
**Image track:**
1. Standardize all images to a common resolution matching your chosen vision backbone's expected input (e.g., 224×224 for ConvNeXt-V2-Tiny classification; YOLO26 handles its own internal letterboxing at train time, so leave detection-track images at native resolution and let Ultralytics' data loader handle resizing).
2. Apply augmentation for the classification track: random horizontal flip, mild rotation (±15°), color jitter (small brightness/contrast changes) — via Albumentations or torchvision transforms. Keep augmentation mild; waste items have canonical orientations less often than, say, natural scenes, so aggressive augmentation can hurt more than help here.
3. Save the final image manifest as a CSV/JSON (`image_path, unified_class, source_dataset, split`) — Ultralytics/timm both accept a folder-per-class structure directly, so this manifest is mainly for your own bookkeeping and reproducibility documentation.

**Tabular (OCI) track:**
4. Load `calibration_data.csv` and `calibration_anchors.csv`.
5. Compute normalized features exactly as specified in the OCI protocol document, Part 1 and Part 2:
   ```python
   import pandas as pd

   df = pd.read_csv("calibration_data.csv")
   anchors = pd.read_csv("calibration_anchors.csv").iloc[0]

   df["f_m"] = ((df["moisture_raw"] - anchors["M_dry"]) /
                (anchors["M_wet"] - anchors["M_dry"])).clip(0, 1)

   # Gas: work in Rs/R0 space, not raw ppm (see OCI protocol Part 1.2)
   # Rs already computed from gas_raw via the R0-calibration formula from Sub-Phase 1.2
   df["g_signal"] = -1 * df["Rs_over_R0"].apply(lambda x: __import__("math").log(x))
   L_min, L_max = anchors["L_min"], anchors["L_max"]
   df["f_g"] = ((df["g_signal"] - L_min) / (L_max - L_min)).clip(0, 1)
   ```
6. Compute the ground-truth severity target: `df["severity"] = df["residue_mass_g"] / df["item_reference_area_or_weight"]`, then binarize at a stated, justified threshold for the logistic regression target in Sub-Phase 3.3 (e.g., `df["contaminated"] = (df["severity"] > THRESHOLD).astype(int)`).
7. Save the processed tabular dataset as `oci_features.csv` with columns: `f_m, f_g, severity, contaminated, category, ambient_temp_c, ambient_humidity_pct`.

### 2.3.5 Expected Deliverables & Verification Criteria
- `data/unified/` fully preprocessed and augmentation-ready for Sub-Phase 3.1
- `oci_features.csv` with computed `f_m`, `f_g`, and both continuous and binarized ground truth
- **Verification:** confirm `f_m` and `f_g` are both bounded in [0, 1] with no NaNs (a NaN here usually traces back to a zero or negative value inside a `log()` call — check your Rs/R0 computation) and that binarized `contaminated` labels are not wildly imbalanced (report the class balance — e.g., "38% contaminated / 62% clean" — since this affects your Part 6 threshold-selection method later).

---

# PHASE 3: Model Development — Independent Vision Classifier + Deterministic OCI

## Sub-Phase 3.1 — Vision Backbone Training (YOLO26 + ConvNeXt-V2-Tiny, Public Data Only)

### 3.1.1 Objectives & Technical Scope
Train a two-stage vision pipeline — YOLO26 for item localization on a cluttered/conveyor-style scene, ConvNeXt-V2-Tiny for fine-grained material classification of the localized crop — entirely on the unified public dataset from Sub-Phase 2.1. No custom images are used here; this model must stand alone, independent of the sensor branch, per the decision-level architecture.

### 3.1.2 System Requirement & Readiness Checklist
- Sub-Phase 2.1's `data/unified/` directory, fully remapped
- GPU access (local or Colab) — CPU training is possible but slow for this step
- `pip install ultralytics timm` (already installed in Sub-Phase 1.1)
- ~4–8 GB free GPU memory for batch sizes in the 16–32 range on these two small-to-medium models

### 3.1.3 Download Links & Resource Directory
- YOLO26 documentation and pretrained weights (auto-download on first use): https://docs.ultralytics.com/models/yolo26
- Ultralytics training guide: https://docs.ultralytics.com/modes/train/
- ConvNeXt-V2 via timm model card: https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k
- timm training/fine-tuning documentation: https://huggingface.co/docs/timm/training_script

### 3.1.4 Step-by-Step Implementation Protocol
1. Build a YOLO-format detection dataset YAML pointing at your unified data (if you have bounding boxes from TACO/ZeroWaste-f's COCO annotations, convert them to YOLO `.txt` label format using Ultralytics' built-in `ultralytics.data.converter.convert_coco` utility). TrashNet/TrashBox images without bounding boxes can be used for the classification stage only (step 5+), since they're typically single, centered objects with no localization annotation.
2. Write `waste_detect.yaml`:
   ```yaml
   path: /absolute/path/to/data/unified
   train: train/images
   val: val/images
   names:
     0: paper
     1: cardboard
     2: plastic
     3: glass
     4: metal
     5: organic
     6: other
   ```
3. Fine-tune YOLO26-n on your detection subset:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo26n.pt")  # auto-downloads pretrained COCO weights
   model.train(data="waste_detect.yaml", epochs=100, imgsz=640, batch=16, patience=20)
   ```
4. Validate: `model.val()` — record mAP50 and mAP50-95 for your report's baseline table.
5. For the classification stage, load ConvNeXt-V2-Tiny via timm and fine-tune on the folder-per-class structure:
   ```python
   import timm, torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])
   train_ds = datasets.ImageFolder("data/unified/train", transform=transform)
   val_ds = datasets.ImageFolder("data/unified/val", transform=transform)
   train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_ds, batch_size=32)

   model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=7)
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
   criterion = torch.nn.CrossEntropyLoss()
   # Standard fine-tuning loop: iterate train_loader, backprop, evaluate on val_loader each epoch
   ```
6. Train for enough epochs to see validation accuracy plateau (typically 15–30 epochs for fine-tuning a pretrained backbone on a dataset this size); save the best checkpoint by validation accuracy, not the final epoch.
7. Run inference on the held-out `test` split (never touched during training/validation) and compute final accuracy, per-class precision/recall, and a confusion matrix (`sklearn.metrics.confusion_matrix`) — report this per-source-dataset too (using the `source_dataset` column from your Sub-Phase 2.1 manifest) to demonstrate cross-dataset generalization.

### 3.1.5 Expected Deliverables & Verification Criteria
- Trained YOLO26-n weights (`runs/detect/train/weights/best.pt`) with recorded mAP
- Trained ConvNeXt-V2-Tiny checkpoint with recorded test accuracy, confusion matrix, and per-source-dataset breakdown
- **Verification:** target test accuracy ≥80% (adjust once you see real numbers, but have a stated target); confirm the confusion matrix's largest error cells make physical sense (e.g., confusing "glass" and "plastic" transparent bottles is more forgivable than confusing "metal" and "organic," which would suggest a deeper bug).

---

## Sub-Phase 3.2 — OCI Feature Normalization (Fixed-Reference Calibration Curves)

### 3.2.1 Objectives & Technical Scope
Finalize and validate the `f_m(M)` and `f_g(G)` normalization functions against your real calibration data from Sub-Phase 2.2/2.3, deciding between linear (fixed-reference min-max) and sigmoid normalization for the moisture channel based on actual observed sensor behavior (per the OCI protocol document, Part 1.1).

### 3.2.2 System Requirement & Readiness Checklist
- `oci_features.csv` from Sub-Phase 2.3
- `pip install scipy` for curve fitting (`scipy.optimize.curve_fit`)

### 3.2.3 Download Links & Resource Directory
- No new external resources — reference `OCI-Formula-Design-and-Calibration-Protocol.md`, Part 1, for the exact decision criteria between linear and sigmoid normalization.

### 3.2.4 Step-by-Step Implementation Protocol
1. Plot raw `moisture_raw` against `severity` (or `residue_mass_g`) across all calibration samples.
2. Fit both a linear model and a logistic/sigmoid curve to this relationship using `scipy.optimize.curve_fit`, and compare residuals (sum of squared errors) between the two fits.
3. If the sigmoid fit's residuals are meaningfully lower (a clear visual S-curve, not just marginally better by chance), switch `f_m(M)` to the sigmoid form `1/(1+exp(-k(M - M_mid)))`, fitting `k` and `M_mid` from this same curve-fit step. Otherwise, keep the simpler fixed-reference linear form from Sub-Phase 2.3.
4. Document whichever choice you made, with the residual comparison numbers, directly in your report — this is the "we tested X vs Y and picked X because..." methodology sentence the OCI protocol recommends.
5. Repeat conceptually for the gas channel, though the OCI protocol's Part 1.2 guidance (work in `-log(Rs/R0)` space) already provides a physically justified log-linear transform — verify empirically that your data doesn't show additional curvature beyond what the log transform already captures.

### 3.2.5 Expected Deliverables & Verification Criteria
- A finalized, documented normalization function for both `f_m` and `f_g`, with the linear-vs-sigmoid decision justified by residual comparison
- Updated `oci_features.csv` with final normalized values recomputed if the normalization function changed from Sub-Phase 2.3's draft
- **Verification:** re-plot normalized `f_m`/`f_g` against `severity` — the relationship should look visually monotonic and reasonably smooth, without abrupt discontinuities.

---

## Sub-Phase 3.3 — OCI Weight Derivation via Regularized Logistic Regression

### 3.3.1 Objectives & Technical Scope
Fit the deterministic OCI formula's coefficients empirically: `OCI = σ(β0 + β1·f_m + β2·f_g [+ β3·f_m·f_g])`. This is a one-time offline fit — the output is a small set of fixed numbers hard-coded into the deployed formula, not a runtime model (per the OCI protocol document, Part 2 and Part 3).

### 3.3.2 System Requirement & Readiness Checklist
- Final `oci_features.csv` from Sub-Phase 3.2
- `pip install scikit-learn` (already installed)

### 3.3.3 Download Links & Resource Directory
- scikit-learn LogisticRegression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- No other external resources — reference `OCI-Formula-Design-and-Calibration-Protocol.md`, Parts 2 and 3, for the exact fitting/collinearity-handling procedure.

### 3.3.4 Step-by-Step Implementation Protocol
1. Check moisture-gas collinearity first: `df[["f_m","f_g"]].corr()` — record the Pearson `r`.
2. If `r < 0.7–0.8`: proceed with standard (lightly regularized) logistic regression. If `r > 0.8`: use a stronger L2 penalty (lower `C` value) and/or test the interaction term (step 5).
3. Fit the base two-feature model:
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import train_test_split

   X = df[["f_m", "f_g"]].values
   y = df["contaminated"].values
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

   clf = LogisticRegression(penalty="l2", C=1.0)  # lower C = stronger regularization if collinearity is high
   clf.fit(X_train, y_train)
   beta0 = clf.intercept_[0]
   beta1, beta2 = clf.coef_[0]
   print("beta0:", beta0, "beta1 (moisture):", beta1, "beta2 (gas):", beta2)
   ```
4. Evaluate on the held-out test split: `clf.score(X_test, y_test)`, plus a full classification report (`sklearn.metrics.classification_report`).
5. Test the interaction term by adding a third feature `f_m * f_g` and comparing model fit via a likelihood-ratio test (or simply compare test-set log-loss/AUC with vs. without the interaction term — keep it only if it meaningfully improves the held-out metric, not just training-set fit).
6. **Run the ablation study** (required per the OCI protocol, Part 3.1): fit and evaluate moisture-only (`f_m` alone) and gas-only (`f_g` alone) logistic regressions, and compare their held-out AUC against the combined two-feature model. Report all three AUCs in your results section as direct evidence that fusion improves over either single sensor.
7. **Hard-code the final coefficients** as constants in your deployment code — the deployed OCI function takes only `beta0, beta1, beta2` (and `beta3` if retained) as fixed numbers, with no live scikit-learn dependency needed on the Raspberry Pi at inference time.

### 3.3.5 Expected Deliverables & Verification Criteria
- Final fixed coefficients (`beta0, beta1, beta2`, optionally `beta3`), documented with their fitting procedure and regularization strength used
- Ablation table: moisture-only AUC, gas-only AUC, combined AUC
- Reported Pearson `r` between `f_m` and `f_g`, and which collinearity-handling choice was made
- **Verification:** combined-model AUC should exceed both single-sensor AUCs — if it doesn't, that's a real, reportable finding (not a bug to hide), but double-check your feature computation before concluding fusion doesn't help.

---

## Sub-Phase 3.4 — Threshold Calibration & Sensor-Dropout Fallback Logic

### 3.4.1 Objectives & Technical Scope
Select the operating decision threshold on the fitted OCI score using a safety-leaning method (not naive Youden's index), and implement the missing-sensor fallback logic that lets the system degrade gracefully if the moisture or gas sensor fails (per the OCI protocol document, Part 5 and Part 6).

### 3.4.2 System Requirement & Readiness Checklist
- Fitted OCI model from Sub-Phase 3.3 (`beta0, beta1, beta2`)
- `pip install scikit-learn matplotlib` (already installed)

### 3.4.3 Download Links & Resource Directory
- scikit-learn ROC/PR curve utilities: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html and https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

### 3.4.4 Step-by-Step Implementation Protocol
1. Generate predicted OCI scores on your held-out test split: `oci_scores = clf.predict_proba(X_test)[:, 1]`.
2. Compute the ROC curve: `fpr, tpr, thresholds = roc_curve(y_test, oci_scores)` and the overall AUC (`roc_auc_score`) for reporting.
3. **Sensitivity-constrained threshold selection (primary method):** filter to thresholds where `tpr >= 0.95` (or your chosen minimum sensitivity target), then among those, select the one minimizing `fpr`:
   ```python
   import numpy as np
   valid_idx = np.where(tpr >= 0.95)[0]
   best_idx = valid_idx[np.argmin(fpr[valid_idx])]
   operating_threshold = thresholds[best_idx]
   ```
4. Also compute and plot the precision-recall curve (`precision_recall_curve`) given likely class imbalance, and report the F2-score at your chosen threshold as a secondary check.
5. **Hard-code `operating_threshold`** as a constant alongside the `beta` coefficients from Sub-Phase 3.3.
6. Implement the sensor-dropout fallback in the deployed OCI function — if a sensor read fails (serial timeout, out-of-range value, or a hardware fault flag), drop that term and **renormalize the remaining weight**:
   ```python
   def compute_oci(f_m=None, f_g=None, beta0=..., beta1=..., beta2=...):
       terms, weights = [], []
       if f_m is not None:
           terms.append(beta1 * f_m); weights.append(abs(beta1))
       if f_g is not None:
           terms.append(beta2 * f_g); weights.append(abs(beta2))
       if not terms:
           raise RuntimeError("Both sensors unavailable — route to manual review")
       # Renormalize: scale surviving terms so their combined contribution
       # matches what full-sensor availability would produce on average
       scale = sum(weights) / sum(w for w in weights)  # placeholder for your derived renormalization logic
       linear_combo = beta0 + sum(terms)
       return 1 / (1 + np.exp(-linear_combo))
   ```
   Treat the renormalization scale factor as something to derive empirically (e.g., by comparing full-sensor vs. single-sensor-only OCI distributions on your calibration set) rather than guessing it — document whatever factor you land on and why.
7. Test this fallback path explicitly: run it once with both sensors present, once with only `f_m`, once with only `f_g`, on the same held-out samples, and compare resulting classifications — this is your sensor-dropout robustness evidence for Phase 6's ablation suite, generated early so you're not scrambling for it later.

### 3.4.5 Expected Deliverables & Verification Criteria
- Final `operating_threshold` constant, selected via the sensitivity-constrained method, with the achieved sensitivity/specificity trade-off documented
- ROC-AUC and PR curve plots saved for the report
- A working `compute_oci()` function that handles all three sensor-availability cases (both present, moisture-only, gas-only) without crashing
- **Verification:** deliberately pass malformed/out-of-range sensor values into `compute_oci()` and confirm it fails gracefully (routes to manual review) rather than raising an unhandled exception or silently producing a nonsensical OCI value.

---

# PHASE 4: Edge Integration, Explainability & MVP Validation

## Sub-Phase 4.1 — Edge Inference Deployment on Raspberry Pi 5 (ONNX/TFLite Export)

### 4.1.1 Objectives & Technical Scope
Export the trained YOLO26 and ConvNeXt-V2-Tiny models from Sub-Phase 3.1 into a Raspberry-Pi-friendly runtime format, transfer them to the Pi, and confirm real-time-adjacent inference speed. No TensorRT here — CPU-based ONNX Runtime or TFLite only, per the Pi 5 (non-Jetson) hardware baseline.

### 4.1.2 System Requirement & Readiness Checklist
- Trained model checkpoints from Sub-Phase 3.1
- On the training machine: `pip install onnx onnxruntime`
- On the Raspberry Pi (inside `~/waste-edge-env` from Sub-Phase 1.3): `pip install onnxruntime numpy opencv-python`

### 4.1.3 Download Links & Resource Directory
- Ultralytics export documentation (ONNX/TFLite/other formats): https://docs.ultralytics.com/modes/export/
- ONNX Runtime docs: https://onnxruntime.ai/docs/
- timm model export guidance (standard `torch.onnx.export`, no special timm-specific export tool needed): https://pytorch.org/docs/stable/onnx.html

### 4.1.4 Step-by-Step Implementation Protocol
1. Export YOLO26 to ONNX on your training machine:
   ```python
   from ultralytics import YOLO
   model = YOLO("runs/detect/train/weights/best.pt")
   model.export(format="onnx", imgsz=640, simplify=True)
   ```
2. Export the ConvNeXt-V2-Tiny classifier via standard PyTorch ONNX export:
   ```python
   import torch
   dummy_input = torch.randn(1, 3, 224, 224)
   torch.onnx.export(model, dummy_input, "convnextv2_tiny.onnx",
                      input_names=["input"], output_names=["output"], opset_version=17)
   ```
3. Transfer both `.onnx` files to the Pi: `scp best.onnx convnextv2_tiny.onnx <user>@<pi-ip>:~/models/`
4. On the Pi, write a minimal inference test script confirming both models load and run:
   ```python
   import onnxruntime as ort
   import numpy as np

   sess = ort.InferenceSession("models/convnextv2_tiny.onnx")
   dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
   out = sess.run(None, {"input": dummy})
   print("Output shape:", out[0].shape)
   ```
5. Benchmark real inference latency on the Pi using an actual captured frame (not dummy noise) — run 50 sequential inferences and record mean/median/95th-percentile latency for both the detector and the classifier.
6. If latency is unacceptable for your target throughput (define a target now, e.g., "under 500ms per item end-to-end" for a slow-moving demo), consider: reducing YOLO26 to the `-n` (nano) scale if not already, or applying post-training INT8 quantization via ONNX Runtime's quantization tools before re-benchmarking.

### 4.1.5 Expected Deliverables & Verification Criteria
- `best.onnx` (detector) and `convnextv2_tiny.onnx` (classifier) running natively on the Pi 5 via ONNX Runtime
- Recorded latency benchmarks (mean/median/p95) for both models on real captured frames
- **Verification:** confirm ONNX model outputs match the original PyTorch model's outputs on the same input image within a small numerical tolerance (e.g., `np.allclose(onnx_output, pytorch_output, atol=1e-3)`) — this catches export bugs before they silently degrade accuracy on-device.

---

## Sub-Phase 4.2 — Explainability Integration (Grad-CAM for Vision, Coefficient/SHAP for OCI)

### 4.2.1 Objectives & Technical Scope
Integrate Grad-CAM on the vision classifier and coefficient-based (optionally SHAP-visualized) explanation on the fitted OCI logistic regression, producing a combined "explanation card" per prediction — this is the full dual-XAI system, appropriate for a decision-level architecture with no opaque fusion network to explain.

### 4.2.2 System Requirement & Readiness Checklist
- Trained ConvNeXt-V2-Tiny checkpoint (PyTorch version, not the ONNX export — Grad-CAM needs access to intermediate gradients, so run this on your training machine or a Pi with the PyTorch model loaded, not the quantized ONNX version)
- `pip install grad-cam shap` (already installed in Sub-Phase 1.1)
- Fixed OCI coefficients from Sub-Phase 3.3/3.4

### 4.2.3 Download Links & Resource Directory
- pytorch-grad-cam repository and usage examples: https://github.com/jacobgil/pytorch-grad-cam
- SHAP documentation (LinearExplainer, appropriate for a logistic regression model): https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html

### 4.2.4 Step-by-Step Implementation Protocol
1. Integrate Grad-CAM on the vision branch, targeting the last convolutional stage of ConvNeXt-V2-Tiny:
   ```python
   from pytorch_grad_cam import GradCAM
   from pytorch_grad_cam.utils.image import show_cam_on_image

   target_layers = [model.stages[-1]]  # last ConvNeXt-V2 stage; adjust attribute name to match timm's internal module naming
   cam = GradCAM(model=model, target_layers=target_layers)
   grayscale_cam = cam(input_tensor=input_image)[0]
   visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
   ```
2. Since the OCI is a 2–3 coefficient logistic regression, its explanation does not require SHAP's sampling machinery — the direct contribution of each feature to a given prediction is simply `beta1 * f_m` and `beta2 * f_g`. Compute and display these directly as the primary explanation.
3. Optionally, generate SHAP values purely as a visualization layer (not a new derivation) using `shap.LinearExplainer`, which is exact (not approximate) for linear/logistic models and computationally trivial:
   ```python
   import shap
   explainer = shap.LinearExplainer(clf, X_train)
   shap_values = explainer.shap_values(X_test)
   shap.summary_plot(shap_values, X_test, feature_names=["moisture", "gas"])
   ```
4. Bundle both explanations into a single "explanation card" function that takes one item's full prediction (vision class + confidence, OCI score + per-sensor contribution, Grad-CAM heatmap) and renders a combined image/report — useful for your live demo and directly reusable as report figures.

### 4.2.5 Expected Deliverables & Verification Criteria
- Grad-CAM heatmap generation working on real captured images, visually highlighting plausible regions (e.g., visible residue/soiling areas, not random background)
- Per-sensor OCI contribution values (`beta1*f_m`, `beta2*f_g`) computed and displayed per prediction
- Combined explanation-card renderer producing one output artifact per item
- **Verification:** manually inspect Grad-CAM outputs on 10–15 test images and confirm the highlighted regions are plausible (not, e.g., uniformly highlighting image corners or background, which would indicate the model latched onto a spurious cue rather than the actual waste item).

---

## Sub-Phase 4.3 — End-to-End MVP Integration Test

### 4.3.1 Objectives & Technical Scope
Wire the vision classifier, the OCI formula, and the sensor-reading pipeline together into one running system on the Raspberry Pi: camera captures an item → vision model classifies material → sensors read moisture/gas → OCI computed → final recyclable/non-recyclable decision with confidence and explanation. **No physical actuation yet** — output is a logged/displayed decision, not a moved conveyor flap.

### 4.3.2 System Requirement & Readiness Checklist
- All of Sub-Phases 4.1 and 4.2 completed and verified on the Pi
- ESP32 sensor rig connected via USB, streaming live serial data (from Sub-Phase 1.2/1.3)
- USB webcam connected and verified (from Sub-Phase 1.3)

### 4.3.3 Download Links & Resource Directory
- No new external resources — this sub-phase integrates prior deliverables only.

### 4.3.4 Step-by-Step Implementation Protocol
1. Write the main integration loop on the Pi (`main_mvp_loop.py`):
   ```python
   import cv2, serial, onnxruntime as ort
   import numpy as np

   cap = cv2.VideoCapture(0)
   ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
   detector = ort.InferenceSession("models/best.onnx")
   classifier = ort.InferenceSession("models/convnextv2_tiny.onnx")

   BETA0, BETA1, BETA2 = ...   # hard-coded from Sub-Phase 3.3/3.4
   OPERATING_THRESHOLD = ...   # hard-coded from Sub-Phase 3.4
   M_DRY, M_WET, L_MIN, L_MAX, R0 = ...  # hard-coded calibration anchors

   def capture_and_classify():
       ret, frame = cap.read()
       # preprocess frame -> run detector -> crop -> run classifier
       # returns predicted_class, class_confidence
       ...

   def read_sensors():
       line = ser.readline().decode().strip()
       moisture_raw, gas_raw, temp, humidity = map(float, line.split(","))
       return moisture_raw, gas_raw, temp, humidity

   def run_cycle():
       predicted_class, class_conf = capture_and_classify()
       moisture_raw, gas_raw, temp, humidity = read_sensors()
       f_m = compute_f_m(moisture_raw, M_DRY, M_WET)
       f_g = compute_f_g(gas_raw, R0, L_MIN, L_MAX)
       oci = compute_oci(f_m, f_g, BETA0, BETA1, BETA2)
       decision = "non-recyclable" if oci > OPERATING_THRESHOLD else "recyclable"
       print(f"Class: {predicted_class} ({class_conf:.2f}) | OCI: {oci:.3f} | Decision: {decision}")
       return predicted_class, oci, decision
   ```
2. Run this loop on a small fresh test batch (20–30 items you did **not** use in Sub-Phase 2.2's calibration collection) — one item at a time, in front of the camera and sensor rig.
3. Log every cycle's full output (class, confidence, `f_m`, `f_g`, OCI, decision, and the explanation-card artifact from Sub-Phase 4.2) to a results CSV.
4. Manually verify each of the 20–30 results against ground truth you assign by eye/hand (is this actually contaminated? is the material class actually correct?).
5. Compute end-to-end accuracy: material classification accuracy and OCI-decision accuracy (compare against your manually assigned ground truth), separately.

### 4.3.5 Expected Deliverables & Verification Criteria — Definition of Done for the MVP
- `main_mvp_loop.py` running stably on the Pi, producing a full decision (class + OCI + recyclable/non-recyclable + explanation) per item
- Results CSV from the 20–30-item fresh test batch
- **This is the MVP milestone.** Concrete pass criteria: vision classifier accuracy ≥80% on the fresh batch; OCI decision achieves your Sub-Phase 3.4 sensitivity target (e.g., ≥90–95% recall on genuinely contaminated items) on the fresh batch; the system runs the full cycle (capture → classify → sense → OCI → decision → explanation) without crashing across all 20–30 items.

---

# PHASE 5 (Extension — begin only after Phase 4's Definition of Done is met): Enhanced Prototype

## Sub-Phase 5.1 — Category-Conditional OCI Weights + Optional Metal Sensor/Load Cell Integration

### 5.1.1 Objectives & Technical Scope
Split the OCI's fitted coefficients by vision-detected category (porous/flat vs. rigid/enclosed), per the OCI protocol's Part 5 structural redesign, and optionally close the rigid-container blind spot by adding an inductive proximity metal sensor and/or load cell as auxiliary signals — new sensors, not a change to the fusion architecture (still decision-level).

### 5.1.2 System Requirement & Readiness Checklist
- A completed, validated Phase 4 MVP
- Category-stratified calibration data (your Sub-Phase 2.2 data already includes a `category` column — porous/flat vs. rigid/enclosed — collected for exactly this purpose)
- **If adding metal sensor:** an inductive proximity sensor (e.g., LJ12A3-4-Z/BX, typically 6–36V NPN NO output) — **note the voltage mismatch:** this class of sensor commonly runs at 12–24V, which is not directly safe to wire into an ESP32's 3.3V-logic GPIO. Use either (a) a sensor variant with a built-in 5V/3.3V-compatible open-collector output plus a pull-up resistor to 3.3V, or (b) a simple NPN transistor/optocoupler level-shifting circuit between the sensor's output and the ESP32 GPIO. Do not connect a 12–24V sensor output directly to an ESP32 pin.
- **If adding load cell:** a 5–10kg strain-gauge load cell + HX711 24-bit ADC breakout (2-wire digital interface: DT and SCK — no analog ESP32 ADC channel needed for this sensor)

### 5.1.3 Download Links & Resource Directory
- HX711 Arduino/ESP32 library: https://github.com/bogde/HX711
- No new resource for the category-conditional refit — same scikit-learn tooling as Sub-Phase 3.3, applied per-category subset.

### 5.1.4 Step-by-Step Implementation Protocol
1. Split `oci_features.csv` by `category`, and re-run Sub-Phase 3.3's logistic regression fitting procedure independently on each category's subset, producing `(beta0_porous, beta1_porous, beta2_porous)` and `(beta0_rigid, beta1_rigid, beta2_rigid)`.
2. Update `compute_oci()` to accept a `category` argument (sourced from the vision classifier's predicted class) and select the matching coefficient set before evaluating.
3. Compare the two categories' fitted `beta2` (gas weight) — per the OCI protocol, expect `beta2` to be markedly smaller for the rigid/enclosed category, since external gas sensing has weak access to internally trapped liquid contamination. Report this explicitly as a finding, not a bug.
4. If adding the metal sensor: wire per the safety note above, connect its (level-shifted) digital output to a free ESP32 GPIO (e.g., GPIO25, configured as digital input), and extend the serial CSV stream with a `metal_detected` boolean field.
5. If adding the load cell: wire the HX711's DT/SCK pins to two free ESP32 GPIOs (e.g., GPIO26/GPIO27), install the `HX711` Arduino library, and calibrate the load cell against 1–2 known reference weights before use (per the library's own calibration routine) — extend the serial stream with a `weight_g` field.
6. If either new sensor is added, extend the OCI formula (or add a small parallel rule) using the weight-anomaly signal specifically for the rigid/enclosed category, where it's most needed — this directly targets the blind spot rather than being folded uniformly into both categories' formulas.

### 5.1.5 Expected Deliverables & Verification Criteria
- Two validated, category-specific OCI coefficient sets, with the "why" of any large weight differences documented
- (If pursued) working metal/load-cell sensor readings integrated into the serial stream and `compute_oci()`
- **Verification:** re-run the Sub-Phase 4.3 fresh-batch test split by category and confirm accuracy is equal or better than the single-global-weight-set version from Phase 4 — category-conditioning should not make results worse; if it does, check for overfitting on a too-small per-category subset.

---

## Sub-Phase 5.2 — Conveyor Belt & Actuator Assembly + GPIO Sorting Control Loop

### 5.2.1 Objectives & Technical Scope
Add physical automated sorting: a small conveyor mechanism and a servo/solenoid-actuated diverter that routes each item into the correct physical bin based on `main_mvp_loop.py`'s decision output.

### 5.2.2 System Requirement & Readiness Checklist
- A small DC-motor-driven conveyor belt mechanism (commonly a hobbyist kit, or a simple motorized belt built from a geared DC motor + pulleys)
- A motor driver module (e.g., L298N or similar H-bridge) to safely drive the conveyor motor from GPIO logic-level signals
- 1–2 servo motors (e.g., SG90) for the diverter flap, or a solenoid + relay module if a push/pull mechanism is preferred
- `pip install gpiozero` (already installed in Sub-Phase 1.3)

### 5.2.3 Download Links & Resource Directory
- gpiozero documentation (motor/servo control classes): https://gpiozero.readthedocs.io/en/stable/api_output.html

### 5.2.4 Step-by-Step Implementation Protocol
1. Wire the motor driver's control inputs to Pi 5 GPIO pins (e.g., GPIO17/GPIO27 for direction, GPIO22 as a PWM-capable enable pin for speed control), with the motor's own power supply separate from the Pi's 5V rail (do not power a DC motor directly from the Pi's GPIO/5V — use the motor driver's dedicated motor-supply input).
2. Wire the diverter servo's signal line to a PWM-capable GPIO (e.g., GPIO18), with its own adequately-rated 5V supply (servos can draw current spikes that destabilize the Pi's supply if shared).
3. Using `gpiozero.Servo` or `gpiozero.AngularServo`, write a simple `route_item(decision)` function that moves the diverter to the "recyclable" or "non-recyclable" physical bin position.
4. Integrate `route_item()` into `main_mvp_loop.py`'s cycle: after computing `decision`, call `route_item(decision)`, then briefly pulse the conveyor motor forward to advance the next item into position.
5. Test the full physical loop with a batch of items spanning all decision outcomes, confirming physical routing matches the logged decision every time.

### 5.2.5 Expected Deliverables & Verification Criteria
- Working conveyor + diverter mechanism, driven directly by the MVP's decision output
- **Verification:** run 20+ items through the full physical system and confirm 100% agreement between the logged software decision and the item's actual physical routing destination — any mismatch here is a wiring/timing bug, not a model accuracy issue, and should be resolved before counting this sub-phase complete.

---

## Sub-Phase 5.3 — MC-Dropout Bayesian Uncertainty on the Vision Classifier + Temperature Scaling

### 5.3.1 Objectives & Technical Scope
Add epistemic uncertainty estimation to the vision classifier via Monte Carlo Dropout, plus post-hoc temperature scaling for calibrated confidence values — feeding a "low confidence → route to manual review" fallback path.

### 5.3.2 System Requirement & Readiness Checklist
- Trained ConvNeXt-V2-Tiny checkpoint (PyTorch version) from Sub-Phase 3.1, with dropout layers present in its architecture (confirm timm's ConvNeXt-V2 implementation includes dropout — if not, add a dropout layer before the final classification head and briefly re-fine-tune)

### 5.3.3 Download Links & Resource Directory
- No new external resources — standard PyTorch `model.train()` dropout-active inference pattern; temperature scaling reference implementation: https://github.com/gpleiss/temperature_scaling

### 5.3.4 Step-by-Step Implementation Protocol
1. At inference time, keep dropout layers active (call `model.train()` rather than `model.eval()` — counterintuitive but this is exactly what enables MC Dropout's stochastic forward passes) while disabling gradient computation (`torch.no_grad()`).
2. Run N=20–30 stochastic forward passes on the same input image, collect the softmax outputs, and compute the mean prediction and the variance across passes as the uncertainty estimate.
3. Fit temperature scaling on a held-out validation split (not the test split) using the reference implementation linked above — this rescales softmax confidence to be better calibrated against true accuracy.
4. Define an uncertainty threshold (e.g., variance above some value, or mean confidence below some value) that routes the item to a "manual review" bin instead of forcing recyclable/non-recyclable.
5. Integrate into `main_mvp_loop.py`: replace the single forward pass in `capture_and_classify()` with the N-pass MC-Dropout loop, and add the manual-review branch to `route_item()`.

### 5.3.5 Expected Deliverables & Verification Criteria
- MC-Dropout uncertainty estimate computed per prediction, with a documented and justified routing threshold
- Temperature-scaled confidence values, with a reliability diagram (predicted confidence vs. observed accuracy) as verification
- **Verification:** deliberately test on a handful of genuinely ambiguous/edge-case items (e.g., a heavily occluded or unusual item) and confirm the system flags them to manual review rather than confidently misclassifying them.

---

# PHASE 6 (Stretch — only if substantial time remains): Federated Learning & Research Benchmarking

## Sub-Phase 6.1 — Federated Learning Simulation & Deployment (Flower + FedAvg, Non-IID Partitioning)

### 6.1.1 Objectives & Technical Scope
Simulate and (if hardware allows) physically deploy federated learning across multiple "smart bin" nodes using Flower, with realistically non-IID data partitioning across simulated bins (e.g., one bin skewed toward organic waste, another toward paper/cardboard).

### 6.1.2 System Requirement & Readiness Checklist
- `pip install flwr` (already installed in Sub-Phase 1.1)
- A trained baseline model from Sub-Phase 3.1 to use as the federated learning starting point
- (Optional, for physical deployment) 2+ Raspberry Pi 5 units, each independently set up per Sub-Phase 1.3

### 6.1.3 Download Links & Resource Directory
- Flower framework: https://github.com/adap/flower
- Flower quickstart (PyTorch): https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
- Flower simulation engine docs: https://flower.ai/docs/framework/how-to-run-simulations.html

### 6.1.4 Step-by-Step Implementation Protocol
1. Follow the Flower PyTorch quickstart to wrap your ConvNeXt-V2-Tiny classifier in a `flwr.client.NumPyClient` implementing `get_parameters`, `fit`, and `evaluate`.
2. Partition your combined vision dataset into 3–5 simulated "bin" clients **non-IID by design** — e.g., weight the sampling so Client A's local data is majority organic/food-contaminated items, Client B's is majority paper/cardboard, rather than a random IID split.
3. Run Flower's simulation engine locally (no physical hardware needed for this step) with a `FedAvg` strategy, for a defined number of federated rounds, logging global model accuracy after each round.
4. Compare federated convergence against a centralized training baseline (the Sub-Phase 3.1 result) trained on the pooled data — report the accuracy gap and convergence speed difference, this is your core federated learning result.
5. (Optional, if 2+ physical Pi units are available) repeat the deployment with real Flower server/client processes across the physical network, confirming only model weight updates cross the network (verify no raw image data is transmitted, e.g., by inspecting network traffic with `tcpdump` during a round).

### 6.1.5 Expected Deliverables & Verification Criteria
- Working Flower simulation with non-IID client partitioning and FedAvg aggregation
- Accuracy-vs-round convergence plot, compared against the centralized baseline
- **Verification:** confirm the federated global model's final accuracy approaches (even if it doesn't fully match) the centralized baseline — a large, unexplained gap suggests a partitioning or aggregation bug worth investigating before reporting the result.

---

## Sub-Phase 6.2 — Ablation Suite, Rigorous Evaluation & IEEE/Elsevier Paper-Writing Workflow

### 6.2.1 Objectives & Technical Scope
Assemble the full ablation suite for the paper, and structure the write-up around the honest novelty framing and limitations already established in the OCI protocol document (Parts 7 and 8).

### 6.2.2 System Requirement & Readiness Checklist
- All prior sub-phase deliverables, results tables, and plots
- A chosen paper template (IEEE or Elsevier — check your target venue's author guidelines directly on the publisher's site before finalizing formatting)

### 6.2.3 Download Links & Resource Directory
- IEEE manuscript templates: https://www.ieee.org/conferences/publishing/templates.html
- Elsevier journal author templates (varies by journal — locate via the specific journal's "Guide for Authors" page on sciencedirect.com/journal-titles)

### 6.2.4 Step-by-Step Implementation Protocol
1. Compile the required ablation results: OCI formula vs. category-conditional OCI (Sub-Phase 5.1); sensor-dropout robustness curve (Sub-Phase 3.4, step 7); moisture-only vs. gas-only vs. combined AUC (Sub-Phase 3.3); federated vs. centralized accuracy (Sub-Phase 6.1); if pursued, a cross-attention learned-fusion comparison against the formula-based OCI (an explicit Track B experiment, run only for this comparison, not as the deployed system).
2. Structure the paper's methodology section directly from the OCI protocol document's Parts 1–6 (normalization justification, weight derivation, threshold calibration, category-awareness) — this document *is* your methodology section in draft form.
3. Use the OCI protocol document's Part 7 literature-grounding framing (WQI/LPI composite-index parallel) for the related-work section, and Part 8's honest limitations list directly for the paper's limitations section.
4. Draft results tables and figures per the ablation list in step 1, ensuring every claimed contribution has a corresponding reported number.
5. Circulate a draft to your guide for feedback before final submission formatting.

### 6.2.5 Expected Deliverables & Verification Criteria
- Complete ablation results table covering all listed comparisons
- Full paper draft, formatted to your chosen venue's template
- **Verification:** confirm every claim made in the paper's abstract/introduction is traceable to a specific reported result or ablation elsewhere in the paper — an unsupported claim caught by you before submission is far better than one caught by a reviewer.
