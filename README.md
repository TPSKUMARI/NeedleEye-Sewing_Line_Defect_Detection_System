# NeedleEye — Sewing Line Defect Detection System

**NeedleEye** is a real-time computer vision application for detecting fabric defects on a live sewing production line. It uses a YOLOv8 object detection model with a PyQt5 interface, and supports two capture modes, blur rejection, LED alerting, and a full defect reporting dashboard.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [File Structure](#3-file-structure)
4. [Configuration Reference](#4-configuration-reference)
5. [How It Works — Architecture Overview](#5-how-it-works--architecture-overview)
6. [Threads Explained](#6-threads-explained)
7. [Capture Modes](#7-capture-modes)
8. [Hybrid Mode — Deep Dive](#8-hybrid-mode--deep-dive)
9. [Blur Rejection](#9-blur-rejection)
10. [Inference Preprocessing](#10-inference-preprocessing)
11. [Detection & Tracking Pipeline](#11-detection--tracking-pipeline)
12. [Dashboard & Reporting](#12-dashboard--reporting)
13. [Camera Settings](#13-camera-settings)
14. [LED Alert System](#14-led-alert-system)
15. [Keyboard Shortcuts](#15-keyboard-shortcuts)
16. [HUD Reference](#16-hud-reference)
17. [Debugging Guide](#17-debugging-guide)
18. [Tuning Guide](#18-tuning-guide)
19. [Known Limitations](#19-known-limitations)

---

## 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 10/11 64-bit |
| Python | 3.9 | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB |
| CPU | 4-core | 6-core or more |
| Camera | Any USB webcam | Industrial USB camera (1080p) |
| GPU | Not required | Optional (CUDA speeds up inference) |
| Serial Port | Not required | COM port for LED alerting |

---

## 2. Installation

### Step 1 — Clone or copy the project

Place all files in a single folder, for example `C:\NeedleEye\`.

### Step 2 — Install Python dependencies

Open a terminal in the project folder and run:

```bash
pip install ultralytics opencv-python PyQt5 numpy pyserial
```

If you are using a CUDA GPU, replace `opencv-python` with `opencv-python-headless` and install the matching PyTorch CUDA build from [https://pytorch.org](https://pytorch.org).

### Step 3 — Place your model file

Copy your trained model into the project folder. The application looks for models in this priority order:

```
last_rolling.onnx   ← checked first (ONNX, fastest)
last_rolling.pt     ← fallback (.pt PyTorch)
```

To switch to a different model, edit the `MODEL_PATH` lines at the top of `needleeye.py`:

```python
MODEL_PATH = "last_rolling.onnx" if os.path.exists("last_rolling.onnx") else "last_rolling.pt"
# MODEL_PATH = "best_m.onnx" if os.path.exists("best_m.onnx") else "best_m.pt"
# MODEL_PATH = "yolov8_defect.onnx" if ...
# MODEL_PATH = "best_continueing.onnx" if ...
```

Uncomment the model you want to use and comment out the others.

### Step 4 — Run the application

```bash
python needleeye.py
```

The application will open in fullscreen frameless mode. Press `Escape` to minimise, `Q` to quit.

---

## 3. File Structure

```
NeedleEye/
│
├── needleeye.py              ← Main application (single file)
├── camera_settings.json      ← Auto-created: saved camera slider values
│
├── captures/
│   ├── raw/                  ← Original unmodified frames at capture time
│   ├── annotated/            ← Frames with detection boxes drawn on them
│   └── labels/               ← YOLO-format .txt label files (one per capture)
│
├── last_rolling.onnx         ← Your model file (place here)
├── best_m.onnx               ← Alternative model (optional)
└── README.md                 ← This file
```

The `captures/` folders are created automatically on first run.

---

## 4. Configuration Reference

All configuration is at the top of `needleeye.py`. No separate config file is needed.

### Camera & Device

```python
DEVICE = 0
# 0 = first USB camera. Change to 1, 2 etc. if you have multiple cameras.
```

### Model & Inference

```python
INFERENCE_IMGSZ = 640
# Resolution fed to YOLO. 640 is the standard YOLOv8 input size.
# Do NOT change unless your model was trained at a different size.

CONF_THRESHOLD = 0.25
# Minimum confidence score for a detection to be accepted.
# Lower = more detections (more false positives).
# Higher = fewer detections (may miss real defects).

IOU_THRESHOLD = 0.45
# Intersection-over-Union threshold for Non-Maximum Suppression.
# Removes duplicate overlapping boxes. Raise if too many overlapping boxes appear.
```

### Camera Resolution

```python
CAPTURE_W, CAPTURE_H = 1920, 1080
# CAPTURE_W, CAPTURE_H = 1280, 720  ← commented out alternative
# Resolution at which frames are captured and saved.
# Higher resolution = sharper raw images but more CPU load.
# Note: YOLO still receives frames at INFERENCE_IMGSZ (640), not this resolution.
```

### Capture Cooldown (Live Mode)

```python
CAPTURE_COOLDOWN = 4.0
# Minimum seconds between captures in LIVE mode.
# Prevents saving dozens of identical frames for the same defect.
```

### Tracking

```python
STABLE_FRAMES_REQUIRED = 1
# Number of consecutive frames a detection must appear before it is counted as stable.
# 1 = any detection is immediately counted.
# Raise to 2 or 3 to reduce false positives from single-frame noise.

TRACK_DIST_THRESHOLD = 50
# Maximum pixel distance between frames for a detection to be considered
# the same object. Raise if the camera or fabric moves fast.
```

### Hybrid Mode Thresholds

```python
STILLNESS_DIFF_THRESHOLD = 3
# Mean pixel difference (on a 320x180 grayscale downsample) below which
# the fabric is considered stationary.
# Too low (e.g. 0.5): fabric never registers as still → only fallback captures.
# Too high (e.g. 20): every frame registers as still → captures too often.
# Recommended starting range: 3–8. Watch the HUD diff value to calibrate.

BLUR_THRESHOLD = 40.0
# Laplacian variance below which a frame is considered too blurry to use.
# Lower = more permissive (accepts more frames).
# Higher = stricter (rejects more frames).
# If "SKIPPED: blurry frame" appears constantly, lower this value.

FALLBACK_CAPTURE_INTERVAL = 4.0
# Maximum seconds allowed between captures in hybrid mode.
# Even if the fabric never stops moving, a capture fires every N seconds.
# This is a safety net to ensure no long fabric section is completely missed.
```

### Preprocessing

```python
INFERENCE_PREPROCESS = False
# Set to True when using a model trained with Roboflow preprocessing:
#   Auto-Orient + Stretch 640x640 + Grayscale + CLAHE contrast.
# Set to False for standard colour BGR models.
```

### LED Serial

```python
LED_SERIAL_PORT = "COM3"    # Change to your Arduino/ESP32 COM port
LED_BAUD_RATE   = 115200
LED_ENABLED     = True      # Set False to disable LED without removing code
```

---

## 5. How It Works — Architecture Overview

```
┌─────────────┐     frames      ┌──────────────┐    results    ┌────────────────┐
│ CameraThread│ ─────────────►  │ FrameBuffer  │ ──────────►   │InferenceThread │
│  (USB cam)  │                 │  (latest     │               │  (YOLO model)  │
└─────────────┘                 │   frame)     │               └────────────────┘
                                └──────────────┘                        │
                                        │                               │ store["results"]
                                        │ frame                         │
                                        ▼                               ▼
                                ┌──────────────┐              ┌─────────────────┐
                                │  QTimer 30ms │◄─────────────│   App.update_   │
                                │  (UI thread) │              │   frame()       │
                                └──────────────┘              └─────────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                     Live Mode    Hybrid Mode    Settings Page
                     (always      (smart         (live feed
                      running      capture)       preview)
                      inference)
```

The application runs three concurrent threads:

- **CameraThread** — captures frames from the USB camera continuously
- **InferenceThread** — runs the YOLO model on frames
- **UI Thread (QTimer)** — redraws the display every 30ms and coordinates capture decisions

---

## 6. Threads Explained

### CameraThread

Reads frames from the camera using OpenCV's `VideoCapture`. Each frame is resized to `CAPTURE_W × CAPTURE_H` and optionally cropped/zoomed based on the zoom slider. The latest frame is written to a `FrameBuffer` (a simple lock-protected single-slot buffer). Only the most recent frame is kept — older frames are overwritten, so the display always shows the latest image.

### FrameBuffer

A single shared slot with a threading lock. `put()` overwrites whatever was there. `get()` returns a copy. This design means there is zero queue buildup — inference always works on the freshest available frame.

### InferenceThread

Loads the YOLO model once at startup and runs in a loop. In **live mode** it continuously pulls frames from `FrameBuffer` and runs inference. In **hybrid mode** it is paused (`_pause_event` cleared) and only wakes when a specific frame is pushed to it via `request_frame()`. Results are written to `store["results"]` which the UI thread reads on the next tick.

### UI Thread (QTimer, 30ms)

Calls `update_frame()` every 30ms. This function:

1. Gets the latest camera frame
2. Reads the latest inference results from `store["results"]`
3. Draws bounding boxes on the frame
4. Decides whether to save a capture
5. Updates the display label

---

## 7. Capture Modes

Switch modes using the **mode button** in the header bar (top right area).

### Live Mode (● LIVE MODE — green)

- Inference runs **continuously** on every frame
- Detection boxes appear on the live feed in real time
- A capture (raw + annotated image + label file) is saved when a stable detection appears **and** `CAPTURE_COOLDOWN` seconds have passed since the last capture
- Best for: high-speed lines where defects pass quickly and you want real-time feedback

### Hybrid Mode (◉ TIMED MODE — amber)

- Inference is **paused** between captures to save CPU
- Captures are triggered by the `MotionDetector` (see section 8)
- The display shows the **last annotated capture image** instead of the live feed — so the operator can inspect results
- A HUD overlay shows live telemetry on top of the last capture
- Best for: variable-speed or stop-start lines where fabric quality is more important than speed

---

## 8. Hybrid Mode — Deep Dive

The `MotionDetector` class decides when to fire a capture. It is called once per display tick (~30ms) with the current frame.

### Trigger Logic

```
Every 30ms:
  ├── Compute diff score (mean pixel diff vs previous frame, on 320×180 downsample)
  ├── Compute blur score (Laplacian variance of current frame)
  │
  ├── If diff < STILLNESS_DIFF_THRESHOLD → fabric is STILL
  │     ├── If gate is OPEN (fabric was moving before):
  │     │     ├── If frame is sharp (blur ≥ BLUR_THRESHOLD) → CAPTURE, lock gate
  │     │     └── If frame is blurry → show "SKIPPED: blurry", keep gate OPEN, retry next tick
  │     └── If gate is LOCKED → show "WAITING: must move before next capture"
  │
  ├── If diff ≥ STILLNESS_DIFF_THRESHOLD → fabric is MOVING
  │     └── Unlock gate (so next stop event will fire)
  │
  └── If FALLBACK_CAPTURE_INTERVAL seconds have elapsed since last capture:
        ├── If frame is sharp → CAPTURE, reset timer
        └── If frame is blurry → show "SKIPPED: blurry", do NOT reset timer
                                  (retry on next tick, ~30ms later)
```

### The Gate (`_waiting_for_motion`)

The gate prevents the system from capturing the same still position repeatedly. Once a capture fires on a still fabric, the gate locks. It only unlocks when the fabric moves again (diff rises above `STILLNESS_DIFF_THRESHOLD`). This means:

- **One capture per stop event** — no matter how long the fabric stays still
- The fallback timer fires independently every `FALLBACK_CAPTURE_INTERVAL` seconds regardless of the gate

### Why the fallback exists

If the sewing machine runs continuously without stopping (fast continuous stitching), the fabric never becomes "still" enough to trigger the stillness path. The fallback ensures a capture happens at least every `FALLBACK_CAPTURE_INTERVAL` seconds as a safety net.

---

## 9. Blur Rejection

Before a frame is sent to YOLO inference, its sharpness is checked using the **Laplacian variance** method.

```
blur_score = variance of Laplacian of grayscale frame
```

- A high variance means sharp edges are present → frame is sharp
- A low variance means smooth, blurry image → frame is rejected

The threshold is `BLUR_THRESHOLD = 40.0`. Frames scoring below this are discarded.

### Why blur rejection matters

A blurry frame fed to YOLO will either:
- Miss defects entirely (false negative)
- Produce low-confidence garbage detections (false positive)

Rejecting blurry frames before inference means every detection that does fire is based on a clean, sharp image.

### Blur rejection does NOT block the system

This is a critical design point fixed in the current version:

- If a still frame is blurry, the gate stays **open** — the system retries on the next tick (~30ms) until either a sharp frame arrives or the fabric moves again
- If a fallback frame is blurry, the fallback timer is **not reset** — it remains "due" and retries every tick until a sharp frame arrives
- The system can never get stuck in a "stopped capturing forever" state due to blur

---

## 10. Inference Preprocessing

Controlled by `INFERENCE_PREPROCESS = True/False` at the top of the file.

When `False` (default): frames are sent to YOLO as-is (standard BGR colour).

When `True`: the following pipeline is applied **only to the frame sent to YOLO** — the raw capture saved to disk and the live display are never affected.

### Preprocessing Pipeline

| Step | Operation | Why |
|------|-----------|-----|
| 1 | Auto-Orient | No-op for camera frames (no EXIF data). Kept for parity with Roboflow export. |
| 2 | Stretch to 640×640 | Matches Roboflow "Stretch" resize. Aspect ratio is NOT preserved. |
| 3 | Grayscale → 3-channel BGR | Converts to grayscale, then back to 3-channel so YOLO receives the expected tensor shape. |
| 4 | CLAHE (Adaptive Equalization) | Matches Roboflow "Auto-Adjust Contrast: Using Adaptive Equalization". Uses LAB colour space L-channel, `clipLimit=2.0`, `tileGridSize=(8,8)`. |

Use `INFERENCE_PREPROCESS = True` when your model was trained with the Roboflow preprocessing shown in the training dataset settings. Use `False` for standard colour models.

---

## 11. Detection & Tracking Pipeline

### Detection

YOLO runs on each frame and returns a list of bounding boxes. Each box has:
- Coordinates `(x1, y1, x2, y2)`
- Confidence score
- Class index

Boxes below `CONF_THRESHOLD` are automatically discarded by YOLO before being returned.

### Tracking

The `Tracker` class assigns persistent IDs to detections across frames. For each new detection, it searches existing tracked objects for the nearest centroid within `TRACK_DIST_THRESHOLD` pixels. If found, the detection is assigned that object's ID and its frame count increments. If not found, a new ID is created.

This gives each defect a stable ID number (`ID0`, `ID1`, etc.) that persists as long as it stays within tracking distance.

### Stability Gate

A detection is only logged and saved when its frame count reaches `STABLE_FRAMES_REQUIRED`. At the default value of `1`, every detection is immediately stable. Raising this to `2` or `3` means a defect must appear in multiple consecutive frames before being counted — useful for reducing single-frame noise.

### Bounding Box Colours

- **Green box** — stable detection (frame count ≥ `STABLE_FRAMES_REQUIRED`)
- **Red box** — unstable detection (still accumulating frame count)

### Saving Captures

When a stable detection occurs and the cooldown has passed, three files are saved:

| File | Location | Contents |
|------|----------|---------|
| Raw frame | `captures/raw/YYYYMMDD_HHMMSS.jpg` | Original unmodified camera frame |
| Annotated frame | `captures/annotated/YYYYMMDD_HHMMSS.jpg` | Frame with boxes and labels drawn |
| Label file | `captures/labels/YYYYMMDD_HHMMSS.txt` | YOLO-format: `class cx cy w h` (normalised) |

The annotated frame is always saved. The raw and label files are also saved for training data collection purposes.

---

## 12. Dashboard & Reporting

Navigate to the dashboard using the **REPORT** button in the bottom nav bar.

### Stat Cards

The scrollable card row at the top shows defect counts by class. The first card always shows the total count. Each class gets its own coloured card.

### Last Defect Preview

The right panel shows the annotated image of the most recent defect. Click the image to open it in a fullscreen lightbox view.

### Event Table

The table below lists all captured defects in reverse chronological order (newest first). Columns:

| Column | Content |
|--------|---------|
| Timestamp | Date and time of capture |
| Class | Class name(s) detected, with confidence scores |
| Confidence | Primary detection confidence |
| Annotated | File path to annotated image |
| Raw | File path to raw image |

### Date/Time Filter

Use the **From** and **To** date pickers to filter the table and cards to a specific time range. Use the **Class** dropdown to filter by defect type. Click **Apply** to refresh, **Reset** to return to the default last-hour view.

### Exporting

**⬇ CSV** — exports all filtered entries to a `.csv` file with columns: Timestamp, Class, Confidence, Annotated Path, Raw Path, Label Path.

**⬇ Report** — exports a formatted `.txt` report including a summary by class (count and percentage) and a full event log.

---

## 13. Camera Settings

Click the **⚙** button in the top right to open the camera settings panel. The live feed continues in the left area while settings are adjusted.

### Available Sliders

| Slider | Camera Property | Notes |
|--------|----------------|-------|
| Brightness | `CAP_PROP_BRIGHTNESS` | 0–100 |
| Contrast | `CAP_PROP_CONTRAST` | 0–100 |
| Saturation | `CAP_PROP_SATURATION` | 0–100 |
| Sharpness | `CAP_PROP_SHARPNESS` | 0–100 |
| White Balance | `CAP_PROP_WHITE_BALANCE_BLUE_U` | Auto WB is disabled when set |
| Focus | `CAP_PROP_FOCUS` | Auto focus is disabled when set |
| Pan | `CAP_PROP_PAN` | -100 to 100 |
| Tilt | `CAP_PROP_TILT` | -100 to 100 |
| Zoom | Software zoom (crop + resize) | 0–100, applied in CameraThread |

Settings are saved automatically to `camera_settings.json` every time a slider is moved. They are reloaded on the next launch.

The **↺ Reset Defaults** button restores sliders to the saved default values defined in `CAMERA_SETTINGS` at the top of the code.

**Note:** Not all cameras support all properties. Unsupported properties are silently ignored by OpenCV.

---

## 14. LED Alert System

When a defect is detected and logged, the system sends an alert to a connected Arduino or ESP32 over serial.

Two bytes are sent:
1. The class name as a string followed by `\n`
2. A single byte `0x01`

```python
self._ser.write((class_name + "\n").encode())
self._ser.write(b"\x01")
```

### Configuration

```python
LED_SERIAL_PORT = "COM3"    # Your device's COM port
LED_BAUD_RATE   = 115200    # Must match your Arduino sketch
LED_ENABLED     = True      # Set False to disable completely
```

### Connection Status

The header bar shows a status dot next to "NeedleEye":
- **Teal/green dot + "LED COM3"** — connected and ready
- **Grey dot + "LED offline"** — not connected or port unavailable

The system continues to work normally when LED is offline — only the alert output is missing.

### Disabling LED

Set `LED_ENABLED = False` to disable without removing any code. The LED controller is still instantiated but all send operations are no-ops.

---

## 15. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Escape` | Minimise window |
| `Q` | Close application |

---

## 16. HUD Reference

### Live Mode HUD

Displayed as text overlay on the live feed (top left):

```
LIVE  FPS: 12.4
```

FPS reflects inference speed. On CPU with ONNX this is typically 5–15 FPS. Low FPS in live mode is normal and does not affect hybrid mode, which pauses inference between captures.

### Hybrid Mode HUD

Two rows of telemetry overlaid on the last annotated capture:

**Row 1 (top):**
```
HYBRID MODE  |  Still-first + fallback timer          Total: 14
```

**Row 2:**
```
Fabric: STILL  |  Sharp (142)  |  TRIGGERED: fabric still
```

| Field | Meaning |
|-------|---------|
| `Fabric: STILL` (green) | diff score is below `STILLNESS_DIFF_THRESHOLD` |
| `Fabric: MOVING` (red/blue) | diff score is at or above threshold |
| `Sharp (NNN)` (green) | blur score ≥ `BLUR_THRESHOLD`, frame is usable |
| `Blurry (NNN)` (red) | blur score < `BLUR_THRESHOLD`, frame rejected |
| `TRIGGERED: fabric still` (green) | capture fired because fabric stopped |
| `TRIGGERED: fallback timer` (cyan) | capture fired because interval elapsed |
| `SKIPPED: blurry frame` (red) | trigger was due but frame was too blurry |
| `WAITING: must move before next capture` (amber) | gate is locked, waiting for movement |
| `Waiting...` (dark) | no trigger condition active |

**Progress bar (bottom):**

A cyan bar shows how far through the `FALLBACK_CAPTURE_INTERVAL` the system is. The text shows `Fallback in N.Ns`.

---

## 17. Debugging Guide

### Camera not opening / black screen

1. Check `DEVICE = 0` — try `1`, `2` if you have multiple cameras
2. Make sure no other application is using the camera
3. Try removing `cv2.CAP_DSHOW` from `cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)` — some cameras don't support DirectShow on Windows

### Model not loading

1. Verify the `.onnx` or `.pt` file is in the same folder as `needleeye.py`
2. Check the filename exactly matches the `MODEL_PATH` line
3. ONNX models require `onnxruntime` — install with `pip install onnxruntime`
4. For GPU ONNX: `pip install onnxruntime-gpu`

### Very low FPS in live mode

1. Switch to the `.onnx` model — it is significantly faster than `.pt` on CPU
2. Reduce `INFERENCE_IMGSZ` to `416` or `320` (requires retraining for best results)
3. Lower `CAPTURE_W, CAPTURE_H` to `1280, 720`
4. Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` higher at the top of the file if you have more CPU cores available

### Hybrid mode — "SKIPPED: blurry frame" always showing

The `BLUR_THRESHOLD` is too high for your camera and lighting. Lower it:

```python
BLUR_THRESHOLD = 20.0   # try this
BLUR_THRESHOLD = 10.0   # very permissive — accepts almost everything
BLUR_THRESHOLD = 0.0    # disables blur rejection entirely
```

Watch the blur score number in the HUD. Set the threshold just below your typical sharp-frame score.

### Hybrid mode — fabric never shows as STILL

The `STILLNESS_DIFF_THRESHOLD` is too low for your camera's noise floor. Raise it:

```python
STILLNESS_DIFF_THRESHOLD = 5    # try this
STILLNESS_DIFF_THRESHOLD = 8    # if still not triggering
STILLNESS_DIFF_THRESHOLD = 15   # very permissive
```

Watch the diff score number in the HUD when you manually hold the fabric still. Set the threshold just above that number.

### Hybrid mode — capturing too often even when fabric moves

The `STILLNESS_DIFF_THRESHOLD` is too high. Lower it:

```python
STILLNESS_DIFF_THRESHOLD = 2
```

### Hybrid mode — stopped capturing after a blurry frame

This was a bug in older versions. The current version is fixed — blur rejection on the fallback path no longer resets `_last_trigger`, so the system retries every ~30ms until a sharp frame arrives. If you are seeing this, make sure you are running the latest version.

### Dashboard shows no entries

1. Confirm you are in the correct date range (the From/To filter defaults to the last hour)
2. Defects are only logged when `stable_detections` is non-empty — check `CONF_THRESHOLD` is not too high
3. In hybrid mode, confirm the trigger is firing (watch HUD for "TRIGGERED" messages)

### LED not connecting

1. Check the COM port number in Device Manager
2. Verify `LED_BAUD_RATE` matches your Arduino sketch
3. Try power-cycling the Arduino
4. Check no other application (Arduino IDE Serial Monitor, etc.) has the port open
5. If the port exists but connection fails, the LED status shows "offline" and the rest of the application works normally

### Camera settings not saving

Check that `camera_settings.json` is writable in the project folder. On some systems the working directory may be different from the script location — add `os.chdir(os.path.dirname(__file__))` at the top of the script if needed.

### Detections appear on wrong position / box is offset

This can happen if the camera resolution reported by `cap.read()` does not match `CAPTURE_W, CAPTURE_H`. The frame is always resized to `CAPTURE_W × CAPTURE_H` after reading, so coordinates are always in that space. If boxes look offset, verify the camera actually delivers the expected resolution.

---

## 18. Tuning Guide

### Finding your STILLNESS_DIFF_THRESHOLD

1. Start the application and switch to Hybrid mode
2. Run the sewing machine normally for 30 seconds
3. Watch the diff value in the HUD (`Fabric: MOVING (X.X)`)
4. Note the diff value when fabric is clearly moving (e.g. `8.4`)
5. Manually stop the fabric and note the diff value when still (e.g. `1.2`)
6. Set threshold to roughly halfway between: `STILLNESS_DIFF_THRESHOLD = 4`

### Finding your BLUR_THRESHOLD

1. In hybrid mode, watch the blur score in the HUD (`Sharp (NNN)` or `Blurry (NNN)`)
2. Note the score when the image looks sharp to your eye (e.g. `120`)
3. Note the score when the image looks blurry (e.g. `18`)
4. Set threshold at roughly 60–70% of your sharp value: `BLUR_THRESHOLD = 80`
5. If too many frames are being skipped, lower it toward the blurry value

### Balancing capture rate vs missed defects

| Goal | Adjustment |
|------|-----------|
| Capture more often | Lower `FALLBACK_CAPTURE_INTERVAL`, lower `STILLNESS_DIFF_THRESHOLD` |
| Reduce duplicate captures | Raise `FALLBACK_CAPTURE_INTERVAL`, raise `CAPTURE_COOLDOWN` |
| Reduce false positives | Raise `CONF_THRESHOLD`, raise `STABLE_FRAMES_REQUIRED` |
| Catch more defects | Lower `CONF_THRESHOLD`, lower `STABLE_FRAMES_REQUIRED` |
| Better quality captures | Raise `BLUR_THRESHOLD` |
| Stop skipping frames | Lower `BLUR_THRESHOLD` |

---

## 19. Known Limitations

- **Windows only** as written (uses `cv2.CAP_DSHOW`). For Linux/Mac, change `cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)` to `cv2.VideoCapture(DEVICE)`.
- **Single camera** — only one camera is supported at a time (`DEVICE = 0`).
- **In-memory log only** — the defect log is not persisted to disk automatically. It is lost when the application closes. Use the CSV or Report export before closing.
- **Box coordinates in annotated images** are always in `CAPTURE_W × CAPTURE_H` space. If you change capture resolution between sessions, annotated images from old sessions may have boxes that do not match the saved raw images if you resize them externally.
- **No GPU auto-detection** — to use CUDA, install the CUDA version of PyTorch and ONNX Runtime manually. The application does not configure this automatically.
- **LED protocol is one-way** — there is no acknowledgement from the Arduino. If the serial buffer fills up, writes may silently fail.
