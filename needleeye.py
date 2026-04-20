# ── CPU LIMIT ─────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import sys
import cv2
import time
import csv
import json
import datetime
import threading
import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

cv2.setNumThreads(4)
cv2.setUseOptimized(True)

# ── CONFIG ─────────────────────────────────────────
DEVICE = 0
MODEL_PATH = "last_rolling.onnx" if os.path.exists("last_rolling.onnx") else "last_rolling.pt"
# MODEL_PATH = "best_m.onnx" if os.path.exists("best_m.onnx") else "best_m.pt"
# MODEL_PATH = "yolov8_defect.onnx" if os.path.exists("yolov8_defect.onnx") else "yolov8_defect.pt"
# MODEL_PATH = "best_continueing.onnx" if os.path.exists("best_continueing.onnx") else "best_continueing.pt"


INFERENCE_IMGSZ        = 640
CONF_THRESHOLD         = 0.25
IOU_THRESHOLD          = 0.45
CAPTURE_W, CAPTURE_H   = 1920, 1080
# CAPTURE_W, CAPTURE_H   = 1280, 720
STABLE_FRAMES_REQUIRED = 1
TRACK_DIST_THRESHOLD   = 50
CAPTURE_COOLDOWN       = 4.0

# ── TIMED CAPTURE CONFIG ──────────────────────────
# In motion mode a frame is captured every MOTION_CAPTURE_INTERVAL seconds
# and sent to the inference engine.  No optical flow required.
MOTION_CAPTURE_INTERVAL = 2.0   # seconds between captures in motion mode

# ── HYBRID CAPTURE CONFIG ─────────────────────────
# Stillness detection: if mean pixel diff between frames is below this,
# fabric is considered stationary → triggers an immediate capture.
STILLNESS_DIFF_THRESHOLD = 3    # lower = stricter (more still required)

# Blur rejection: Laplacian variance below this → frame discarded.
BLUR_THRESHOLD           = 40.0  # raise if sharp frames are still being skipped

# Safety-net fallback: even if fabric never fully stills, capture every N seconds.
FALLBACK_CAPTURE_INTERVAL = 4.0  # seconds — acts as the maximum gap between captures

# ── INFERENCE PREPROCESSING CONFIG ───────────────
# Set INFERENCE_PREPROCESS = True when using a model trained with:
#   Auto-Orient, Resize 640x640 (stretch), Grayscale, CLAHE contrast.
# Set False for standard BGR colour models (e.g. best_m).
INFERENCE_PREPROCESS = False   # ← flip to True for the grayscale/CLAHE model

# ── LED / SERIAL CONFIG ────────────────────────────
LED_SERIAL_PORT = "COM3"    # Change to your Arduino/ESP32 port
LED_BAUD_RATE   = 115200
LED_ENABLED     = True      # Set False to disable without removing code

# ── CAMERA PROPERTY DEFAULTS ──────────────────────────
CAMERA_SETTINGS = {
    "brightness":                50,
    "contrast":                  50,
    "saturation":                50,
    "sharpness":                 50,
    "white_balance_temperature": 50,
    "focus_absolute":            30,
    "pan_absolute":               0,
    "tilt_absolute":              0,
    "zoom_absolute":              0,
}

# ── PERSISTENT CAMERA SETTINGS ────────────────────
SETTINGS_FILE = "camera_settings.json"

def load_camera_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            merged = dict(CAMERA_SETTINGS)
            merged.update({k: v for k, v in saved.items() if k in merged})
            return merged
        except Exception:
            pass
    return dict(CAMERA_SETTINGS)

def save_camera_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

CAMERA_SETTINGS = load_camera_settings()

PROP_MAP = {
    "brightness":                (cv2.CAP_PROP_BRIGHTNESS,           None,                     (0,   100)),
    "contrast":                  (cv2.CAP_PROP_CONTRAST,             None,                     (0,   100)),
    "saturation":                (cv2.CAP_PROP_SATURATION,           None,                     (0,   100)),
    "sharpness":                 (cv2.CAP_PROP_SHARPNESS,            None,                     (0,   100)),
    "white_balance_temperature": (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, cv2.CAP_PROP_AUTO_WB,    (0,   100)),
    "focus_absolute":            (cv2.CAP_PROP_FOCUS,                cv2.CAP_PROP_AUTOFOCUS,   (0,   100)),
    "pan_absolute":              (cv2.CAP_PROP_PAN,                  None,                     (-100, 100)),
    "tilt_absolute":             (cv2.CAP_PROP_TILT,                 None,                     (-100, 100)),
    "zoom_absolute":             (None,                              None,                     (0,   100)),
}

PROP_LABELS = {
    "brightness":                "Brightness",
    "contrast":                  "Contrast",
    "saturation":                "Saturation",
    "sharpness":                 "Sharpness",
    "white_balance_temperature": "White Balance",
    "focus_absolute":            "Focus",
    "pan_absolute":              "Pan",
    "tilt_absolute":             "Tilt",
    "zoom_absolute":             "Zoom",
}

# ── STYLE ─────────────────────────────────────────
APP_STYLE = """
QMainWindow { background:#0c0d14; }
QWidget     { background:#0c0d14; }
QLabel      { color:#dde0f0; }
QListWidget { background:#11121c; color:#dde0f0; border:none; }
QScrollArea { border:none; }
QTableWidget {
    background:#11121c; color:#dde0f0;
    gridline-color:#1a1c2e; border:none;
    selection-background-color:#2a2060;
}
QTableWidget QHeaderView::section {
    background:#0c0d14; color:#7c6ff7;
    font-weight:bold; font-size:11px;
    border:none; border-bottom:1px solid #1a1c2e;
    padding:6px;
}
QComboBox {
    background:#11121c; color:#dde0f0;
    border:1px solid #1a1c2e; border-radius:6px;
    padding:4px 10px; font-size:12px;
}
QComboBox::drop-down { border:none; }
QComboBox QAbstractItemView {
    background:#11121c; color:#dde0f0;
    selection-background-color:#2a2060;
}
QDateTimeEdit {
    background:#11121c; color:#dde0f0;
    border:1px solid #1a1c2e; border-radius:6px;
    padding:4px 8px; font-size:12px;
}
QDateTimeEdit::up-button, QDateTimeEdit::down-button { width:0; }
QSlider::groove:horizontal {
    height:4px; background:#1a1c2e; border-radius:2px;
}
QSlider::handle:horizontal {
    width:14px; height:14px; margin:-5px 0;
    background:#7c6ff7; border-radius:7px;
}
QSlider::sub-page:horizontal { background:#7c6ff7; border-radius:2px; }
"""

NAV_ACTIVE = """
QPushButton {
    background:#7c6ff7; color:white;
    border:none; border-radius:8px; font-weight:bold;
}
QPushButton:hover { background:#9d93ff; }
"""
NAV_IDLE = """
QPushButton {
    background:#11121c; color:#454870;
    border:1px solid #1a1c2e; border-radius:8px;
}
QPushButton:hover { background:#1a1c2e; color:#7c6ff7; }
"""
SETTINGS_BTN = """
QPushButton {
    background:#11121c; color:#7c6ff7;
    border:1px solid #1a1c2e; border-radius:8px;
    font-size:18px;
}
QPushButton:hover { background:#1a1c2e; }
"""
CARD_STYLE = """
QFrame {
    background:#11121c;
    border:1px solid #1a1c2e;
    border-radius:10px;
}
"""
EXPORT_BTN = """
QPushButton {
    background:#11121c; color:#00ffcc;
    border:1px solid #00ffcc; border-radius:8px;
    font-size:12px; font-weight:bold; padding:6px 14px;
}
QPushButton:hover { background:#00ffcc; color:#0c0d14; }
"""
FILTER_BTN = """
QPushButton {
    background:#7c6ff7; color:white;
    border:none; border-radius:6px;
    font-size:12px; padding:5px 14px;
}
QPushButton:hover { background:#9d93ff; }
"""

# ── DEFECT LOG ─────────────────────────────────────
class DefectLog:
    def __init__(self):
        self._entries = []
        self._lock    = threading.Lock()

    def add(self, entry):
        with self._lock:
            self._entries.append(entry)

    def get_all(self):
        with self._lock:
            return list(self._entries)

    def get_filtered(self, from_dt=None, to_dt=None, class_filter=None):
        with self._lock:
            out = []
            for e in self._entries:
                if from_dt and e["ts"] < from_dt:
                    continue
                if to_dt and e["ts"] > to_dt:
                    continue
                if class_filter and class_filter != "All" and e["class_name"] != class_filter:
                    continue
                out.append(e)
            return out

    def class_counts(self, entries=None):
        if entries is None:
            entries = self.get_all()
        counts = {}
        for e in entries:
            counts[e["class_name"]] = counts.get(e["class_name"], 0) + 1
        return counts

    def all_class_names(self):
        return sorted({e["class_name"] for e in self.get_all()})


# ── BLUR & STILLNESS HELPERS ──────────────────────
def is_blurry(frame, threshold=BLUR_THRESHOLD):
    """Returns True if the frame is too blurry for reliable inference.

    Uses Laplacian variance — lower variance = more blur.
    Threshold tuning:
      - Raise BLUR_THRESHOLD if sharp frames are wrongly rejected.
      - Lower it if blurry frames are getting through.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_fabric_still(prev_frame, curr_frame, threshold=STILLNESS_DIFF_THRESHOLD):
    """Returns True when fabric movement between two consecutive frames is negligible.

    Computes mean absolute pixel difference on a downscaled grayscale pair
    to keep CPU cost low.
    Threshold tuning:
      - Lower STILLNESS_DIFF_THRESHOLD for stricter stillness (less false triggers).
      - Raise it if the fabric never registers as still.
    """
    if prev_frame is None or curr_frame is None:
        return False
    # Downscale for speed — we only need a rough motion estimate
    small_prev = cv2.resize(prev_frame, (320, 180))
    small_curr = cv2.resize(curr_frame, (320, 180))
    gray_prev  = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    gray_curr  = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
    diff       = cv2.absdiff(gray_prev, gray_curr)
    return float(diff.mean()) < threshold


# ── INFERENCE PREPROCESSING ──────────────────────
def preprocess_for_inference(frame):
    """Applies the preprocessing pipeline used during model training.

    Pipeline (matches Roboflow export settings):
      1. Auto-Orient  — no-op for camera frames (no EXIF), kept for parity.
      2. Stretch-resize to 640×640 — matches training resize exactly.
         Note: aspect ratio is NOT preserved (stretch, not letterbox).
      3. Grayscale — convert BGR → gray, then back to BGR so YOLO still
         receives a 3-channel tensor as expected.
      4. CLAHE (Adaptive Histogram Equalization) — matches
         "Auto-Adjust Contrast: Using Adaptive Equalization" in Roboflow.
         clipLimit=2.0, tileGridSize=(8,8) are standard CLAHE defaults.

    Only applied when INFERENCE_PREPROCESS = True.
    The original frame passed in is never modified — a copy is always returned.
    """
    img = frame.copy()

    # Step 1 — Auto-Orient: camera frames carry no EXIF, nothing to do.

    # Step 2 — Stretch to 640×640 (no letterboxing, matches training)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Step 3 — Grayscale → back to 3-channel BGR for YOLO input
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Step 4 — CLAHE adaptive contrast equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # ch_b, ch_g, ch_r = cv2.split(img)
    # img = cv2.merge([
    #     clahe.apply(ch_b),
    #     clahe.apply(ch_g),
    #     clahe.apply(ch_r),
    # ])
    lab          = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img          = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


# ── TIMED MOTION DETECTOR ─────────────────────────
class MotionDetector:
    """Hybrid capture trigger for sewing inspection.

    Fires a capture when EITHER condition is met:
      1. Fabric transitions from moving → still (once per stop event).
      2. Fallback timer expires (FALLBACK_CAPTURE_INTERVAL) — safety net so
         no long fabric stretch is ever missed even if stillness is never reached.

    ── Blur rejection rules (the fix) ──────────────────────────────────────
    Previous bug: blur rejection on the fallback path was calling
      self._last_trigger = now
    which reset the timer, so the next fallback would not fire for another
    full FALLBACK_CAPTURE_INTERVAL — effectively making the system stop
    capturing after the first blurry frame.

    Fixed behaviour:
      • Still + blurry  → gate stays OPEN, retry every frame until sharp.
                          _waiting_for_motion is NOT set (that would lock forever).
      • Fallback + blurry → skip this tick but do NOT reset _last_trigger.
                            The timer keeps counting so the fallback fires again
                            on the very next display tick (~30 ms later) until
                            a sharp frame comes through.

    Config knobs (top of file):
      STILLNESS_DIFF_THRESHOLD  — how still is "still"
      BLUR_THRESHOLD            — minimum sharpness to accept a frame
      FALLBACK_CAPTURE_INTERVAL — maximum seconds between captures
    """

    def __init__(self):
        self._last_trigger       = 0.0
        self._prev_frame         = None
        self._lock               = threading.Lock()

        # Gate flag — set to True ONLY after a real successful capture on a
        # still frame.  Blur rejection must never set this flag.
        self._waiting_for_motion = False

        # HUD telemetry — updated each call, read by _update_motion_mode
        self.last_diff      = 0.0
        self.last_blur      = 0.0
        self.trigger_reason = ""   # "still" | "fallback" | "blur_rejected" | "waiting" | ""

    def update(self, frame):
        """Returns True when a capture should be attempted.

        Call once per display frame (~30 ms) with the current camera frame.

        Decision table:
          still=T, gate=open,   sharp  → capture, lock gate        → return True
          still=T, gate=open,   blurry → skip, gate stays open     → return False
          still=T, gate=locked         → skip, show "waiting"      → return False
          still=F                      → unlock gate               → return False
          fallback due,         sharp  → capture, reset timer      → return True
          fallback due,         blurry → skip, timer NOT reset     → return False
        """
        with self._lock:
            now          = time.time()
            elapsed      = now - self._last_trigger
            fallback_due = elapsed >= FALLBACK_CAPTURE_INTERVAL

            # Compute stillness score
            still = is_fabric_still(self._prev_frame, frame)

            # Compute blur score for HUD (always, not just on trigger)
            gray           = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.last_blur = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Compute diff score for HUD
            if self._prev_frame is not None:
                small_p        = cv2.resize(self._prev_frame, (320, 180))
                small_c        = cv2.resize(frame,            (320, 180))
                gp             = cv2.cvtColor(small_p, cv2.COLOR_BGR2GRAY)
                gc             = cv2.cvtColor(small_c, cv2.COLOR_BGR2GRAY)
                self.last_diff = float(cv2.absdiff(gp, gc).mean())
            else:
                self.last_diff = 999.0

            self._prev_frame = frame.copy()

            # Fabric moving → unlock gate so the next stop event can fire
            if not still:
                self._waiting_for_motion = False

            # ── Stillness trigger (once per stop event) ──────────────────
            if still and not self._waiting_for_motion:
                if self.last_blur < BLUR_THRESHOLD:
                    # Blurry while still — keep gate OPEN so we retry the
                    # next frame.  Do NOT set _waiting_for_motion here.
                    self.trigger_reason = "blur_rejected"
                    return False

                # Sharp still frame — capture and lock gate
                self._waiting_for_motion = True
                self._last_trigger       = now
                self.trigger_reason      = "still"
                return True

            # ── Fallback timer (safety net for continuous motion) ─────────
            if fallback_due:
                if self.last_blur < BLUR_THRESHOLD:
                    # Blurry on fallback — do NOT reset _last_trigger.
                    # fallback_due will remain True next tick so we keep
                    # trying every ~30 ms until a sharp frame arrives.
                    self.trigger_reason = "blur_rejected"
                    return False

                # Sharp fallback frame — capture and reset timer
                self._last_trigger  = now
                self.trigger_reason = "fallback"
                return True

            # ── Waiting for fabric to move before next still trigger ──────
            if self._waiting_for_motion:
                self.trigger_reason = "waiting"
            else:
                self.trigger_reason = ""
            return False

    def reset(self):
        with self._lock:
            self._last_trigger       = 0.0
            self._prev_frame         = None
            self._waiting_for_motion = False
            self.trigger_reason      = ""

    # ── elapsed / remaining helpers used by the HUD ──
    def elapsed(self):
        with self._lock:
            return time.time() - self._last_trigger

    def remaining(self):
        return max(0.0, FALLBACK_CAPTURE_INTERVAL - self.elapsed())

    def progress(self):
        return min(self.elapsed() / FALLBACK_CAPTURE_INTERVAL, 1.0)


# ── CLICKABLE IMAGE LABEL ──────────────────────────
class ClickableImageLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)


# ── LED CONTROLLER ────────────────────────────────
class LEDController:
    def __init__(self):
        self._ser  = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        if not LED_ENABLED:
            return
        try:
            import serial as pyserial
            self._ser = pyserial.Serial(LED_SERIAL_PORT, LED_BAUD_RATE, timeout=1)
            time.sleep(2)
            print(f"[LED] Connected on {LED_SERIAL_PORT}")
        except Exception as ex:
            self._ser = None
            print(f"[LED] Not connected ({ex}) — LED alerts disabled.")

    def trigger(self, class_name: str = "Defect"):
        if self._ser is None or not LED_ENABLED:
            return
        with self._lock:
            try:
                self._ser.write((class_name + "\n").encode())
                self._ser.write(b"\x01")
            except Exception as ex:
                print(f"[LED] Send error: {ex}")
                self._ser = None

    def close(self):
        with self._lock:
            if self._ser:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    @property
    def connected(self):
        return self._ser is not None and self._ser.is_open


# ── BUFFER ─────────────────────────────────────────
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.lock  = threading.Lock()

    def put(self, f):
        with self.lock:
            self.frame = f

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ── CAMERA THREAD ───────────────────────────────────
class CameraThread(QThread):
    def __init__(self, buf):
        super().__init__()
        self.buf        = buf
        self.running    = True
        self.cap        = None
        self.lock       = threading.Lock()
        self.zoom_level = 0

    def run(self):
        self.cap = cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)
        self._apply_defaults()

        while self.running:
            with self.lock:
                ret, frame = self.cap.read()
                zoom       = self.zoom_level

            if not ret:
                continue

            frame = cv2.resize(frame, (CAPTURE_W, CAPTURE_H))

            if zoom > 0:
                factor = 1.0 + (zoom / 100.0) * 3.0
                h, w   = frame.shape[:2]
                ch     = int(h / factor)
                cw     = int(w / factor)
                y0     = (h - ch) // 2
                x0     = (w - cw) // 2
                frame  = cv2.resize(
                    frame[y0: y0 + ch, x0: x0 + cw],
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

            self.buf.put(frame)

        self.cap.release()

    def _apply_defaults(self):
        for name, value in CAMERA_SETTINGS.items():
            if value is None or name not in PROP_MAP:
                continue
            if name == "zoom_absolute":
                self.zoom_level = int(value)
                continue
            prop_id, auto_id, _ = PROP_MAP[name]
            if prop_id is None:
                continue
            if auto_id is not None:
                self.cap.set(auto_id, 0)
            self.cap.set(prop_id, float(value))

    def set_prop(self, prop_id, value, auto_id=None):
        with self.lock:
            if prop_id is None:
                self.zoom_level = int(value)
                return
            if self.cap:
                if auto_id is not None:
                    self.cap.set(auto_id, 0)
                self.cap.set(prop_id, float(value))

    def stop(self):
        self.running = False
        self.wait()


# ── INFERENCE ──────────────────────────────────────
class InferenceThread(QThread):
    fps_updated = pyqtSignal(float)

    def __init__(self, buf, store):
        super().__init__()
        self.buf              = buf
        self.store            = store
        self.running          = True
        self._requested_frame = None
        self._model           = None
        self._paused          = False
        self._pause_event     = threading.Event()
        self._pause_event.set()

    def run(self):
        model       = YOLO(MODEL_PATH)
        prev        = time.time()
        self._model = model

        while self.running:
            self._pause_event.wait()

            req = self._requested_frame
            if req is not None:
                self._requested_frame = None
                frame = req
            elif self._paused:
                time.sleep(0.005)
                continue
            else:
                frame = self.buf.get()

            if frame is None:
                time.sleep(0.001)
                continue

            # Apply training-matched preprocessing if enabled
            infer_frame = preprocess_for_inference(frame) if INFERENCE_PREPROCESS else frame

            results = model(
                infer_frame,
                imgsz=INFERENCE_IMGSZ,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )

            self.store["results"] = results[0]

            now = time.time()
            self.fps_updated.emit(1 / (now - prev))
            prev = now

    def pause(self):
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        self._paused = False
        self._pause_event.set()

    def request_frame(self, frame):
        self._requested_frame = frame
        self._pause_event.set()

    def stop(self):
        self._pause_event.set()
        self.running = False
        self.wait()


# ── TRACKER ────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.objects  = {}
        self.next_id  = 0

    def update(self, boxes):
        updated = {}
        results = []

        for (x1, y1, x2, y2, conf, cls) in boxes:
            cx, cy    = (x1 + x2) // 2, (y1 + y2) // 2
            best_id   = None
            best_dist = 1e9

            for oid, (px, py, count) in self.objects.items():
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_dist and d < TRACK_DIST_THRESHOLD:
                    best_id   = oid
                    best_dist = d

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                updated[best_id] = (cx, cy, 1)
            else:
                _, _, c = self.objects[best_id]
                updated[best_id] = (cx, cy, c + 1)

            results.append((x1, y1, x2, y2, conf, cls, best_id, updated[best_id][2]))

        self.objects = updated
        return results


# ── CAMERA CONTROL SLIDER ────────────────────────────
class ControlSlider(QWidget):
    def __init__(self, prop_name, cam):
        super().__init__()
        self.prop_name = prop_name
        self.cam       = cam

        prop_id, auto_id, (lo, hi) = PROP_MAP[prop_name]
        self.prop_id  = prop_id
        self.auto_id  = auto_id

        default = CAMERA_SETTINGS.get(prop_name, (lo + hi) // 2)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)

        label = QLabel(PROP_LABELS.get(prop_name, prop_name))
        label.setStyleSheet("QLabel { color:#aab0d0; font-size:12px; }")
        label.setFixedWidth(110)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(int(default) if default is not None else (lo + hi) // 2)

        self.val_label = QLabel(str(self.slider.value()))
        self.val_label.setStyleSheet("QLabel { color:#7c6ff7; font-size:11px; }")
        self.val_label.setFixedWidth(36)
        self.val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(label)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_label)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, v):
        self.val_label.setText(str(v))
        self.cam.set_prop(self.prop_id, v, self.auto_id)
        CAMERA_SETTINGS[self.prop_name] = v
        save_camera_settings(CAMERA_SETTINGS)


# ── MAIN APP ──────────────────────────────────────
class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowTitle("NeedleEye")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(screen.width(), screen.height())
        self.move(screen.topLeft())

        self.buf        = FrameBuffer()
        self.store      = {"results": None}
        self.tracker    = Tracker()
        self.defect_log = DefectLog()

        self.cam = CameraThread(self.buf)
        self.inf = InferenceThread(self.buf, self.store)
        self.inf.fps_updated.connect(self.set_fps)
        self.led = LEDController()

        self.fps          = 0
        self.last_capture = 0
        self.total_count  = 0

        # ── Mode: "live" or "motion" ──
        self.mode               = "live"
        self.motion_detector    = MotionDetector()
        self._obj_pending_frame = None

        os.makedirs("captures/raw",       exist_ok=True)
        os.makedirs("captures/annotated", exist_ok=True)
        os.makedirs("captures/labels",    exist_ok=True)

        # ── STACK ──────────────────────────────────
        self.stack = QStackedWidget()

        # ── PAGE 0 — LIVE ──────────────────────────
        self.live_label = QLabel()
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.addWidget(self.live_label)

        # ── PAGE 1 — DASHBOARD ─────────────────────
        self.stack.addWidget(self._build_dashboard())

        # ── PAGE 2 — SETTINGS ──────────────────────
        self.settings_page = QWidget()
        s_outer = QHBoxLayout(self.settings_page)
        s_outer.setContentsMargins(0, 0, 0, 0)
        s_outer.setSpacing(0)

        self.settings_feed = QLabel()
        self.settings_feed.setAlignment(Qt.AlignCenter)
        self.settings_feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.settings_feed.setStyleSheet("QLabel { background:#06070d; }")
        s_outer.addWidget(self.settings_feed, stretch=3)

        panel = QWidget()
        panel.setFixedWidth(320)
        panel.setStyleSheet("QWidget { background:#11121c; }")
        p_l = QVBoxLayout(panel)
        p_l.setContentsMargins(20, 20, 20, 20)
        p_l.setSpacing(6)

        title = QLabel("Camera Settings")
        title.setStyleSheet("QLabel { color:#7c6ff7; font-size:15px; font-weight:bold; background:transparent; }")
        p_l.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#1a1c2e;")
        p_l.addWidget(sep)
        p_l.addSpacing(6)

        for prop_name in CAMERA_SETTINGS:
            if prop_name in PROP_MAP:
                p_l.addWidget(ControlSlider(prop_name, self.cam))

        p_l.addStretch()

        reset_btn = QPushButton("↺  Reset Defaults")
        reset_btn.setFixedHeight(36)
        reset_btn.setStyleSheet("""
            QPushButton {
                background:#1a1c2e; color:#7c6ff7;
                border:1px solid #7c6ff7; border-radius:8px;
                font-size:12px;
            }
            QPushButton:hover { background:#7c6ff7; color:white; }
        """)
        reset_btn.clicked.connect(self._reset_camera_defaults)
        p_l.addWidget(reset_btn)

        s_outer.addWidget(panel, stretch=0)
        self.stack.addWidget(self.settings_page)

        # ── HEADER BAR ──────────────────────────────
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet("QWidget{background:#09090f;border-bottom:1px solid #1a1c2e;}")
        hdr_l = QHBoxLayout(header)
        hdr_l.setContentsMargins(16, 0, 16, 0)

        app_title = QLabel("NeedleEye")
        app_title.setStyleSheet("QLabel{color:#7c6ff7;font-size:13px;font-weight:bold;letter-spacing:2px;}")
        hdr_l.addWidget(app_title)
        hdr_l.addSpacing(16)

        self.led_status = QLabel()
        self.led_status.setFixedSize(10, 10)
        hdr_l.addWidget(self.led_status)
        self.led_status_label = QLabel()
        hdr_l.addWidget(self.led_status_label)
        self._update_led_status()

        hdr_l.addStretch()

        self.btn_mode = QPushButton("● LIVE MODE")
        self.btn_mode.setFixedSize(130, 30)
        self.btn_mode.setStyleSheet("""
            QPushButton{background:#1a2a1a;color:#00ff88;
                border:1px solid #00ff88;border-radius:6px;font-size:11px;font-weight:bold;}
            QPushButton:hover{background:#00ff88;color:#0c0d14;}
        """)
        self.btn_mode.clicked.connect(self._toggle_mode)
        hdr_l.addWidget(self.btn_mode)
        hdr_l.addSpacing(10)

        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedSize(34, 34)
        self.btn_settings.setStyleSheet(SETTINGS_BTN)
        self.btn_settings.clicked.connect(self.toggle_settings)
        hdr_l.addWidget(self.btn_settings)

        # ── NAV BAR ────────────────────────────────
        nav = QWidget()
        nav.setFixedHeight(56)
        nav.setStyleSheet("QWidget{background:#09090f;border-top:1px solid #1a1c2e;}")
        nav_l = QHBoxLayout(nav)
        nav_l.setContentsMargins(0, 0, 0, 0)

        self.btn_live   = QPushButton("LIVE")
        self.btn_report = QPushButton("REPORT")
        self.btn_live.setFixedSize(120, 38)
        self.btn_report.setFixedSize(120, 38)
        self.btn_live.setStyleSheet(NAV_ACTIVE)
        self.btn_report.setStyleSheet(NAV_IDLE)

        self.btn_live.clicked.connect(lambda: self._go(0))
        self.btn_report.clicked.connect(lambda: self._go(1))

        nav_l.addStretch()
        nav_l.addWidget(self.btn_live)
        nav_l.addSpacing(20)
        nav_l.addWidget(self.btn_report)
        nav_l.addStretch()

        # ── MAIN LAYOUT ────────────────────────────
        main_l = QVBoxLayout()
        main_l.setContentsMargins(0, 0, 0, 0)
        main_l.setSpacing(0)
        main_l.addWidget(header)
        main_l.addWidget(self.stack)
        main_l.addWidget(nav)

        w = QWidget()
        w.setLayout(main_l)
        self.setCentralWidget(w)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.cam.start()
        self.inf.start()

    # ── NAV ────────────────────────────────────────
    def _go(self, idx):
        self.stack.setCurrentIndex(idx)
        self.btn_live.setStyleSheet(NAV_ACTIVE   if idx == 0 else NAV_IDLE)
        self.btn_report.setStyleSheet(NAV_ACTIVE if idx == 1 else NAV_IDLE)

    def toggle_settings(self):
        if self.stack.currentIndex() == 2:
            self._go(0)
        else:
            self.stack.setCurrentIndex(2)
            self.btn_live.setStyleSheet(NAV_IDLE)
            self.btn_report.setStyleSheet(NAV_IDLE)

    def _toggle_mode(self):
        if self.mode == "live":
            self.mode = "motion"
            self.motion_detector.reset()
            self._obj_pending_frame = None
            self.inf.pause()
            self.store["results"] = None
            self.btn_mode.setText("◉ TIMED MODE")
            self.btn_mode.setStyleSheet("""
                QPushButton{background:#2a1a00;color:#ffaa00;
                    border:1px solid #ffaa00;border-radius:6px;font-size:11px;font-weight:bold;}
                QPushButton:hover{background:#ffaa00;color:#0c0d14;}
            """)
        else:
            self.mode = "live"
            self.motion_detector.reset()
            self.inf.resume()
            self.btn_mode.setText("● LIVE MODE")
            self.btn_mode.setStyleSheet("""
                QPushButton{background:#1a2a1a;color:#00ff88;
                    border:1px solid #00ff88;border-radius:6px;font-size:11px;font-weight:bold;}
                QPushButton:hover{background:#00ff88;color:#0c0d14;}
            """)

    def _reset_camera_defaults(self):
        panel = self.settings_page.layout().itemAt(1).widget()
        for i in range(panel.layout().count()):
            widget = panel.layout().itemAt(i).widget()
            if isinstance(widget, ControlSlider):
                default = CAMERA_SETTINGS.get(widget.prop_name)
                if default is not None:
                    widget.slider.setValue(int(default))

    def set_fps(self, f):
        if not self.inf._paused or self.inf._requested_frame is not None:
            self.fps = f

    # ── TIMED MOTION MODE ──────────────────────────
    def _update_motion_mode(self, frame):
        """Hybrid capture mode — triggers on stillness or fallback timer.

        Display shows the last annotated capture (not live feed) so the
        operator can inspect each inference result before the next capture.
        The HUD overlays live telemetry: diff score, blur score, and
        what triggered the last capture (or why it was rejected).
        """
        trigger = self.motion_detector.update(frame)

        # Trigger: grab frame and send to inference
        if trigger:
            self._obj_pending_frame = frame.copy()
            self.inf.request_frame(frame.copy())

        # Collect result when inference is done
        if self._obj_pending_frame is not None:
            res = self.store["results"]
            if res is not None:
                self._process_detections(
                    self._obj_pending_frame, res,
                    self._obj_pending_frame.copy()
                )
                self._obj_pending_frame = None

        # ── Display: show last annotated image instead of live feed ──
        if hasattr(self, "_last_ann_path") and self._last_ann_path:
            display = cv2.imread(self._last_ann_path)
            if display is None:
                display = self._motion_blank(frame)
        else:
            display = self._motion_blank(frame)

        # ── HUD overlay ──────────────────────────────────────────────
        fh, fw    = display.shape[:2]
        remaining = self.motion_detector.remaining()
        progress  = self.motion_detector.progress()

        # Diff & blur scores from detector
        diff_val  = self.motion_detector.last_diff
        blur_val  = self.motion_detector.last_blur
        reason    = self.motion_detector.trigger_reason

        # Fabric state indicator
        fabric_moving = diff_val >= STILLNESS_DIFF_THRESHOLD
        state_text    = "MOVING" if fabric_moving else "STILL"
        state_color   = (0, 100, 255) if fabric_moving else (0, 255, 100)

        # Blur quality indicator
        blur_ok    = blur_val >= BLUR_THRESHOLD
        blur_text  = f"Sharp ({blur_val:.0f})" if blur_ok else f"Blurry ({blur_val:.0f})"
        blur_color = (0, 255, 150) if blur_ok else (0, 80, 255)

        # Trigger reason badge
        reason_color = {
            "still":         (0, 255, 100),
            "fallback":      (0, 200, 255),
            "blur_rejected": (0, 80, 255),
            "waiting":       (180, 140, 0),
            "":              (60, 60, 80),
        }.get(reason, (60, 60, 80))
        reason_text = {
            "still":         "TRIGGERED: fabric still",
            "fallback":      "TRIGGERED: fallback timer",
            "blur_rejected": "SKIPPED: blurry frame",
            "waiting":       "WAITING: must move before next capture",
            "":              "Waiting...",
        }.get(reason, "")

        # Fallback progress bar at bottom
        bar_w  = fw - 40
        filled = int(bar_w * progress)
        cv2.rectangle(display, (20, fh - 14), (20 + bar_w, fh - 6), (20, 20, 40), -1)
        bar_col = (0, 200, 255)
        cv2.rectangle(display, (20, fh - 14), (20 + filled, fh - 6), bar_col, -1)
        cv2.putText(display, f"Fallback in {remaining:.1f}s",
                    (22, fh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_col, 1)

        # Top HUD bar (dark background strip)
        cv2.rectangle(display, (0, 0), (fw, 52), (0, 0, 0), -1)

        # Row 1: mode label + total
        cv2.putText(display,
                    "HYBRID MODE  |  Still-first + fallback timer",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Total: {self.total_count}",
                    (fw - 120, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 204), 1, cv2.LINE_AA)

        # Row 2: fabric state | blur | trigger reason
        cv2.putText(display, f"Fabric: {state_text}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)
        cv2.putText(display, f"| {blur_text}",
                    (160, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, blur_color, 1, cv2.LINE_AA)
        cv2.putText(display, f"| {reason_text}",
                    (360, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, reason_color, 1, cv2.LINE_AA)

        self._render_frame_to_label(display, self.live_label)


    def _motion_blank(self, frame):
        """Black placeholder shown in timed mode before the first capture."""
        blank = np.zeros_like(frame)
        fh, fw = blank.shape[:2]
        cv2.putText(blank, "Waiting for first capture...",
                    (fw // 2 - 200, fh // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (60, 60, 100), 2, cv2.LINE_AA)
        return blank

    # ── SHARED DETECTION PROCESSOR ─────────────────
    def _process_detections(self, clean_frame, res, annotated_frame):
        stable_detections = []
        tracked           = []
        class_names       = res.names

        if res.boxes:
            boxes = []
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                cls  = int(b.cls[0])
                boxes.append((x1, y1, x2, y2, conf, cls))

            tracked = self.tracker.update(boxes)
            for x1, y1, x2, y2, conf, cls, oid, count in tracked:
                color = (0, 255, 0) if count >= STABLE_FRAMES_REQUIRED else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{class_names.get(cls, str(cls))} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_y = max(y1, th + 4)
                cv2.rectangle(annotated_frame,
                            (x1, tag_y - th - 4), (x1 + tw + 4, tag_y + baseline),
                            color, -1)
                cv2.putText(annotated_frame, label_text, (x1 + 2, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"ID{oid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if count >= STABLE_FRAMES_REQUIRED:
                    stable_detections.append((cls, x1, y1, x2, y2))

        now      = datetime.datetime.now()
        ts       = now.strftime("%Y%m%d_%H%M%S")
        raw_path = f"captures/raw/{ts}.jpg"
        ann_path = f"captures/annotated/{ts}.jpg"
        lbl_path = f"captures/labels/{ts}.txt"

        cv2.imwrite(raw_path, clean_frame)
        cv2.imwrite(ann_path, annotated_frame)

        # Always update the display path so timed mode shows the latest frame
        self._last_ann_path = ann_path

        # Only write label file and log if there were actual detections
        if not stable_detections:
            return

        ih, iw = clean_frame.shape[:2]
        with open(lbl_path, "w") as f:
            for (cls, x1, y1, x2, y2) in stable_detections:
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        cls_conf = {}
        for tx1, ty1, tx2, ty2, tconf, tcls, toid, tcount in tracked:
            if tcount >= STABLE_FRAMES_REQUIRED:
                cls_conf[tcls] = max(cls_conf.get(tcls, 0.0), tconf)

        det_summary  = ", ".join(
            f"{class_names.get(cls, '?')} ({cls_conf.get(cls, 0):.2f})"
            for (cls, _, _, _, _) in stable_detections)
        primary_cls  = class_names.get(stable_detections[0][0], "?")
        primary_conf = round(float(cls_conf.get(stable_detections[0][0], 0.0)), 3)

        self.led.trigger(primary_cls)
        self.defect_log.add({
            "ts": now, "class_name": primary_cls,
            "classes": det_summary, "conf": primary_conf,
            "raw": raw_path, "ann": ann_path, "lbl": lbl_path,
        })
        self.total_count += 1
        self._refresh_dashboard()

    def _render_frame_to_label(self, frame, label):
        h, w   = frame.shape[:2]
        lw, lh = label.width(), label.height()
        if lw < 10 or lh < 10:
            return
        scale   = min(lw / w, lh / h)
        nw, nh  = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))

    # ── FRAME UPDATE ───────────────────────────────
    def update_frame(self):
        frame = self.buf.get()
        if frame is None:
            return

        current_page = self.stack.currentIndex()

        if current_page == 2:
            self._render_frame_to_label(frame, self.settings_feed)
            return

        if self.mode == "motion":
            self._update_motion_mode(frame)
            return

        # ── LIVE MODE ─────────────────────────────
        res    = self.store["results"]
        stable = []
        stable_detections = []
        clean_frame = frame.copy()

        if res and res.boxes:
            boxes = []
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                cls  = int(b.cls[0])
                boxes.append((x1, y1, x2, y2, conf, cls))

            tracked     = self.tracker.update(boxes)
            class_names = res.names

            for x1, y1, x2, y2, conf, cls, oid, count in tracked:
                color = (0, 255, 0) if count >= STABLE_FRAMES_REQUIRED else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label_text = f"{class_names.get(cls, str(cls))} {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_y = max(y1, th + 4)
                cv2.rectangle(frame,
                              (x1, tag_y - th - 4), (x1 + tw + 4, tag_y + baseline),
                              color, -1)
                cv2.putText(frame, label_text, (x1 + 2, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"ID{oid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                if count >= STABLE_FRAMES_REQUIRED:
                    stable.append(1)
                    stable_detections.append((cls, x1, y1, x2, y2))

        if stable and time.time() - self.last_capture > CAPTURE_COOLDOWN:
            self.last_capture = time.time()
            self._process_detections(clean_frame, res, frame)

        h, w   = frame.shape[:2]
        lw, lh = self.live_label.width(), self.live_label.height()
        if lw < 10 or lh < 10:
            return

        scale   = min(lw / w, lh / h)
        nw, nh  = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        img = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        painter = QPainter(pix)
        painter.setPen(QColor("#00ffcc"))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(10, 20, f"LIVE  FPS: {self.fps:.1f}")
        painter.end()

        self.live_label.setPixmap(pix)

    # ── DASHBOARD ──────────────────────────────────
    def _build_dashboard(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(10)

        toolbar = QWidget()
        toolbar.setStyleSheet("QWidget{background:#0f1020;border-radius:8px;}")
        toolbar.setFixedHeight(46)
        top = QHBoxLayout(toolbar)
        top.setContentsMargins(12, 0, 12, 0)
        top.setSpacing(8)

        title = QLabel("Defect Dashboard")
        title.setStyleSheet("QLabel{color:#7c6ff7;font-size:14px;font-weight:bold;background:transparent;}")
        top.addWidget(title)

        sep = QLabel("|")
        sep.setStyleSheet("QLabel{color:#1a1c2e;background:transparent;}")
        top.addWidget(sep)

        lbl_from = QLabel("From:")
        lbl_from.setStyleSheet("QLabel{color:#aab0d0;font-size:11px;background:transparent;}")
        top.addWidget(lbl_from)
        self.dt_from = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-3600))
        self.dt_from.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.dt_from.setCalendarPopup(True)
        self.dt_from.setFixedWidth(145)
        top.addWidget(self.dt_from)

        lbl_to = QLabel("To:")
        lbl_to.setStyleSheet("QLabel{color:#aab0d0;font-size:11px;background:transparent;}")
        top.addWidget(lbl_to)
        self.dt_to = QDateTimeEdit(QDateTime.currentDateTime())
        self.dt_to.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.dt_to.setCalendarPopup(True)
        self.dt_to.setFixedWidth(145)
        top.addWidget(self.dt_to)

        self.filter_class = QComboBox()
        self.filter_class.addItem("All")
        self.filter_class.setFixedWidth(110)
        top.addWidget(self.filter_class)

        btn_filter = QPushButton("Apply")
        btn_filter.setStyleSheet(FILTER_BTN)
        btn_filter.setFixedSize(64, 30)
        btn_filter.clicked.connect(self._refresh_dashboard)
        top.addWidget(btn_filter)

        btn_reset = QPushButton("Reset")
        btn_reset.setStyleSheet("""QPushButton{background:#1a1c2e;color:#aab0d0;
            border:1px solid #2a2c3e;border-radius:6px;font-size:11px;}
            QPushButton:hover{background:#2a2c3e;}""")
        btn_reset.setFixedSize(56, 30)
        btn_reset.clicked.connect(self._reset_filter)
        top.addWidget(btn_reset)

        top.addStretch()

        btn_csv = QPushButton("⬇ CSV")
        btn_csv.setStyleSheet(EXPORT_BTN)
        btn_csv.setFixedSize(80, 30)
        btn_csv.clicked.connect(self._export_csv)
        top.addWidget(btn_csv)

        btn_report_exp = QPushButton("⬇ Report")
        btn_report_exp.setStyleSheet(EXPORT_BTN)
        btn_report_exp.setFixedSize(86, 30)
        btn_report_exp.clicked.connect(self._export_report)
        top.addWidget(btn_report_exp)

        root.addWidget(toolbar)

        mid = QHBoxLayout()
        mid.setSpacing(10)

        cards_scroll = QScrollArea()
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setFixedHeight(110)
        cards_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        cards_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.cards_inner  = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_inner)
        self.cards_layout.setAlignment(Qt.AlignLeft)
        self.cards_layout.setSpacing(10)
        cards_scroll.setWidget(self.cards_inner)
        mid.addWidget(cards_scroll, stretch=3)

        preview_frame = QFrame()
        preview_frame.setStyleSheet(CARD_STYLE)
        preview_frame.setFixedWidth(240)
        pf_l = QVBoxLayout(preview_frame)
        pf_l.setContentsMargins(8, 8, 8, 8)

        plbl_row = QHBoxLayout()
        plbl = QLabel("Last Defect")
        plbl.setStyleSheet("QLabel{color:#7c6ff7;font-size:11px;font-weight:bold;background:transparent;}")
        plbl_row.addWidget(plbl)
        plbl_row.addStretch()
        zoom_hint = QLabel("🔍 click to enlarge")
        zoom_hint.setStyleSheet("QLabel{color:#2a2c4e;font-size:9px;background:transparent;}")
        plbl_row.addWidget(zoom_hint)
        pf_l.addLayout(plbl_row)

        self.preview_img = ClickableImageLabel()
        self.preview_img.setAlignment(Qt.AlignCenter)
        self.preview_img.setFixedHeight(74)
        self.preview_img.setStyleSheet("""
            QLabel{background:#06070d;border-radius:6px;}
            QLabel:hover{border:1px solid #7c6ff7;}
        """)
        self.preview_img.clicked.connect(self._open_lightbox)
        pf_l.addWidget(self.preview_img)

        self.preview_caption = QLabel("—")
        self.preview_caption.setStyleSheet(
            "QLabel{color:#454870;font-size:10px;background:transparent;}")
        self.preview_caption.setAlignment(Qt.AlignCenter)
        pf_l.addWidget(self.preview_caption)
        mid.addWidget(preview_frame, stretch=0)

        root.addLayout(mid)

        self.event_table = QTableWidget()
        self.event_table.setColumnCount(5)
        self.event_table.setHorizontalHeaderLabels(
            ["Timestamp", "Class", "Confidence", "Annotated", "Raw"])
        self.event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.event_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.setStyleSheet(
            "QTableWidget{alternate-background-color:#0f1020;}")
        self.event_table.verticalHeader().setVisible(False)
        root.addWidget(self.event_table)

        return page

    def _make_stat_card(self, class_name, count, color):
        card = QFrame()
        card.setStyleSheet(
            f"QFrame{{background:#11121c;border:1px solid {color};border-radius:10px;}}")
        card.setFixedSize(140, 90)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)

        dot_row = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"QLabel{{color:{color};font-size:10px;background:transparent;}}")
        name_lbl = QLabel(class_name)
        name_lbl.setStyleSheet("QLabel{color:#aab0d0;font-size:11px;background:transparent;}")
        dot_row.addWidget(dot)
        dot_row.addWidget(name_lbl)
        dot_row.addStretch()
        cl.addLayout(dot_row)

        count_lbl = QLabel(str(count))
        count_lbl.setStyleSheet(
            f"QLabel{{color:{color};font-size:28px;font-weight:bold;background:transparent;}}")
        cl.addWidget(count_lbl)

        sub_lbl = QLabel("detections")
        sub_lbl.setStyleSheet("QLabel{color:#454870;font-size:10px;background:transparent;}")
        cl.addWidget(sub_lbl)

        return card

    def _get_filtered_entries(self):
        from_dt = self.dt_from.dateTime().toPyDateTime()
        to_dt   = self.dt_to.dateTime().toPyDateTime()
        cls_f   = self.filter_class.currentText()
        return self.defect_log.get_filtered(
            from_dt, to_dt, None if cls_f == "All" else cls_f)

    def _refresh_dashboard(self):
        self.dt_to.setDateTime(QDateTime.currentDateTime())

        known   = self.defect_log.all_class_names()
        current = self.filter_class.currentText()
        self.filter_class.blockSignals(True)
        self.filter_class.clear()
        self.filter_class.addItem("All")
        for cn in known:
            self.filter_class.addItem(cn)
        idx = self.filter_class.findText(current)
        self.filter_class.setCurrentIndex(idx if idx >= 0 else 0)
        self.filter_class.blockSignals(False)

        entries = self._get_filtered_entries()
        counts  = self.defect_log.class_counts(entries)

        COLORS = ["#7c6ff7", "#00ffcc", "#ff6b6b", "#ffd166", "#06d6a0",
                  "#118ab2", "#ef476f", "#ffd166", "#26c6da", "#ab47bc"]

        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total_card = self._make_stat_card("TOTAL", len(entries), "#7c6ff7")
        self.cards_layout.addWidget(total_card)

        for i, (cls_name, cnt) in enumerate(
                sorted(counts.items(), key=lambda x: -x[1])):
            color = COLORS[(i + 1) % len(COLORS)]
            self.cards_layout.addWidget(self._make_stat_card(cls_name, cnt, color))

        if entries:
            last = entries[-1]
            self._last_ann_path = last["ann"]
            pix = QPixmap(last["ann"])
            if not pix.isNull():
                self.preview_img.setPixmap(
                    pix.scaled(self.preview_img.width(), self.preview_img.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.preview_caption.setText(
                f"{last['class_name']}  {last['ts'].strftime('%H:%M:%S')}")
        else:
            self._last_ann_path = None
            self.preview_img.clear()
            self.preview_caption.setText("—")

        self.event_table.setRowCount(0)
        for e in reversed(entries):
            row = self.event_table.rowCount()
            self.event_table.insertRow(row)
            self.event_table.setItem(
                row, 0, QTableWidgetItem(e["ts"].strftime("%Y-%m-%d %H:%M:%S")))
            self.event_table.setItem(
                row, 1, QTableWidgetItem(e.get("classes", e["class_name"])))
            self.event_table.setItem(
                row, 2, QTableWidgetItem(f"{e['conf']:.3f}"))
            self.event_table.setItem(row, 3, QTableWidgetItem(e["ann"]))
            self.event_table.setItem(row, 4, QTableWidgetItem(e["raw"]))

    def _reset_filter(self):
        self.dt_from.setDateTime(QDateTime.currentDateTime().addSecs(-3600))
        self.dt_to.setDateTime(QDateTime.currentDateTime())
        self.filter_class.setCurrentIndex(0)
        self._refresh_dashboard()

    def _open_lightbox(self):
        if not hasattr(self, "_last_ann_path") or not self._last_ann_path:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Defect Preview")
        dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dlg.setAttribute(Qt.WA_TranslucentBackground)

        overlay = QWidget(dlg)
        overlay.setStyleSheet(
            "QWidget{background:rgba(6,7,13,220);border-radius:12px;}")

        img_lbl = QLabel(overlay)
        img_lbl.setAlignment(Qt.AlignCenter)
        pix    = QPixmap(self._last_ann_path)
        screen = QApplication.primaryScreen().availableGeometry()
        max_w  = int(screen.width()  * 0.80)
        max_h  = int(screen.height() * 0.80)
        scaled = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_lbl.setPixmap(scaled)
        img_lbl.setFixedSize(scaled.width(), scaled.height())

        caption = QLabel(self.preview_caption.text(), overlay)
        caption.setStyleSheet(
            "QLabel{color:#7c6ff7;font-size:13px;font-weight:bold;background:transparent;}")
        caption.setAlignment(Qt.AlignCenter)

        close_btn = QPushButton("✕  Close", overlay)
        close_btn.setStyleSheet("""QPushButton{
            background:#1a1c2e;color:#dde0f0;border:1px solid #2a2c4e;
            border-radius:8px;padding:6px 20px;font-size:13px;}
            QPushButton:hover{background:#7c6ff7;color:white;}""")
        close_btn.clicked.connect(dlg.accept)

        vbox = QVBoxLayout(overlay)
        vbox.setContentsMargins(20, 20, 20, 20)
        vbox.setSpacing(10)
        vbox.addWidget(img_lbl)
        vbox.addWidget(caption)
        vbox.addWidget(close_btn, alignment=Qt.AlignCenter)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(overlay)

        dlg.adjustSize()
        dlg.move(
            screen.center().x() - dlg.width()  // 2,
            screen.center().y() - dlg.height() // 2,
        )
        dlg.exec_()

    def _export_csv(self):
        entries = self._get_filtered_entries()
        if not entries:
            QMessageBox.information(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "defect_report.csv", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Timestamp", "Class", "Confidence",
                        "Annotated Path", "Raw Path", "Label Path"])
            for e in entries:
                w.writerow([e["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                            e.get("classes", e["class_name"]), e["conf"],
                            e["ann"], e["raw"], e["lbl"]])
        QMessageBox.information(
            self, "Export", f"Saved {len(entries)} rows to:\n{path}")

    def _export_report(self):
        entries = self._get_filtered_entries()
        if not entries:
            QMessageBox.information(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "defect_report.txt", "Text Files (*.txt)")
        if not path:
            return
        counts = self.defect_log.class_counts(entries)
        from_s = self.dt_from.dateTime().toString("yyyy-MM-dd HH:mm")
        to_s   = self.dt_to.dateTime().toString("yyyy-MM-dd HH:mm")
        lines  = [
            "=" * 60,
            "  DEFECT INSPECTION REPORT",
            "=" * 60,
            f"  Period   : {from_s}  →  {to_s}",
            f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Total    : {len(entries)} detections",
            "",
            "  SUMMARY BY CLASS",
            "  " + "-" * 30,
        ]
        for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / len(entries) if entries else 0
            lines.append(f"  {cls_name:<25} {cnt:>5}  ({pct:.1f}%)")
        lines += ["", "  EVENT LOG", "  " + "-" * 56,
                  f"  {'Timestamp':<22} {'Class':<20} {'Conf':>6}"]
        lines.append("  " + "-" * 56)
        for e in entries:
            lines.append(
                f"  {e['ts'].strftime('%Y-%m-%d %H:%M:%S'):<22}"
                f" {e['class_name']:<20} {e['conf']:>6.3f}")
        lines += ["", "=" * 60, "  END OF REPORT", "=" * 60]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        QMessageBox.information(self, "Export", f"Report saved to:\n{path}")

    def _update_led_status(self):
        if not hasattr(self, "led"):
            return
        if self.led.connected:
            self.led_status.setStyleSheet(
                "QLabel{background:#00ffcc;border-radius:5px;}")
            self.led_status_label.setText(f"LED  {LED_SERIAL_PORT}")
            self.led_status_label.setStyleSheet("QLabel{color:#00ffcc;font-size:11px;}")
        else:
            self.led_status.setStyleSheet(
                "QLabel{background:#454870;border-radius:5px;}")
            self.led_status_label.setText("LED  offline")
            self.led_status_label.setStyleSheet("QLabel{color:#454870;font-size:11px;}")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.showMinimized()
        elif e.key() == Qt.Key_Q:
            self.close()

    def closeEvent(self, e):
        self.cam.stop()
        self.inf.stop()
        self.led.close()
        e.accept()


# ── RUN ─────────────────────────────────────────
if __name__ == "__main__":
    # Required for PyInstaller + PyTorch/multiprocessing on Windows.
    # Must be called before any GUI or model code runs.
    import multiprocessing
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    w = App()
    w.show()
    sys.exit(app.exec_())