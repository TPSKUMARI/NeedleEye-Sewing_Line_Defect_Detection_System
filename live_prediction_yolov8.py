"""
live_prediction_yolov8.py  –  NeedleEye: Defect & Stitch Detection
Optimised for CPU-only execution on Windows (Geekom A5 / Ryzen 5 5600U).

Architecture
─────────────
  CameraThread  →  FrameBuffer  →  InferenceThread("Defect")
                               →  InferenceThread("Stitch")
                                        ↓ DetectionStore (× 2)
                          QTimer (50 ms) → compositor → Qt display

Run `python export_model.py --model best.pt` first to get best.onnx (2-4× faster).

Windows camera index: DEVICE = 0 is usually the built-in webcam.
                      DEVICE = 1 is the first external USB camera.
Adjust DEVICE below if the wrong camera opens.
"""

# ── Set CPU thread limits BEFORE any numeric library loads ─────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS",     "2")
os.environ.setdefault("MKL_NUM_THREADS",     "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
# ──────────────────────────────────────────────────────────────────────────

import sys
import cv2
import datetime
import time
import threading

import numpy as np
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtGui  import QImage, QPixmap, QFont, QColor

# ── OpenCV tuning ──────────────────────────────────────────────────────────
cv2.setNumThreads(4)
cv2.setUseOptimized(True)


# ═══════════════════════════════════════════════════════
#  UI STYLESHEET
# ═══════════════════════════════════════════════════════
APP_STYLE = """
* {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #dde0f0;
}
QMainWindow, QWidget {
    background-color: #0c0d14;
}

/* ── Side panel card ── */
#sidePanel {
    background-color: #11121c;
    border-left: 1px solid #1e2035;
}

/* ── Section cards ── */
#card {
    background-color: #181a28;
    border: 1px solid #1e2035;
    border-radius: 10px;
    padding: 4px;
}

/* ── Section labels (UPPERCASE headers) ── */
#sectionHeader {
    color: #454870;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
}

/* ── GroupBox for camera controls ── */
QGroupBox {
    background-color: #181a28;
    border: 1px solid #1e2035;
    border-radius: 10px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    font-size: 10px;
    font-weight: bold;
    color: #454870;
    letter-spacing: 2px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: #454870;
}

/* ── Sliders ── */
QSlider::groove:horizontal {
    border: none;
    height: 3px;
    background: #1e2035;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                stop:0 #4f46c8, stop:1 #7c6ff7);
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #7c6ff7;
    border: 2px solid #0c0d14;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #9d93ff;
}
QSlider::handle:horizontal:disabled {
    background: #2a2d44;
    border-color: #0c0d14;
}

/* ── Checkboxes ── */
QCheckBox {
    color: #454870;
    font-size: 10px;
    spacing: 4px;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border-radius: 3px;
    border: 1px solid #2a2d44;
    background: #11121c;
}
QCheckBox::indicator:checked {
    background: #7c6ff7;
    border-color: #7c6ff7;
    image: none;
}

/* ── Capture button ── */
#captureBtn {
    background-color: #7c6ff7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 0;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
}
#captureBtn:hover {
    background-color: #9d93ff;
}
#captureBtn:pressed {
    background-color: #5a50d4;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background: #0c0d14;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2d44;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollArea { border: none; }
"""


# ═══════════════════════════════════════════════════════
#  CONFIG  – edit here only
# ═══════════════════════════════════════════════════════
DEVICE = 0          # 0 = first camera, 1 = second (external USB). Adjust as needed.

# Auto-prefer ONNX if already exported (run export_model.py first)
DEFECT_MODEL_PATH = "yolov8_defect.onnx"   if os.path.exists("yolov8_defect.onnx")   else "yolov8_defect.pt"
STITCH_MODEL_PATH = "yolov8n_stitch.onnx" if os.path.exists("yolov8n_stitch.onnx") else "yolov8n_stitch.pt"

INFERENCE_IMGSZ   = 640       # down from 1280 → ~4× less compute
CONF_THRESHOLD    = 0.25
IOU_THRESHOLD     = 0.7

CAPTURE_W         = 1920      # downscale 4K → 1080p before inference
CAPTURE_H         = 1080

INFER_EVERY_N_FRAMES = 10      # run inference on every Nth camera frame (skip the rest)

# ── Camera property defaults (applied at startup via DirectShow) ───────────
# Change any value here; set to None to leave at the camera driver default.
# Range for most props is 0-100 (normalised); Pan/Tilt use -100 to +100.
# If a setting has no visible effect, your camera driver doesn't support it.
CAMERA_SETTINGS = {
    # prop name                  value   auto-disable prop (or None)
    "brightness":                  50,   # cv2.CAP_PROP_BRIGHTNESS
    "contrast":                    50,   # cv2.CAP_PROP_CONTRAST
    "saturation":                  50,   # cv2.CAP_PROP_SATURATION
    "sharpness":                   50,   # cv2.CAP_PROP_SHARPNESS
    "white_balance_temperature":   50,   # disables auto-WB first
    "focus_absolute":              30,   # disables autofocus first
    "pan_absolute":                 0,   # cv2.CAP_PROP_PAN
    "tilt_absolute":                0,   # cv2.CAP_PROP_TILT
    "zoom_absolute":                0,   # cv2.CAP_PROP_ZOOM
}

# Annotation colours (BGR)
DEFECT_COLOR      = (0,   0, 255)   # Red
STITCH_COLOR      = (0, 220,  80)   # Green
BOX_THICKNESS     = 2
FONT_SCALE        = 0.55
FONT              = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════
#  SHARED BUFFERS
# ═══════════════════════════════════════════════════════

class FrameBuffer:
    """
    Thread-safe holder for the *latest* camera frame.

    Includes a monotonic frame_count so each inference thread can
    independently decide "have I already processed this frame?" without
    competing on a shared threading.Event (which caused a race condition
    when two consumer threads called get_latest on the same event).

    Usage pattern for inference threads:
        last_seen = -1
        while running:
            frame, count = buf.get()
            if count == last_seen or count % INFER_EVERY_N == 0:
                ...run inference...
                last_seen = count
            else:
                time.sleep(POLL_INTERVAL)
    """
    def __init__(self):
        self._frame:       np.ndarray | None = None
        self._frame_count: int               = 0
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray):
        with self._lock:
            self._frame       = frame
            self._frame_count += 1

    def get(self) -> tuple[np.ndarray | None, int]:
        """Return (latest_frame_copy, frame_count). Non-blocking."""
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._frame_count

    def peek(self) -> np.ndarray | None:
        """Non-blocking read of latest frame (for display timer)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class DetectionStore:
    """Thread-safe store for the most recent Results object from one model."""
    def __init__(self):
        self._results = None
        self._lock    = threading.Lock()

    def update(self, results):
        with self._lock:
            self._results = results

    def get(self):
        with self._lock:
            return self._results


# ═══════════════════════════════════════════════════════
#  CAMERA THREAD
# ═══════════════════════════════════════════════════════

class CameraThread(QThread):
    def __init__(self, frame_buffer: FrameBuffer):
        super().__init__()
        self._running    = False
        self.frame_buffer = frame_buffer
        self._cap        = None
        self._cap_lock   = threading.Lock()   # guards cap.read() vs cap.set()

    def run(self):
        # Property map: setting name → (cv2 prop id, auto-disable prop or None)
        _PROP_MAP = {
            "brightness":                (cv2.CAP_PROP_BRIGHTNESS,        None),
            "contrast":                  (cv2.CAP_PROP_CONTRAST,          None),
            "saturation":                (cv2.CAP_PROP_SATURATION,        None),
            "sharpness":                 (cv2.CAP_PROP_SHARPNESS,         None),
            "white_balance_temperature": (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, cv2.CAP_PROP_AUTO_WB),
            "focus_absolute":            (cv2.CAP_PROP_FOCUS,             cv2.CAP_PROP_AUTOFOCUS),
            "pan_absolute":              (cv2.CAP_PROP_PAN,               None),
            "tilt_absolute":             (cv2.CAP_PROP_TILT,              None),
            "zoom_absolute":             (cv2.CAP_PROP_ZOOM,              None),
        }

        # CAP_DSHOW = Windows DirectShow backend (required for UVC cameras on Windows)
        self._cap = cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)
        cap = self._cap

        if not cap.isOpened():
            print(f"[Camera] ✗ FAILED to open device index {DEVICE}.")
            print(f"[Camera]   Try changing DEVICE = {DEVICE} in the CONFIG section.")
            print(f"[Camera]   Available indices to try: 0, 1, 2, 3")
            # Probe which indices work so the user knows what to set
            for probe in range(4):
                test = cv2.VideoCapture(probe, cv2.CAP_DSHOW)
                if test.isOpened():
                    print(f"[Camera]   → Device {probe} is available")
                    test.release()
            return   # exit thread; preview stays on "Waiting for camera…"

        print(f"[Camera] ✓ Opened device index {DEVICE}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        # Apply startup camera settings from CAMERA_SETTINGS config
        for name, value in CAMERA_SETTINGS.items():
            if value is None or name not in _PROP_MAP:
                continue
            prop_id, auto_id = _PROP_MAP[name]
            if auto_id is not None:
                cap.set(auto_id, 0)      # disable auto mode first
            cap.set(prop_id, float(value))
            print(f"[Camera] {name} = {value}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] {w}×{h} → resizing to {CAPTURE_W}×{CAPTURE_H} for inference")

        self._running = True
        while self._running:
            with self._cap_lock:
                ret, frame = cap.read()
            if not ret:
                continue
            # Single downscale shared by both inference threads
            small = cv2.resize(frame, (CAPTURE_W, CAPTURE_H),
                               interpolation=cv2.INTER_LINEAR)
            self.frame_buffer.put(small)

        cap.release()
        self._cap = None

    def set_prop(self, prop_id: int, value: float):
        """Thread-safe camera property setter (called from main/UI thread)."""
        with self._cap_lock:
            if self._cap is not None:
                self._cap.set(prop_id, value)

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════════════
#  INFERENCE THREAD  (one instance per model)
# ═══════════════════════════════════════════════════════

class InferenceThread(QThread):
    fps_updated = pyqtSignal(str, float)   # (model_name, fps)

    def __init__(self, name: str, model_path: str,
                 frame_buffer: FrameBuffer,
                 detection_store: DetectionStore):
        super().__init__()
        self.name             = name
        self.model_path       = model_path
        self.frame_buffer     = frame_buffer
        self.detection_store  = detection_store
        self._running         = False

    def run(self):
        if not os.path.exists(self.model_path):
            print(f"[{self.name}] ⚠ Model not found: {self.model_path} – thread idle")
            return

        model = YOLO(self.model_path)
        print(f"[{self.name}] Loaded '{self.model_path}'  classes={list(model.names.values())}")
        print(f"[{self.name}] Inferring on every {INFER_EVERY_N_FRAMES}th frame")

        prev_time  = time.time()
        last_count = 0            # frame counter value we last ran inference on
        self._running = True

        while self._running:
            frame, count = self.frame_buffer.get()

            if frame is None:
                # Camera not ready yet – wait briefly
                time.sleep(0.01)
                continue

            if count == last_count:
                # Same frame we already processed (or already skipped) – yield CPU
                time.sleep(0.005)
                continue

            if count % INFER_EVERY_N_FRAMES != 0:
                # This is not an inference frame – record and skip
                last_count = count
                continue

            # ── This is an inference frame ──────────────────────
            last_count = count

            results = model(
                frame,
                imgsz=INFERENCE_IMGSZ,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
            self.detection_store.update(results[0])

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now
            self.fps_updated.emit(self.name, fps)

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════════════
#  COMPOSITOR HELPER
# ═══════════════════════════════════════════════════════

def draw_detections(frame: np.ndarray, results, color: tuple) -> np.ndarray:
    """
    Draw bounding boxes + labels from an ultralytics Results object
    onto `frame` (in-place).  Returns frame for chaining.
    """
    if results is None:
        return frame
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    names = results.names
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3),
                    FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════
#  WINDOWS CAMERA CONTROL  (OpenCV CAP_PROP_* via DirectShow)
# ═══════════════════════════════════════════════════════

# Maps control name → (cv2 property id, disables_auto_prop_id_or_None)
# The optional second element is the "disable auto" property to set=0 first.
PROP_MAP = {
    "brightness":                (cv2.CAP_PROP_BRIGHTNESS,        None),
    "contrast":                  (cv2.CAP_PROP_CONTRAST,          None),
    "saturation":                (cv2.CAP_PROP_SATURATION,        None),
    "sharpness":                 (cv2.CAP_PROP_SHARPNESS,         None),
    "white_balance_temperature": (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, cv2.CAP_PROP_AUTO_WB),
    "focus_absolute":            (cv2.CAP_PROP_FOCUS,             cv2.CAP_PROP_AUTOFOCUS),
    "pan_absolute":              (cv2.CAP_PROP_PAN,               None),
    "tilt_absolute":             (cv2.CAP_PROP_TILT,              None),
    "zoom_absolute":             (cv2.CAP_PROP_ZOOM,              None),
}


# ═══════════════════════════════════════════════════════
#  CAMERA CONTROL ROW  (slider + lock)
# ═══════════════════════════════════════════════════════

class ControlRow(QWidget):
    """Slim labelled slider row with value badge and lock checkbox."""
    def __init__(self, name, ctrl, min_val, max_val, init_val, cam_thread=None):
        super().__init__()
        self.ctrl       = ctrl
        self.cam_thread = cam_thread

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 3, 2, 3)
        layout.setSpacing(6)

        # Label
        lbl = QLabel(name)
        lbl.setFixedWidth(88)
        lbl.setStyleSheet("color:#6870a0; font-size:11px;")
        layout.addWidget(lbl)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(init_val)
        self.slider.setFixedHeight(18)
        layout.addWidget(self.slider, 1)

        # Value badge
        self.value_label = QLabel(str(init_val))
        self.value_label.setFixedWidth(34)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(
            "background:#1e2035; color:#9d93ff; font-size:10px;"
            "border-radius:4px; padding:1px 0;"
        )
        layout.addWidget(self.value_label)

        # Lock
        self.lock = QCheckBox("Hold")
        layout.addWidget(self.lock)

        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.enforce)
        self.slider.valueChanged.connect(self.on_change)
        self.lock.toggled.connect(self.on_toggle)

    def apply(self, value):
        if self.cam_thread is None or self.ctrl not in PROP_MAP:
            return
        prop_id, auto_prop_id = PROP_MAP[self.ctrl]
        if auto_prop_id is not None:
            self.cam_thread.set_prop(auto_prop_id, 0)
        self.cam_thread.set_prop(prop_id, float(value))

    def on_change(self, value):
        self.value_label.setText(str(value))
        if not self.lock.isChecked():
            self.apply(value)

    def on_toggle(self, checked):
        self.slider.setEnabled(not checked)
        if checked:
            self.timer.start()
            self.apply(self.slider.value())
        else:
            self.timer.stop()

    def enforce(self):
        self.apply(self.slider.value())



# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION  –  Two-page design (Live | Report)
# ═══════════════════════════════════════════════════════════════════

from PyQt5.QtWidgets import (
    QPushButton, QSizePolicy, QScrollArea,
    QGridLayout, QStackedWidget
)
from PyQt5.QtGui import QPainter, QBrush


# ── Navigation button styles ────────────────────────────────────────
_NAV_ACTIVE = """QPushButton {
    background:#7c6ff7; color:#ffffff;
    border:none; border-radius:6px;
    font-size:12px; font-weight:bold;
    padding:8px 28px; letter-spacing:1px;
}"""

_NAV_IDLE = """QPushButton {
    background:#11121c; color:#454870;
    border:1px solid #1a1c2e; border-radius:6px;
    font-size:12px; padding:8px 28px;
}
QPushButton:hover { background:#1a1c2e; color:#7c6ff7; }"""


# ── Tiny stat card for the report page ─────────────────────────────
class _CountCard(QWidget):
    """A single rounded card showing class name + running total."""
    def __init__(self, class_name, accent):
        super().__init__()
        self.setFixedSize(160, 100)
        self.setStyleSheet(
            f"background:#181a28; border-radius:12px;"
            f"border:1px solid #1e2035;"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        dot_and_name = QHBoxLayout()
        dot_and_name.setContentsMargins(0, 0, 0, 0)
        dot = QLabel("■")
        dot.setStyleSheet(f"color:{accent}; font-size:11px; background:transparent; border:none;")
        name_lbl = QLabel(class_name)
        name_lbl.setStyleSheet("color:#6870a0; font-size:10px; background:transparent; border:none;")
        name_lbl.setWordWrap(True)
        dot_and_name.addWidget(dot)
        dot_and_name.addWidget(name_lbl, 1)
        lay.addLayout(dot_and_name)

        self.num_lbl = QLabel("0")
        self.num_lbl.setStyleSheet(
            f"color:{accent}; font-size:28px; font-weight:bold;"
            f"background:transparent; border:none;"
        )
        lay.addWidget(self.num_lbl)

        unit = QLabel("captures")
        unit.setStyleSheet("color:#2e3060; font-size:9px; background:transparent; border:none;")
        lay.addWidget(unit)

    def set_count(self, n):
        self.num_lbl.setText(str(n))


# ── Thumbnail card for the gallery ─────────────────────────────────
class _ThumbCard(QWidget):
    def __init__(self, pixmap, ts_str, summary):
        super().__init__()
        self.setFixedWidth(180)
        self.setStyleSheet(
            "background:#181a28; border-radius:10px; border:1px solid #1e2035;"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 8)
        lay.setSpacing(4)

        img_lbl = QLabel()
        img_lbl.setFixedSize(168, 100)
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setStyleSheet("background:#0d0e18; border-radius:6px; border:none;")
        img_lbl.setPixmap(pixmap.scaled(168, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(img_lbl)

        ts_lbl = QLabel(ts_str)
        ts_lbl.setStyleSheet("color:#9095c0; font-size:9px; background:transparent; border:none;")
        lay.addWidget(ts_lbl)

        sum_lbl = QLabel(summary)
        sum_lbl.setStyleSheet("color:#454870; font-size:9px; background:transparent; border:none;")
        sum_lbl.setWordWrap(True)
        lay.addWidget(sum_lbl)


# ══════════════════════════════════════════════════════════════════ 
class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeedleEye  ·  AI Inspection")
        self.resize(1500, 900)
        self.setMinimumSize(800, 500)

        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)

        # ── Shared buffers & threads ──────────────────────────────
        self.frame_buffer  = FrameBuffer()
        self.defect_store  = DetectionStore()
        self.stitch_store  = DetectionStore()
        self.cam_thread    = CameraThread(self.frame_buffer)
        self.defect_thread = InferenceThread(
            "Defect", DEFECT_MODEL_PATH, self.frame_buffer, self.defect_store)
        self.stitch_thread = InferenceThread(
            "Stitch", STITCH_MODEL_PATH, self.frame_buffer, self.stitch_store)
        self._defect_fps   = 0.0
        self._stitch_fps   = 0.0
        self.defect_thread.fps_updated.connect(self._on_fps)
        self.stitch_thread.fps_updated.connect(self._on_fps)

        # ── Session state ─────────────────────────────────────────
        self._session_totals      = {}   # {class_name: cumulative_count}
        self._count_cards         = {}   # {class_name: _CountCard}
        self._last_auto_capture_t = 0.0
        self._capture_frame       = None
        self._live_state          = True

        # ── Root layout (stacked pages + nav bar) ─────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # ── Nav bar (always visible at bottom) ────────────────────
        nav = QWidget()
        nav.setFixedHeight(52)
        nav.setStyleSheet("background:#0d0e18; border-top:2px solid #1a1c2e;")
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(20, 6, 20, 6)
        nav_lay.setSpacing(10)

        self.btn_live   = QPushButton("  LIVE VIEW")
        self.btn_report = QPushButton("  REPORT")
        self.btn_live.setStyleSheet(_NAV_ACTIVE)
        self.btn_report.setStyleSheet(_NAV_IDLE)
        self.btn_live.clicked.connect(lambda: self._switch(0))
        self.btn_report.clicked.connect(lambda: self._switch(1))

        nav_lay.addStretch()
        nav_lay.addWidget(self.btn_live)
        nav_lay.addWidget(self.btn_report)
        nav_lay.addStretch()
        root.addWidget(nav)

        # ════════════════════════════════════════════════
        #  PAGE 0 — LIVE VIEW
        # ════════════════════════════════════════════════
        live_page = QWidget()
        live_page.setStyleSheet("background:#07080f;")
        live_lay = QVBoxLayout(live_page)
        live_lay.setContentsMargins(0, 0, 0, 0)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#07080f;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        live_lay.addWidget(self.preview)
        self.stack.addWidget(live_page)

        # ════════════════════════════════════════════════
        #  PAGE 1 — INSPECTION REPORT
        # ════════════════════════════════════════════════
        report_page = QWidget()
        report_page.setStyleSheet("background:#0c0d14;")
        report_outer = QVBoxLayout(report_page)
        report_outer.setContentsMargins(0, 0, 0, 0)
        report_outer.setSpacing(0)

        # Report header bar
        rh = QWidget()
        rh.setFixedHeight(60)
        rh.setStyleSheet("background:#0d0e18; border-bottom:1px solid #1a1c2e;")
        rh_lay = QHBoxLayout(rh)
        rh_lay.setContentsMargins(28, 0, 28, 0)

        r_title = QLabel("NeedleEye")
        r_title.setStyleSheet(
            "font-size:18px; font-weight:bold; color:#e4e6f8; letter-spacing:1px;"
        )
        rh_lay.addWidget(r_title)

        r_sub = QLabel("  ·  Inspection Report")
        r_sub.setStyleSheet("font-size:13px; color:#353758;")
        rh_lay.addWidget(r_sub)
        rh_lay.addStretch()

        self._clear_btn = QPushButton("Clear Session")
        self._clear_btn.setStyleSheet(
            "QPushButton { background:#1a1c2e; color:#454870; border:1px solid #1e2035;"
            "border-radius:6px; padding:6px 16px; font-size:11px; }"
            "QPushButton:hover { background:#252840; color:#7c6ff7; }"
        )
        self._clear_btn.clicked.connect(self._clear_session)
        rh_lay.addWidget(self._clear_btn)
        report_outer.addWidget(rh)

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:#0c0d14; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        report_body = QWidget()
        report_body.setStyleSheet("background:#0c0d14;")
        self._report_body_lay = QVBoxLayout(report_body)
        self._report_body_lay.setContentsMargins(28, 24, 28, 24)
        self._report_body_lay.setSpacing(24)

        # ── Section: Session totals ──
        sec1 = QLabel("SESSION TOTALS")
        sec1.setStyleSheet("color:#2e3060; font-size:10px; font-weight:bold; letter-spacing:2px;")
        self._report_body_lay.addWidget(sec1)

        self._cards_wrap = QWidget()
        self._cards_wrap.setStyleSheet("background:transparent;")
        self._cards_flow = QHBoxLayout(self._cards_wrap)
        self._cards_flow.setContentsMargins(0, 0, 0, 0)
        self._cards_flow.setSpacing(12)
        self._cards_flow.addStretch()

        # Placeholder when no data yet
        self._no_data_lbl = QLabel("No defects detected yet this session.")
        self._no_data_lbl.setStyleSheet("color:#2e3060; font-size:13px;")
        self._cards_flow.insertWidget(0, self._no_data_lbl)
        self._report_body_lay.addWidget(self._cards_wrap)

        # ── Section: Captured images ──
        sec2 = QLabel("CAPTURED DEFECT IMAGES")
        sec2.setStyleSheet("color:#2e3060; font-size:10px; font-weight:bold; letter-spacing:2px;")
        self._report_body_lay.addWidget(sec2)

        gallery_scroll = QScrollArea()
        gallery_scroll.setFixedHeight(210)
        gallery_scroll.setWidgetResizable(True)
        gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gallery_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        self._gallery_widget = QWidget()
        self._gallery_widget.setStyleSheet("background:transparent;")
        self._gallery_lay = QHBoxLayout(self._gallery_widget)
        self._gallery_lay.setContentsMargins(0, 0, 0, 0)
        self._gallery_lay.setSpacing(12)
        self._gallery_lay.addStretch()
        gallery_scroll.setWidget(self._gallery_widget)
        self._report_body_lay.addWidget(gallery_scroll)

        self._report_body_lay.addStretch()
        scroll.setWidget(report_body)
        report_outer.addWidget(scroll)

        self.stack.addWidget(report_page)

        # ── Timers ────────────────────────────────────────────────
        self.display_timer = QTimer()
        self.display_timer.setInterval(50)
        self.display_timer.timeout.connect(self._refresh_display)

        self._live_timer = QTimer()
        self._live_timer.setInterval(800)
        self._live_timer.timeout.connect(self._pulse_live)

        # ── Start ─────────────────────────────────────────────────
        self.cam_thread.start()
        self.defect_thread.start()
        self.stitch_thread.start()
        self.display_timer.start()
        self._live_timer.start()

        print(f"[App] Defect model : {DEFECT_MODEL_PATH}")
        print(f"[App] Stitch model : {STITCH_MODEL_PATH}")
        print(f"[App] Inference    : every {INFER_EVERY_N_FRAMES} frames, imgsz={INFERENCE_IMGSZ}")

    # ── Page switch ───────────────────────────────────────────────

    def _switch(self, idx):
        self.stack.setCurrentIndex(idx)
        self.btn_live.setStyleSheet(_NAV_ACTIVE if idx == 0 else _NAV_IDLE)
        self.btn_report.setStyleSheet(_NAV_ACTIVE if idx == 1 else _NAV_IDLE)

    # ── Live dot pulse ────────────────────────────────────────────

    def _pulse_live(self):
        self._live_state = not self._live_state

    # ── FPS slot ──────────────────────────────────────────────────

    @pyqtSlot(str, float)
    def _on_fps(self, model_name, fps):
        if model_name == "Defect":
            self._defect_fps = fps
        else:
            self._stitch_fps = fps

    # ── Per-class count helper ────────────────────────────────────

    def _count_by_class(self, results):
        counts = {}
        if results is None or results.boxes is None:
            return counts
        for box in results.boxes:
            name = results.names[int(box.cls[0])]
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ── Update cumulative counts + report cards ───────────────────

    def _accumulate(self, defect_counts, stitch_counts):
        """Called on each auto-capture. Adds counts to session totals."""
        did_something = False
        for name, n in {**defect_counts, **stitch_counts}.items():
            self._session_totals[name] = self._session_totals.get(name, 0) + n
            did_something = True

        if did_something:
            # Hide placeholder
            self._no_data_lbl.setVisible(False)
            # Create or update count cards
            for name, total in self._session_totals.items():
                if name in defect_counts or total > 0 and name not in stitch_counts:
                    accent = "#ff4757"
                elif name in stitch_counts:
                    accent = "#2ed573"
                else:
                    accent = "#7c6ff7"

                if name not in self._count_cards:
                    card = _CountCard(name, accent)
                    # Insert before the stretch
                    self._cards_flow.insertWidget(self._cards_flow.count() - 1, card)
                    self._count_cards[name] = card
                self._count_cards[name].set_count(total)

    # ── Add gallery thumbnail ─────────────────────────────────────

    def _add_gallery_item(self, frame_bgr, defect_counts, stitch_counts):
        """Add a thumbnail card to the report gallery."""
        # Build QPixmap from numpy
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)

        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        # Build summary string
        parts = [f"{n}:{c}" for n, c in {**defect_counts, **stitch_counts}.items()]
        summary = "  ".join(parts) if parts else ""

        card = _ThumbCard(pix, ts, summary)
        # Insert newest at left (before stretch)
        self._gallery_lay.insertWidget(0, card)

        # Limit gallery to 30 items
        while self._gallery_lay.count() > 31:  # 30 cards + 1 stretch
            item = self._gallery_lay.takeAt(self._gallery_lay.count() - 2)
            if item and item.widget():
                item.widget().deleteLater()

    # ── Clear session ─────────────────────────────────────────────

    def _clear_session(self):
        self._session_totals.clear()
        for card in self._count_cards.values():
            card.deleteLater()
        self._count_cards.clear()
        self._no_data_lbl.setVisible(True)

        while self._gallery_lay.count() > 1:
            item = self._gallery_lay.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    # ── Main display refresh ──────────────────────────────────────

    def _refresh_display(self):
        frame = self.frame_buffer.peek()
        if frame is None:
            return

        annotated = frame.copy()
        defect_r  = self.defect_store.get()
        stitch_r  = self.stitch_store.get()
        draw_detections(annotated, defect_r, DEFECT_COLOR)
        draw_detections(annotated, stitch_r, STITCH_COLOR)
        self._capture_frame = annotated

        defect_counts = self._count_by_class(defect_r)
        stitch_counts = self._count_by_class(stitch_r)
        defect_total  = sum(defect_counts.values())

        # Auto-capture (max once per 4 s when defects present)
        if defect_total > 0:
            now = time.time()
            if now - self._last_auto_capture_t >= 4.0:
                self._last_auto_capture_t = now
                self._accumulate(defect_counts, stitch_counts)
                self._add_gallery_item(annotated, defect_counts, stitch_counts)
                ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fp  = os.path.join(self.capture_dir, f"defect_{ts}.jpg")
                cv2.imwrite(fp, annotated)

        # Only update the live preview when on live page
        if self.stack.currentIndex() != 0:
            return

        pw = self.preview.width()
        ph = self.preview.height()
        if pw < 2 or ph < 2:
            return
        fh, fw = annotated.shape[:2]
        scale  = min(pw / fw, ph / fh)
        disp_w = max(1, int(fw * scale))
        disp_h = max(1, int(fh * scale))

        small = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        # ── QPainter overlay: title + FPS + LIVE dot ──────────────
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        # Dark pill background
        p.setBrush(QBrush(QColor(7, 8, 15, 185)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(10, 10, 215, 52, 8, 8)

        # Title
        f1 = QFont("Segoe UI", 13)
        f1.setBold(True)
        p.setFont(f1)
        p.setPen(QColor("#e4e6f8"))
        p.drawText(20, 34, "NeedleEye")

        # FPS
        f2 = QFont("Segoe UI", 9)
        p.setFont(f2)
        p.setPen(QColor("#7c6ff7"))
        p.drawText(20, 52, f"D: {self._defect_fps:.1f} fps    S: {self._stitch_fps:.1f} fps")

        # LIVE dot
        live_col = QColor("#2ed573") if self._live_state else QColor("#1a3d2a")
        f3 = QFont("Segoe UI", 9)
        f3.setBold(True)
        p.setFont(f3)
        p.setPen(live_col)
        p.drawText(disp_w - 68, 30, "\u25cf LIVE")

        p.end()
        self.preview.setPixmap(pixmap)

    # ── Key events ────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self._manual_capture()

    def _manual_capture(self):
        if self._capture_frame is not None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(self.capture_dir, f"manual_{ts}.jpg")
            cv2.imwrite(fp, self._capture_frame)
            print(f"[Info] Manual capture → {fp}")

    def closeEvent(self, event):
        self.display_timer.stop()
        self._live_timer.stop()
        self.cam_thread.stop()
        self.defect_thread.stop()
        self.stitch_thread.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
