"""
capture_tool.py  –  NeedleEye: Image Capture Tool
Standalone capture utility for collecting training images.

Features
─────────
  • Live camera preview (full-resolution, aspect-ratio preserved)
  • Label selector  – type or pick a class name before capturing
  • Camera controls – Brightness, Contrast, Saturation, Sharpness,
                      White Balance, Focus, Zoom (sliders + Hold lock)
  • Capture button / Space bar  → saves  <label>/<YYYYMMDD_HHMMSS_NNN>.jpg
  • Counter badge and recent-thumbnail strip
  • Same dark NeedleEye style as live_prediction_defect.py

Folder layout
─────────────
  captures_training/
      <label_name>/
          20250101_120000_001.jpg
          ...

Usage
─────
  python capture_tool.py
  Press  Space  or click  CAPTURE  to save.
  Press  Escape  to quit.
"""

# ── CPU thread limits ─────────────────────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import sys
import cv2
import datetime
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QLineEdit,
    QScrollArea, QSizePolicy, QGroupBox,
    QCheckBox, QComboBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui  import QImage, QPixmap, QFont, QColor, QPainter, QBrush

cv2.setNumThreads(4)
cv2.setUseOptimized(True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE      = 0
CAPTURE_W   = 1920
CAPTURE_H   = 1080
SAVE_ROOT   = "captures_training"

# Default label suggestions (edit freely)
DEFAULT_LABELS = ["defect", "no_defect", "stitch", "background"]

# Camera property map: name → (cv2_prop_id, auto_disable_prop_or_None)
PROP_MAP = {
    "Brightness":        (cv2.CAP_PROP_BRIGHTNESS,           None),
    "Contrast":          (cv2.CAP_PROP_CONTRAST,             None),
    "Saturation":        (cv2.CAP_PROP_SATURATION,           None),
    "Sharpness":         (cv2.CAP_PROP_SHARPNESS,            None),
    "White Balance":     (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, cv2.CAP_PROP_AUTO_WB),
    "Focus":             (cv2.CAP_PROP_FOCUS,                cv2.CAP_PROP_AUTOFOCUS),
    "Zoom":              (cv2.CAP_PROP_ZOOM,                 None),
}

# ── STYLESHEET ────────────────────────────────────────────────────────────────
STYLE = """
* {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #dde0f0;
}
QMainWindow, QWidget {
    background-color: #0c0d14;
}
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
    width: 12px; height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #9d93ff; }
QSlider::handle:horizontal:disabled {
    background: #2a2d44;
    border-color: #0c0d14;
}
QCheckBox {
    color: #454870;
    font-size: 10px;
    spacing: 4px;
}
QCheckBox::indicator {
    width: 13px; height: 13px;
    border-radius: 3px;
    border: 1px solid #2a2d44;
    background: #11121c;
}
QCheckBox::indicator:checked {
    background: #7c6ff7;
    border-color: #7c6ff7;
}
QLineEdit {
    background: #181a28;
    border: 1px solid #1e2035;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
    color: #e4e6f8;
}
QLineEdit:focus {
    border-color: #7c6ff7;
}
QComboBox {
    background: #181a28;
    border: 1px solid #1e2035;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #9095c0;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #181a28;
    border: 1px solid #1e2035;
    selection-background-color: #7c6ff7;
    color: #dde0f0;
}
QScrollBar:vertical {
    background: #0c0d14; width: 6px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2d44; border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollArea { border: none; }
"""

# ── CAMERA THREAD ─────────────────────────────────────────────────────────────
class CameraThread(QThread):
    frame_ready = pyqtSignal(object)   # emits numpy array

    def __init__(self):
        super().__init__()
        self._running  = False
        self._cap      = None
        self._cap_lock = threading.Lock()

    def run(self):
        self._cap = cv2.VideoCapture(DEVICE, cv2.CAP_DSHOW)
        cap = self._cap

        if not cap.isOpened():
            print(f"[Camera] Failed to open device {DEVICE}")
            for i in range(4):
                t = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if t.isOpened():
                    print(f"[Camera]   Device {i} is available")
                    t.release()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened {w}×{h} → resizing to {CAPTURE_W}×{CAPTURE_H}")

        self._running = True
        while self._running:
            with self._cap_lock:
                ret, frame = cap.read()
            if not ret:
                continue
            small = cv2.resize(frame, (CAPTURE_W, CAPTURE_H),
                               interpolation=cv2.INTER_LINEAR)
            self.frame_ready.emit(small)

        cap.release()
        self._cap = None

    def set_prop(self, prop_id: int, value: float):
        """Thread-safe camera property setter."""
        with self._cap_lock:
            if self._cap is not None:
                self._cap.set(prop_id, value)

    def stop(self):
        self._running = False
        self.wait()


# ── CAMERA CONTROL ROW ────────────────────────────────────────────────────────
class ControlRow(QWidget):
    """Slim labelled slider with value badge and Hold lock."""
    def __init__(self, name, min_val=0, max_val=100, init_val=50, cam=None):
        super().__init__()
        self.prop_name = name
        self.cam       = cam

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 3, 2, 3)
        lay.setSpacing(6)

        lbl = QLabel(name)
        lbl.setFixedWidth(96)
        lbl.setStyleSheet("color:#6870a0; font-size:11px;")
        lay.addWidget(lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(init_val)
        self.slider.setFixedHeight(18)
        lay.addWidget(self.slider, 1)

        self.val_lbl = QLabel(str(init_val))
        self.val_lbl.setFixedWidth(34)
        self.val_lbl.setAlignment(Qt.AlignCenter)
        self.val_lbl.setStyleSheet(
            "background:#1e2035; color:#9d93ff; font-size:10px;"
            "border-radius:4px; padding:1px 0;"
        )
        lay.addWidget(self.val_lbl)

        self.hold = QCheckBox("Hold")
        lay.addWidget(self.hold)

        self._hold_timer = QTimer()
        self._hold_timer.setInterval(500)
        self._hold_timer.timeout.connect(self._enforce)

        self.slider.valueChanged.connect(self._on_change)
        self.hold.toggled.connect(self._on_hold)

    def _apply(self, value):
        if self.cam is None or self.prop_name not in PROP_MAP:
            return
        prop_id, auto_id = PROP_MAP[self.prop_name]
        if auto_id is not None:
            self.cam.set_prop(auto_id, 0)
        self.cam.set_prop(prop_id, float(value))

    def _on_change(self, v):
        self.val_lbl.setText(str(v))
        if not self.hold.isChecked():
            self._apply(v)

    def _on_hold(self, checked):
        self.slider.setEnabled(not checked)
        if checked:
            self._hold_timer.start()
            self._apply(self.slider.value())
        else:
            self._hold_timer.stop()

    def _enforce(self):
        self._apply(self.slider.value())


# ── THUMBNAIL STRIP CARD ──────────────────────────────────────────────────────
class _ThumbCard(QWidget):
    def __init__(self, pixmap, label, ts_str):
        super().__init__()
        self.setFixedWidth(140)
        self.setStyleSheet(
            "background:#181a28; border-radius:8px; border:1px solid #1e2035;"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 6)
        lay.setSpacing(3)

        img = QLabel()
        img.setFixedSize(130, 78)
        img.setAlignment(Qt.AlignCenter)
        img.setStyleSheet("background:#0d0e18; border-radius:4px; border:none;")
        img.setPixmap(pixmap.scaled(130, 78, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(img)

        lbl_w = QLabel(label)
        lbl_w.setStyleSheet("color:#7c6ff7; font-size:10px; font-weight:bold;"
                            "background:transparent; border:none;")
        lbl_w.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl_w)

        ts_w = QLabel(ts_str)
        ts_w.setStyleSheet("color:#454870; font-size:9px; background:transparent; border:none;")
        ts_w.setAlignment(Qt.AlignCenter)
        lay.addWidget(ts_w)


# ── MAIN APP ──────────────────────────────────────────────────────────────────
class CaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeedleEye  ·  Capture Tool")
        self.resize(1400, 860)
        self.setMinimumSize(900, 580)

        os.makedirs(SAVE_ROOT, exist_ok=True)

        self._latest_frame = None
        self._capture_count = 0
        self._frame_lock = threading.Lock()

        # ── Camera thread ──
        self.cam = CameraThread()
        self.cam.frame_ready.connect(self._on_frame)
        self.cam.start()

        # ── Root layout: preview | right panel ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── LEFT: preview ──
        left = QWidget()
        left.setStyleSheet("background:#07080f;")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#07080f;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_lay.addWidget(self.preview, 1)

        # Thumbnail strip
        strip_outer = QWidget()
        strip_outer.setFixedHeight(132)
        strip_outer.setStyleSheet("background:#0d0e18; border-top:1px solid #1a1c2e;")
        strip_outer_lay = QVBoxLayout(strip_outer)
        strip_outer_lay.setContentsMargins(12, 8, 12, 8)
        strip_outer_lay.setSpacing(4)

        strip_hdr = QLabel("RECENT CAPTURES")
        strip_hdr.setStyleSheet(
            "color:#2e3060; font-size:9px; font-weight:bold; letter-spacing:2px;"
        )
        strip_outer_lay.addWidget(strip_hdr)

        self._strip_scroll = QScrollArea()
        self._strip_scroll.setWidgetResizable(True)
        self._strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._strip_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        self._strip_widget = QWidget()
        self._strip_widget.setStyleSheet("background:transparent;")
        self._strip_lay = QHBoxLayout(self._strip_widget)
        self._strip_lay.setContentsMargins(0, 0, 0, 0)
        self._strip_lay.setSpacing(8)
        self._strip_lay.addStretch()
        self._strip_scroll.setWidget(self._strip_widget)
        strip_outer_lay.addWidget(self._strip_scroll)
        left_lay.addWidget(strip_outer)

        root.addWidget(left, 1)

        # ── RIGHT: control panel ──
        right = QWidget()
        right.setObjectName("sidePanel")
        right.setFixedWidth(300)
        right.setStyleSheet(
            "background:#11121c; border-left:1px solid #1e2035;"
        )
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(18, 20, 18, 20)
        right_lay.setSpacing(16)

        # Title
        title = QLabel("NeedleEye")
        title.setStyleSheet(
            "font-size:18px; font-weight:bold; color:#e4e6f8; letter-spacing:1px;"
        )
        right_lay.addWidget(title)

        sub = QLabel("Capture Tool")
        sub.setStyleSheet("font-size:11px; color:#353758; margin-top:-10px;")
        right_lay.addWidget(sub)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color:#1e2035;")
        right_lay.addWidget(div)

        # ── Label section ──
        lbl_hdr = QLabel("LABEL")
        lbl_hdr.setStyleSheet(
            "color:#454870; font-size:9px; font-weight:bold; letter-spacing:2px;"
        )
        right_lay.addWidget(lbl_hdr)

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Type label name…")
        self.label_input.setText(DEFAULT_LABELS[0] if DEFAULT_LABELS else "")
        right_lay.addWidget(self.label_input)

        self.label_combo = QComboBox()
        self.label_combo.addItems(DEFAULT_LABELS)
        self.label_combo.currentTextChanged.connect(self.label_input.setText)
        right_lay.addWidget(self.label_combo)

        # ── Capture section ──
        cap_hdr = QLabel("CAPTURE")
        cap_hdr.setStyleSheet(
            "color:#454870; font-size:9px; font-weight:bold; letter-spacing:2px;"
        )
        right_lay.addWidget(cap_hdr)

        self.capture_btn = QPushButton("⬤  CAPTURE  [Space]")
        self.capture_btn.setObjectName("captureBtn")
        self.capture_btn.setFixedHeight(46)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background:#7c6ff7; color:white;
                border:none; border-radius:8px;
                font-size:14px; font-weight:bold;
                letter-spacing:1px;
            }
            QPushButton:hover { background:#9d93ff; }
            QPushButton:pressed { background:#5a50d4; }
        """)
        self.capture_btn.clicked.connect(self.capture)
        right_lay.addWidget(self.capture_btn)

        # Counter
        self.counter_lbl = QLabel("Saved: 0 images")
        self.counter_lbl.setStyleSheet(
            "color:#454870; font-size:11px; text-align:center;"
        )
        self.counter_lbl.setAlignment(Qt.AlignCenter)
        right_lay.addWidget(self.counter_lbl)

        # Save path display
        self.path_lbl = QLabel(f"→ {SAVE_ROOT}/<label>/")
        self.path_lbl.setStyleSheet(
            "color:#2e3060; font-size:9px; font-style:italic;"
        )
        self.path_lbl.setWordWrap(True)
        right_lay.addWidget(self.path_lbl)

        div2 = QFrame()
        div2.setFrameShape(QFrame.HLine)
        div2.setStyleSheet("color:#1e2035;")
        right_lay.addWidget(div2)

        # ── Camera controls ──
        cam_group = QGroupBox("CAMERA CONTROLS")
        cam_group_lay = QVBoxLayout(cam_group)
        cam_group_lay.setSpacing(2)
        cam_group_lay.setContentsMargins(4, 8, 4, 6)

        self._ctrl_rows = {}
        for name in PROP_MAP:
            row = ControlRow(name, min_val=0, max_val=100, init_val=50, cam=self.cam)
            cam_group_lay.addWidget(row)
            self._ctrl_rows[name] = row

        cam_group_scroll = QScrollArea()
        cam_group_scroll.setWidgetResizable(True)
        cam_group_scroll.setWidget(cam_group)
        cam_group_scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        right_lay.addWidget(cam_group_scroll, 1)

        # Reset button
        reset_btn = QPushButton("Reset All Controls")
        reset_btn.setStyleSheet("""
            QPushButton {
                background:#1a1c2e; color:#454870;
                border:1px solid #1e2035; border-radius:6px;
                padding:6px 0; font-size:11px;
            }
            QPushButton:hover { background:#252840; color:#7c6ff7; }
        """)
        reset_btn.clicked.connect(self._reset_controls)
        right_lay.addWidget(reset_btn)

        root.addWidget(right)

        # ── Refresh timer ──
        self._display_timer = QTimer()
        self._display_timer.setInterval(33)   # ~30 fps display
        self._display_timer.timeout.connect(self._refresh_preview)
        self._display_timer.start()

        # Flash overlay state
        self._flash_alpha = 0

    # ── Receive frame from camera thread ─────────────────────────────────────
    def _on_frame(self, frame):
        with self._frame_lock:
            self._latest_frame = frame

    # ── Refresh the preview label ─────────────────────────────────────────────
    def _refresh_preview(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return

        pw = self.preview.width()
        ph = self.preview.height()
        if pw < 2 or ph < 2:
            return

        fh, fw = frame.shape[:2]
        scale  = min(pw / fw, ph / fh)
        nw     = max(1, int(fw * scale))
        nh     = max(1, int(fh * scale))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        pixmap  = QPixmap.fromImage(img)

        # ── Overlay: label badge + flash ──
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        # Flash effect on capture
        if self._flash_alpha > 0:
            p.setBrush(QBrush(QColor(255, 255, 255, self._flash_alpha)))
            p.setPen(Qt.NoPen)
            p.drawRect(0, 0, nw, nh)
            self._flash_alpha = max(0, self._flash_alpha - 20)

        # Dark pill
        label_text = self.label_input.text().strip() or "—"
        p.setBrush(QBrush(QColor(7, 8, 15, 190)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(10, 10, 200, 56, 8, 8)

        f1 = QFont("Segoe UI", 13); f1.setBold(True)
        p.setFont(f1)
        p.setPen(QColor("#e4e6f8"))
        p.drawText(20, 34, "NeedleEye")

        f2 = QFont("Segoe UI", 9)
        p.setFont(f2)
        p.setPen(QColor("#7c6ff7"))
        p.drawText(20, 52, f"Label: {label_text}")

        # Counter badge (top right)
        badge_txt = f"{self._capture_count} saved"
        f3 = QFont("Segoe UI", 9); f3.setBold(True)
        p.setFont(f3)
        p.setPen(QColor("#2ed573"))
        p.drawText(nw - 80, 30, badge_txt)

        p.end()
        self.preview.setPixmap(pixmap)

    # ── Capture ───────────────────────────────────────────────────────────────
    def capture(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return

        label = self.label_input.text().strip()
        if not label:
            label = "unlabeled"

        # Sanitise label for use as folder name
        label_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)

        save_dir = os.path.join(SAVE_ROOT, label_safe)
        os.makedirs(save_dir, exist_ok=True)

        self._capture_count += 1
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{self._capture_count:04d}.jpg"
        path  = os.path.join(save_dir, fname)

        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[Capture] {path}")

        # Update UI
        self.counter_lbl.setText(f"Saved: {self._capture_count} images")
        self.path_lbl.setText(f"→ {save_dir}/")

        # Flash
        self._flash_alpha = 120

        # Add thumbnail
        rgb = cv2.cvtColor(
            cv2.resize(frame, (260, 156), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB
        )
        img = QImage(rgb.data, 260, 156, 3 * 260, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        ts_short = datetime.datetime.now().strftime("%H:%M:%S")
        card = _ThumbCard(pix, label_safe, ts_short)
        # Insert at the left (newest first)
        self._strip_lay.insertWidget(0, card)
        # Cap at 20 thumbnails
        while self._strip_lay.count() > 21:
            item = self._strip_lay.takeAt(self._strip_lay.count() - 2)
            if item and item.widget():
                item.widget().deleteLater()

    # ── Reset controls ────────────────────────────────────────────────────────
    def _reset_controls(self):
        for row in self._ctrl_rows.values():
            row.slider.setValue(50)
            row.hold.setChecked(False)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.capture()
        elif event.key() == Qt.Key_Escape:
            self.close()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._display_timer.stop()
        self.cam.stop()
        event.accept()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = CaptureApp()
    w.show()
    sys.exit(app.exec_())