import cv2
import threading
import time
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────
MODEL_PATH = "best.pt"
DEVICE = 0
IMGSZ = 640

# ── Shared Frame Buffer ────────────────────────────────
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

# ── Camera Thread (REFERENCE STYLE) ────────────────────
class CameraThread(threading.Thread):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(DEVICE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.buffer.put(frame)

        cap.release()

    def stop(self):
        self.running = False

# ── Inference Thread ───────────────────────────────────
class InferenceThread(threading.Thread):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer
        self.running = True

        self.model = YOLO(MODEL_PATH)
        self.model.to("cpu")   # ✅ force CPU

    def run(self):
        while self.running:
            frame = self.buffer.get()
            if frame is None:
                time.sleep(0.01)
                continue

            # ✅ YOLO handles resize internally
            results = self.model(
                frame,
                imgsz=IMGSZ,
                device="cpu",
                verbose=False
            )

            annotated = results[0].plot()

            cv2.imshow("YOLOv8 CPU Live", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.running = False

# ── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    buffer = FrameBuffer()

    cam_thread = CameraThread(buffer)
    infer_thread = InferenceThread(buffer)

    cam_thread.start()
    infer_thread.start()

    infer_thread.join()
    cam_thread.stop()