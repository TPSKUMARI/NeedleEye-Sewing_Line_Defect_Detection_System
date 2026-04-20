"""
export_model.py  –  One-time script to convert .pt → ONNX (or OpenVINO IR)
Run ONCE on the Geekom A5 (or any machine with the model file):

  # Export defect model
  python export_model.py --model best.pt

  # Export stitch model
  python export_model.py --model stitch.pt

  # Use OpenVINO IR instead (fastest on AMD/Intel CPUs with AVX2)
  python export_model.py --model best.pt --format openvino

After exporting, live_prediction_yolov8.py will automatically detect and
use best.onnx / stitch.onnx (or best_openvino_model/ dir) on next launch.
"""

import argparse
import os
from ultralytics import YOLO

IMGSZ   = 640    # must match INFERENCE_IMGSZ in live_prediction_yolov8.py
OPSET   = 12     # ONNX opset – 12 is widely compatible

def export(model_path: str, fmt: str, imgsz: int):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    print(f"[Export] Loading {model_path} ...")
    model = YOLO(model_path)

    print(f"[Export] Exporting → format={fmt}  imgsz={imgsz} ...")
    path = model.export(
        format=fmt,
        imgsz=imgsz,
        opset=OPSET,
        simplify=True,   # ONNX simplification reduces graph complexity
        dynamic=False,   # fixed batch=1 is faster for inference
    )
    print(f"[Export] ✓ Saved to: {path}")
    print()
    print("Update live_prediction_yolov8.py if needed:")
    print(f"  DEFECT_MODEL_PATH = \"{os.path.basename(str(path))}\"")
    print(f"  (or STITCH_MODEL_PATH)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model for CPU inference")
    parser.add_argument("--model",  default="best.pt",
                        help="Path to .pt model file (default: best.pt)")
    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "openvino"],
                        help="Export format: onnx (default) or openvino")
    parser.add_argument("--imgsz",  type=int, default=IMGSZ,
                        help=f"Inference image size (default: {IMGSZ})")
    args = parser.parse_args()

    export(args.model, args.format, args.imgsz)
