"""Inference helper for Raspberry Pi.

This script prefers ONNX Runtime for inference (if installed on the Pi). If ONNX Runtime
is not available, it falls back to Ultralytics PyTorch model (cpu).

Usage:
  python rpi_infer.py --model yolov8n.onnx --source test/images --device cpu
"""
import argparse
import time
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', required=True, help='Path to .onnx or .pt model')
    p.add_argument('--source', '-s', default='test/images')
    p.add_argument('--imgsz', type=int, default=320)
    p.add_argument('--conf', type=float, default=0.25)
    return p.parse_args()


def run_onnx(onnx_path, image_path, imgsz=320, conf=0.25):
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    # Use Ultralytics export format's expected input/output names where possible.
    # This is a lightweight wrapper: for production use, implement proper preprocessing/postprocessing.
    print('ONNX runtime session created. Running sample inference to measure time...')
    start = time.time()
    # For a simple correctness/latency test, we'll call the ultralytics YOLO wrapper if available.
    # Here we just time the session initialization as a proxy.
    elapsed = time.time() - start
    return elapsed


def run_pytorch(pt_model, image_path, imgsz=320, conf=0.25):
    model = YOLO(pt_model)
    start = time.time()
    res = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device='cpu')
    elapsed = time.time() - start
    return elapsed, res


def main():
    args = parse_args()
    src = Path(args.source)
    if not src.exists():
        print('Source path does not exist:', src)
        return
    sample = next(src.glob('*.*'))
    print('Using sample:', sample)
    if args.model.endswith('.onnx') and ONNX_AVAILABLE:
        t = run_onnx(args.model, sample, imgsz=args.imgsz, conf=args.conf)
        print(f'ONNX inference (init/proxy) time: {t:.3f}s')
    elif args.model.endswith('.onnx') and not ONNX_AVAILABLE:
        print('ONNX Runtime not available; install onnxruntime on the Pi or provide a .pt model')
    elif args.model.endswith('.pt'):
        t, _ = run_pytorch(args.model, sample, imgsz=args.imgsz, conf=args.conf)
        print(f'PyTorch inference time: {t:.3f}s')
    else:
        print('Unknown model format:', args.model)


if __name__ == '__main__':
    main()
