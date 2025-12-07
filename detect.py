#!/usr/bin/env python3
"""Minimal detection wrapper for Ultralytics YOLOv8.

Examples:
  .\.venv\Scripts\python.exe detect.py --weights yolov8n.pt --source test/images --conf 0.25 --device cpu
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Run YOLOv8 detection')
    p.add_argument('--weights', '-w', default='yolov8n.pt')
    p.add_argument('--source', '-s', default='test/images')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--save', action='store_true', help='Save predictions to runs/detect')
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    model = YOLO(args.weights)
    print(f"Running detect: weights={args.weights} source={src} conf={args.conf} device={args.device}")
    results = model.predict(source=str(src), conf=args.conf, imgsz=args.imgsz, device=args.device, save=args.save)
    print('Done. Results length:', len(results))


if __name__ == '__main__':
    main()
