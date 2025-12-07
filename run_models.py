#!/usr/bin/env python3
"""Run detection with multiple YOLOv8 model sizes (nano -> xlarge).

This script runs inference on each model listed and reports inference time. Use for benchmarking or verifying models.
"""
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_NAMES = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']


def run_on_image(model_name, image_path, device='cpu'):
    model = YOLO(model_name)
    start = time.time()
    res = model.predict(source=str(image_path), device=device, imgsz=640)
    elapsed = time.time() - start
    return elapsed, res


def main():
    img = Path('test/images')
    if not img.exists():
        print('No test images found at test/images. Exiting.')
        return
    sample = next(img.glob('*.*'))
    print('Using sample image:', sample)
    for m in MODEL_NAMES:
        print('Running', m)
        t, _ = run_on_image(m, sample)
        print(f'{m} elapsed {t:.3f}s')


if __name__ == '__main__':
    main()
