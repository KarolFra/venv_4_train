"""Export YOLOv8 PyTorch weights to ONNX and optionally quantize (post-training).

Usage:
  python export_onnx.py --weights yolov8n.pt --output yolov8n.onnx --opset 14

Notes:
- For Raspberry Pi 5, ONNX + ONNX Runtime (with CPU optimizations) is a good path.
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', '-w', required=True)
    p.add_argument('--output', '-o', required=True)
    p.add_argument('--opset', type=int, default=14)
    p.add_argument('--dynamic', action='store_true', help='Export dynamic axes')
    return p.parse_args()


def main():
    args = parse_args()
    print(f'Exporting {args.weights} -> {args.output} opset={args.opset} dynamic={args.dynamic}')
    model = YOLO(args.weights)
    model.export(format='onnx', opset=args.opset, dynamic=args.dynamic, save_path=args.output)
    print('Export complete.')


if __name__ == '__main__':
    main()
