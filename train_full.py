"""Orchestrate training runs for multiple YOLOv8 model sizes with sensible defaults.

Usage examples:
  python train_full.py --models yolov8n.pt yolov8s.pt --epochs 50 --imgsz 640
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import yaml


def make_data_yaml(data_arg, data_dir):
    target = Path('data_from_dir.yaml')
    if data_dir is None:
        return data_arg
    data_dir = Path(data_dir)
    d = {}
    try:
        if Path(data_arg).exists():
            with open(data_arg, 'r', encoding='utf8') as f:
                d = yaml.safe_load(f) or {}
    except Exception:
        d = {}
    train_p = data_dir / 'train' / 'images'
    val_p = data_dir / 'valid' / 'images'
    test_p = data_dir / 'test' / 'images'
    if not train_p.exists():
        train_p = data_dir / 'images'
    d['train'] = str(train_p)
    d['val'] = str(val_p)
    d['test'] = str(test_p)
    with open(target, 'w', encoding='utf8') as f:
        yaml.safe_dump(d, f)
    return str(target)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['yolov8n.pt'], help='List of .pt models to train/finetune')
    p.add_argument('--data', default='data.yaml')
    p.add_argument('--data_dir', default=None, help='Optional: path to a dataset folder (will generate a temporary data yaml pointing into this folder)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--hyp', default='hyp.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    data_arg = args.data
    if args.data_dir:
        data_arg = make_data_yaml(args.data, args.data_dir)
    for m in args.models:
        print('Training model', m)
        model = YOLO(m)
        model.train(data=data_arg, epochs=args.epochs, imgsz=args.imgsz, device=args.device, hyp=args.hyp)


if __name__ == '__main__':
    main()
