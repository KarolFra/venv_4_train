"""Train a YOLOv8 model and export best.pt to ONNX when done.

This script sets an explicit project/name to ensure outputs go to runs/train/<name>.
It will run training and then export the saved best weights to ONNX using Ultralytics export.

Usage:
  python train_and_export.py --model yolov8n.pt --data data.yaml --epochs 50 --name myrun
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import time
import yaml


def make_data_yaml(data_arg, data_dir):
    """Build a temporary data.yaml that points into data_dir.

    Priority for metadata (nc, names, etc.):
    1) data_dir/data.yaml if present
    2) the provided data_arg if it exists
    Otherwise, only paths are written.
    """
    target = Path('data_from_dir.yaml')
    if data_dir is None:
        return data_arg
    data_dir = Path(data_dir)
    d = {}
    # Prefer dataset's own data.yaml when present
    ds_yaml = data_dir / 'data.yaml'
    try:
        if ds_yaml.exists():
            with open(ds_yaml, 'r', encoding='utf8') as f:
                d = yaml.safe_load(f) or {}
        elif Path(data_arg).exists():
            with open(data_arg, 'r', encoding='utf8') as f:
                d = yaml.safe_load(f) or {}
    except Exception:
        d = {}
    # Point paths into the provided data_dir
    train_p = data_dir / 'train' / 'images'
    val_p = data_dir / 'valid' / 'images'
    test_p = data_dir / 'test' / 'images'
    if not train_p.exists():
        train_p = data_dir / 'images'
    d['train'] = str(train_p)
    d['val'] = str(val_p)
    d['test'] = str(test_p)
    # If names exists but nc is missing or inconsistent, align nc to names length
    if 'names' in d and isinstance(d['names'], list):
        d['nc'] = len(d['names'])
    with open(target, 'w', encoding='utf8') as f:
        yaml.safe_dump(d, f)
    return str(target)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yolov8n.pt')
    p.add_argument('--data', default='data.yaml')
    p.add_argument('--data_dir', default=None, help='Optional: path to a dataset folder (will generate a temporary data yaml pointing into this folder)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--name', default='myrun')
    p.add_argument('--resume', action='store_true', help='Resume from last run with same name')
    p.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every N epochs (default -1 uses ultralytics default)')
    p.add_argument('--export_opset', type=int, default=14)
    return p.parse_args()


def main():
    args = parse_args()
    data_arg = args.data
    if getattr(args, 'data_dir', None):
        data_arg = make_data_yaml(args.data, args.data_dir)
    save_dir = Path('runs') / 'train' / args.name
    print('Will save to', save_dir)
    model = YOLO(args.model)
    train_kwargs = dict(data=data_arg, epochs=args.epochs, imgsz=args.imgsz, device=args.device, name=args.name, project='runs/train')
    if args.resume:
        train_kwargs['resume'] = True
    if args.save_period is not None and args.save_period >= 0:
        train_kwargs['save_period'] = args.save_period
    model.train(**train_kwargs)
    # after training, look for best.pt
    best = save_dir / 'weights' / 'best.pt'
    last = save_dir / 'weights' / 'last.pt'
    chosen = None
    if best.exists():
        chosen = best
    elif last.exists():
        chosen = last
    else:
        print('No best.pt or last.pt found in', save_dir)
        return
    print('Found checkpoint:', chosen)
    # export
    onnx_out = save_dir / (chosen.stem + '.onnx')
    print('Exporting to', onnx_out)
    model = YOLO(str(chosen))
    # Ultralytics 8.1.0 does not accept 'save_path'; exporter writes to run folder
    model.export(format='onnx', opset=args.export_opset)
    print('Export complete (see run folder for .onnx):', onnx_out)


if __name__ == '__main__':
    main()
