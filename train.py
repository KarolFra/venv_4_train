#!/usr/bin/env python3
"""Minimal training wrapper for Ultralytics YOLOv8.

Usage examples (PowerShell):
  .\.venv\Scripts\python.exe train.py --model yolov8n.pt --data data.yaml --epochs 20 --device cpu

On Raspberry Pi, prefer `yolov8n.pt` and `--device cpu`.
"""
import argparse
import os
from ultralytics import YOLO
from pathlib import Path
import yaml
import ultralytics
print(ultralytics.__version__)

def make_data_yaml(data_arg, data_dir):
    """If data_dir is provided, create a temporary YAML that points to its train/valid/test images.
    Reuse nc/names from existing data_arg if present. Paths are written relative to CWD when possible."""
    target = Path('data.yaml')
    if data_dir is None:
        return data_arg
    data_dir = Path(data_dir)

    def to_rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(Path.cwd()))
        except Exception:
            return str(p)

    d = {}
    # try to load existing data.yaml to preserve nc/names
    try:
        if Path(data_arg).exists():
            with open(data_arg, 'r', encoding='utf8') as f:
                d = yaml.safe_load(f) or {}
    except Exception:
        d = {}
    # construct sensible defaults
    train_p = data_dir / 'train' / 'images'
    val_p = data_dir / 'valid' / 'images'
    test_p = data_dir / 'test' / 'images'
    if not train_p.exists():
        # fallback to images/ directly
        train_p = data_dir / 'images'
    if not val_p.exists():
        val_p = data_dir / 'valid' / 'images'
    if not test_p.exists():
        test_p = data_dir / 'test' / 'images'
    # Prefer relative paths w.r.t current working directory
    d['train'] = to_rel(train_p)
    d['val'] = to_rel(val_p)
    d['test'] = to_rel(test_p)
    # keep existing nc/names if present, otherwise leave them out
    with open(target, 'w', encoding='utf8') as f:
        yaml.safe_dump(d, f)
    return str(target)


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv11n on a dataset')
    p.add_argument('--model', '-m', default='yolo11n.pt', help='Weights to start from (yolov8n.pt, yolov8s.pt, ... or a custom .pt)')
    p.add_argument('--data', '-d', default='data.yaml', help='Path to dataset YAML')
    p.add_argument('--data_dir', default=None, help='Optional: path to a dataset folder (will generate a temporary data yaml pointing into this folder)')
    p.add_argument('--epochs', '-e', type=int, default=120)
    p.add_argument('--imgsz', type=int, default=640, help='Image size')
    p.add_argument('--batch', type=int, default=None, help='Batch size (None uses default)')
    p.add_argument('--workers', type=int, default=None, help='DataLoader workers (reduce to 0 on Windows/low RAM)')
    p.add_argument('--cache', default=None, help="Cache images to 'ram' or 'disk' (or 'False' to disable)")
    p.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    p.add_argument('--name', default=None, help='Experiment name')
    p.add_argument('--project', default='runs/train', help='Base project directory for saving runs')
    p.add_argument('--save_period', type=int, default=5, help='Save checkpoint every N epochs (Ultralytics default if None)')
    p.add_argument('--resume', action='store_true', help='Resume the most recent run (or the run given by --name)')
    p.add_argument('--single-cls', action='store_true', help='Treat all classes as a single class during training')
    p.add_argument('--classes', nargs='+', type=int, default=None, help='Train on a subset of classes (by index)')
    return p.parse_args()


def main():
    args = parse_args()
    # Always prefer local data.yaml in the current folder if it exists.
    # Only generate a temporary data_from_dir.yaml when no local data file exists.
    data_arg = args.data
    if not Path(data_arg).exists() and args.data_dir:
        data_arg = make_data_yaml(args.data, args.data_dir)
    elif Path(data_arg).exists() and args.data_dir:
        print(f"Found local '{data_arg}'. Ignoring --data_dir and using local dataset description.")
    print(f"Starting training: model={args.model} data={data_arg} epochs={args.epochs} imgsz={args.imgsz} device={args.device}")
    model = YOLO(args.model)
    train_kwargs = dict(data=data_arg, epochs=args.epochs, imgsz=args.imgsz, project=args.project)
    if args.batch:
        train_kwargs['batch'] = args.batch
    if args.workers is not None:
        train_kwargs['workers'] = args.workers
    if args.cache is not None:
        # Handle common string booleans
        cache_val = args.cache
        if isinstance(cache_val, str) and cache_val.lower() in ('false', '0', 'no', 'none'):
            cache_val = False
        train_kwargs['cache'] = cache_val
    if args.name:
        train_kwargs['name'] = args.name
    if args.save_period is not None:
        train_kwargs['save_period'] = args.save_period
    if args.single_cls:
        train_kwargs['single_cls'] = True
    if args.classes is not None:
        train_kwargs['classes'] = args.classes

    # Resume logic: if --resume is set, try to resume the latest (or named) run
    if args.resume:
        # If name was not provided, attempt to auto-detect the latest run under project
        if not args.name:
            try:
                proj = Path(args.project)
                if proj.exists():
                    # sort directories by last modified, look for one that contains weights/last.pt
                    run_dirs = sorted([p for p in proj.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
                    for rd in run_dirs:
                        if (rd / 'weights' / 'last.pt').exists():
                            args.name = rd.name
                            break
            except Exception:
                pass
        if args.name:
            print(f"Resuming run: project={args.project} name={args.name}")
            train_kwargs['name'] = args.name
        else:
            print("Warning: --resume requested but no existing run found; starting a new run.")
        train_kwargs['resume'] = True

    # If CPU requested, ensure CUDA is disabled even when resuming
    if str(args.device).lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Include device in overrides to ensure it is respected when resuming
    train_kwargs['device'] = args.device
    # Ultralytics will print progress and save to runs/train/{name}
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
