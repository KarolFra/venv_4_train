#!/usr/bin/env python3
r"""Validate and optionally fix YOLO annotation label class indices.

Usage examples:
    python validate_labels.py --data_dir C:/Users/huenk/Downloads/20251019yolo_pcbb
    python validate_labels.py --data_dir ... --fix-decrement
    python validate_labels.py --data_dir ... --remove-invalid --nc 9
    python validate_labels.py --data_dir ... --update-data-nc

This script is safe by default: it only reports problems. If you pass a fix flag
it will create a backup of each labels folder before modifying files.
"""
import argparse
from pathlib import Path
import shutil
import time
import yaml
from collections import Counter


def find_label_files(data_dir: Path):
    # look for train/labels, valid/labels, test/labels, or any labels/* .txt recursively
    candidates = []
    for sub in ('train', 'valid', 'test'):
        p = data_dir / sub / 'labels'
        if p.exists():
            candidates.append(p)
    # fallback: search recursively for .txt files that look like labels
    if not candidates:
        for p in data_dir.rglob('labels'):
            if p.is_dir():
                candidates.append(p)
    return candidates


def scan_labels(label_dir: Path):
    files = list(label_dir.glob('*.txt'))
    max_class = -1
    class_counts = Counter()
    invalid_files = []
    total_lines = 0
    for f in files:
        try:
            text = f.read_text(encoding='utf8').strip()
        except Exception:
            invalid_files.append((f, 'read error'))
            continue
        if not text:
            continue
        for i, line in enumerate(text.splitlines()):
            parts = line.split()
            if not parts:
                continue
            try:
                c = int(float(parts[0]))
            except Exception:
                invalid_files.append((f, f'bad class value on line {i+1}'))
                continue
            total_lines += 1
            class_counts[c] += 1
            if c > max_class:
                max_class = c
    return files, max_class, class_counts, invalid_files, total_lines


def backup_dir(p: Path):
    stamp = time.strftime('%Y%m%dT%H%M%S')
    dest = p.parent / (p.name + '_backup_' + stamp)
    shutil.copytree(p, dest)
    return dest


def fix_decrement(label_dir: Path):
    # subtract 1 from class id in every label line (useful if labels are 1-based)
    backup = backup_dir(label_dir)
    for f in label_dir.glob('*.txt'):
        lines = f.read_text(encoding='utf8').splitlines()
        out = []
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            parts = line.split()
            try:
                c = int(float(parts[0])) - 1
                if c < 0:
                    # keep original and mark
                    out.append(line)
                else:
                    parts[0] = str(c)
                    out.append(' '.join(parts))
            except Exception:
                out.append(line)
        f.write_text('\n'.join(out), encoding='utf8')
    return backup


def remove_invalid_lines(label_dir: Path, nc: int):
    backup = backup_dir(label_dir)
    removed = 0
    for f in label_dir.glob('*.txt'):
        lines = f.read_text(encoding='utf8').splitlines()
        out = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            try:
                c = int(float(parts[0]))
            except Exception:
                continue
            if 0 <= c < nc:
                out.append(' '.join(parts))
            else:
                removed += 1
        f.write_text('\n'.join(out), encoding='utf8')
    return backup, removed


def update_data_yaml(data_yaml_path: Path, new_nc: int):
    if not data_yaml_path.exists():
        print('data.yaml not found at', data_yaml_path)
        return False
    with open(data_yaml_path, 'r', encoding='utf8') as f:
        d = yaml.safe_load(f) or {}
    d['nc'] = int(new_nc)
    with open(data_yaml_path, 'w', encoding='utf8') as f:
        yaml.safe_dump(d, f)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--fix-decrement', action='store_true', help='Subtract 1 from all class ids (useful for 1-based labels)')
    p.add_argument('--remove-invalid', action='store_true', help='Remove label lines with class ids outside [0, nc-1]')
    p.add_argument('--nc', type=int, default=None, help='Number of classes expected (overrides data.yaml nc)')
    p.add_argument('--update-data-nc', action='store_true', help='Update data.yaml nc to match detected max class + 1')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print('Data dir not found:', data_dir)
        return

    # try to read data.yaml for nc
    data_yaml = Path('data.yaml')
    detected_nc = None
    if data_yaml.exists():
        try:
            with open(data_yaml, 'r', encoding='utf8') as f:
                dy = yaml.safe_load(f) or {}
                detected_nc = dy.get('nc')
        except Exception:
            detected_nc = None
    if args.nc is not None:
        detected_nc = args.nc

    label_dirs = find_label_files(data_dir)
    if not label_dirs:
        print('No label directories found under', data_dir)
        return

    overall_max = -1
    overall_counts = Counter()
    overall_invalid_files = []
    total_lines = 0

    for ld in label_dirs:
        print('\nScanning label dir:', ld)
        files, max_class, counts, invalid_files, lines = scan_labels(ld)
        print(f'  files: {len(files)} lines: {lines} max_class: {max_class}')
        print('  class counts:', dict(sorted(counts.items())))
        if invalid_files:
            print('  invalid files (read/parsing errors):')
            for f, reason in invalid_files:
                print('   -', f, reason)
        overall_counts.update(counts)
        overall_invalid_files.extend(invalid_files)
        total_lines += lines
        if max_class > overall_max:
            overall_max = max_class

    print('\nSummary:')
    print('  total labeled lines:', total_lines)
    print('  overall max class id:', overall_max)
    print('  overall class distribution:', dict(sorted(overall_counts.items())))
    if detected_nc is not None:
        print('  expected nc:', detected_nc)
        if overall_max >= detected_nc:
            print('  PROBLEM: found class id >= nc -> some labels exceed expected range')

    # perform fixes if requested
    if args.fix_decrement:
        print('\nApplying fix: decrementing class ids by 1 in each labels folder')
        for ld in label_dirs:
            backup = fix_decrement(ld)
            print('  backup created at', backup)
        print('Fix applied. Re-run this script to verify results.')

    if args.remove_invalid:
        if detected_nc is None:
            print('Please supply --nc or ensure data.yaml has nc to use --remove-invalid')
        else:
            print(f'\nRemoving invalid label lines with class >= {detected_nc}')
            total_removed = 0
            for ld in label_dirs:
                backup, removed = remove_invalid_lines(ld, detected_nc)
                print('  backup created at', backup, 'removed lines:', removed)
                total_removed += removed
            print('Total removed lines:', total_removed)

    if args.update_data_nc:
        if overall_max >= 0:
            new_nc = overall_max + 1
            ok = update_data_yaml(data_yaml, new_nc)
            if ok:
                print('data.yaml updated with nc =', new_nc)
            else:
                print('Failed to update data.yaml')
        else:
            print('No labels found to determine nc')

    print('\nDone.')


if __name__ == '__main__':
    main()
