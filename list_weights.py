"""List any saved weights under runs/*/weights and common locations.

Run: python list_weights.py
"""
from pathlib import Path

RUNS = Path('runs')


def find_weights():
    found = []
    if not RUNS.exists():
        print('No runs/ directory found')
        return found
    for p in RUNS.rglob('weights'):
        for f in p.iterdir():
            if f.suffix in ('.pt', '.onnx'):
                found.append(f)
    # Also check project root
    for f in Path('.').iterdir():
        if f.suffix in ('.pt', '.onnx'):
            found.append(f)
    return found


if __name__ == '__main__':
    files = find_weights()
    if not files:
        print('No weight files found')
    else:
        print('Found weight files:')
        for f in files:
            print('-', f)
