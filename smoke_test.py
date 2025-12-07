import sys
import importlib

packages = ['ultralytics', 'torch', 'torchvision', 'numpy', 'cv2', 'PIL', 'yaml']

def check(pkg):
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, '__version__', None)
        print(f'{pkg} imported, version={ver}')
    except Exception as e:
        print(f'{pkg} import FAILED: {e}', file=sys.stderr)


if __name__ == '__main__':
    for p in packages:
        check(p)
