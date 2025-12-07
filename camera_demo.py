import argparse
import time
import platform
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def open_camera(index=0):
    if platform.system().lower().startswith('win'):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def load_models(paths):
    models = []
    for p in paths:
        try:
            m = YOLO(str(Path(p)))
            models.append((str(p), m))
        except Exception as e:
            print(f"Nie udało się wczytać modelu: {p} -> {e}")
    return models


def parse_args():
    ap = argparse.ArgumentParser(description='YOLOv8 kamera z rotacją modeli')
    ap.add_argument('--models', nargs='+', default=None, help='Ścieżki do wag (.pt/.onnx), w kolejności rotacji')
    ap.add_argument('--switch-sec', type=float, default=10.0, help='Zmiana modelu co N sekund')
    ap.add_argument('--source', type=int, default=0, help='Index kamery (domyślnie 0)')
    ap.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    ap.add_argument('--imgsz', type=int, default=640, help='Rozmiar wejściowy')
    ap.add_argument('--device', default='cpu', help='Urządzenie: cpu lub cuda:0')
    ap.add_argument('--save', default=None, help='Opcjonalna ścieżka wyjścia wideo (mp4)')
    ap.add_argument('--verbose', action='store_true', help='Szczegółowe logi Ultralytics')
    return ap.parse_args()


def main():
    args = parse_args()

    # Domyślna lista znanych wag, jeśli nie podano --models
    default_models = [
        r"runs\train\train3_connector\weights\best.pt",
        r"runs\train\train9_KICAD\weights\best.pt",
        r"runs\train\train3\weights\best.pt",
    ]
    model_paths = args.models if args.models else default_models
    model_paths = [p for p in model_paths if Path(p).exists()]
    if not model_paths:
        print('Brak modeli do wczytania. Podaj --models ścieżki do wag.')
        return

    models = load_models(model_paths)
    if not models:
        print('Nie udało się wczytać żadnego modelu.')
        return

    cap = open_camera(args.source)
    if not cap.isOpened():
        print('Nie udało się otworzyć kamery.')
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))

    idx = 0
    cur_path, model = models[idx]
    last_switch = time.time()

    prev_ts = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Strumień wideo zakończony.')
            break

        now = time.time()
        if now - last_switch >= args.switch_sec:
            idx = (idx + 1) % len(models)
            cur_path, model = models[idx]
            last_switch = now
            print(f"[Switch] Aktualny model: {Path(cur_path).name} ({idx+1}/{len(models)})")

        dt = (now - prev_ts) if prev_ts else 0.0
        prev_ts = now

        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=args.verbose)
        r = results[0]
        rgb = r.plot()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])

        if dt > 0:
            cv2.putText(bgr, f"FPS: {1/dt:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        cv2.putText(bgr, f"Model: {Path(cur_path).name}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        try:
            boxes = r.boxes
            n = int(len(boxes)) if boxes is not None else 0
            names = getattr(model, 'names', {})
            cv2.putText(bgr, f"Detekcje: {n}  conf>={args.conf}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            if n:
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy()
                order = np.argsort(-confs)[:3]
                for i, idx2 in enumerate(order):
                    cname = names.get(int(cls[idx2]), str(int(cls[idx2])))
                    cv2.putText(bgr, f"{i+1}. {cname} {confs[idx2]:.2f}", (10, 105 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2)
        except Exception:
            pass

        try:
            cv2.imshow('YOLOv8 Camera (Q=quit)', bgr)
        except cv2.error as e:
            print('OpenCV GUI error on imshow. Tip: uninstall headless build:')
            print('  .venv\\Scripts\\python.exe -m pip uninstall -y opencv-python-headless')
            print('  .venv\\Scripts\\python.exe -m pip install -U opencv-python')
            raise
        if writer:
            writer.write(bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
