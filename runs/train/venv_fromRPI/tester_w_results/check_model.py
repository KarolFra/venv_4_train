#!/usr/bin/env python3
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

from flask import Flask, jsonify, request, send_file, abort
import cv2
import numpy as np
import torch

# ---- optional CPU/RAM debug ----
try:
    import psutil
except Exception:  # psutil optional
    psutil = None

# --- dopisz do check_model.py ---
from flask import Flask, request, jsonify, send_file
from pathlib import Path
from datetime import datetime
import cv2, numpy as np, os
from ultralytics import YOLO

app = Flask(__name__)
MODEL_PATH = "/home/pi/ConveyorBelt-mqtt/Venv/models/runs_yolo/pcb-detect/weights/best.pt"
model = YOLO(MODEL_PATH)

def annotate_and_save(in_path: Path, out_path: Path, conf=0.25, iou=0.45, jpg_q=85):
    img = cv2.imread(str(in_path))
    if img is None:
        return {"input": str(in_path), "error": "cannot_read"}
    res = model.predict(source=img, conf=conf, iou=iou, verbose=False)[0]
    boxes = res.boxes
    dets = []
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        names = res.names or model.names
        for (x1,y1,x2,y2),c,k in zip(xyxy, confs, clses):
            dets.append({
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "confidence": float(c),
                "label": (names[int(k)] if isinstance(names, (list,dict)) else f"class_{int(k)}")
            })
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(img, f"{dets[-1]['label']} {dets[-1]['confidence']:.2f}",
                        (int(x1), max(0,int(y1)-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
    return {"input": str(in_path), "output": str(out_path), "count": len(dets), "detections": dets}

@app.get("/check_folder")
def check_folder():
    root = request.args.get("root")
    if not root:
        return jsonify({"error":"missing 'root' param"}), 400
    root_path = Path(root)
    if not root_path.exists():
        return jsonify({"error":"root_not_found", "path": str(root_path)}), 404

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("/home/pi/ConveyorBelt-mqtt/Venv/results/check_model")/ts

    exts = {".jpg",".jpeg",".png",".bmp"}
    items = []
    for p in root_path.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            rel = p.relative_to(root_path)
            out = results_dir/rel.parent/f"annot_{p.name}"
            items.append(annotate_and_save(p, out))

    return jsonify({"results_dir": str(results_dir), "items": items})

# (opcjonalnie) serwuj plik do podglądu
@app.get("/file")
def file_dl():
    path = request.args.get("path")
    if not path or not Path(path).exists():
        return jsonify({"error":"not_found", "path": path}), 404
    return send_file(path, mimetype="image/jpeg")
# --- koniec dopisku ---


# ---- config / paths ----
BASE = Path(__file__).resolve().parent
RESULTS_DIR = (BASE / "results" / "check_model").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model resolution order (first existing wins):
MODEL_CANDIDATES = [
    Path(os.getenv("YOLO_MODEL_PATH", "")),
    BASE / "models" / "best_model.pt",
    BASE / "models" / "runs_yolo" / "pcb-detect" / "weights" / "best.pt",
    BASE / "yolo11n.pt",
]

YOLO_CONF = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))

# ---- lazy load model (Ultralytics) ----
from ultralytics import YOLO
_model = None
_model_path: Optional[Path] = None

def load_model() -> Tuple[YOLO, Optional[Path], str]:
    global _model, _model_path
    if _model is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return _model, _model_path, device

    chosen: Optional[Path] = None
    for c in MODEL_CANDIDATES:
        if c and Path(c).is_file():
            chosen = Path(c)
            break
    if chosen is None:
        raise FileNotFoundError("No YOLO weights found. Set YOLO_MODEL_PATH or place best.pt.")

    model = YOLO(str(chosen))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        try:
            model.model.to(device)
        except Exception:
            pass

    _model = model
    _model_path = chosen
    return _model, _model_path, device

# ---- detection ----
def run_on_image(img_path: Path, out_dir: Path) -> Tuple[str, List[dict]]:
    if not img_path.is_file():
        raise FileNotFoundError(str(img_path))

    model, weights_path, device = load_model()

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # inference
    t0 = time.perf_counter()
    results = model.predict(source=img, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    dt = time.perf_counter() - t0

    dets: List[dict] = []
    if results and results[0].boxes is not None:
        r = results[0]
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy()
        names = r.names or getattr(model, "names", None)

        for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clses):
            label = None
            if isinstance(names, dict):
                label = names.get(int(ci))
            elif isinstance(names, list) and 0 <= int(ci) < len(names):
                label = names[int(ci)]
            dets.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "confidence": float(cf),
                "label": label or f"class_{int(ci)}",
            })
            # draw
            p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            txt = f"{label or int(ci)} {float(cf):.2f}"
            cv2.putText(img, txt, (p1[0], max(0, p1[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # stamp
    cv2.putText(img, f"{Path(weights_path).name if weights_path else 'model'} | {dt*1000:.1f} ms",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"annot_{img_path.stem}.jpg"
    out_path = out_dir / out_name
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"Failed to write: {out_path}")

    return str(out_path), dets

# ---- Flask app ----
app = Flask(__name__)

@app.get("/health")
def health():
    try:
        _, w, device = load_model()
        model_path = str(w) if w else None
    except Exception as e:
        model_path = f"ERROR: {e}"
        device = "unknown"

    cpu = psutil.cpu_percent(interval=0.2) if psutil else None
    mem = psutil.virtual_memory().percent if psutil else None
    return jsonify({
        "model_path": model_path,
        "device": device,
        "cpu_percent": cpu,
        "mem_percent": mem,
    })

@app.get("/check")
def check():
    """
    /check?files=/abs/p1.jpg,/abs/p2.jpg
    (max 10 plików). Zwraca JSON + ścieżki do pobrania.
    """
    files_param = request.args.get("files", "").strip()
    if not files_param:
        return abort(400, "Provide ?files=/path/a.jpg,/path/b.jpg")

    files = [Path(p.strip()) for p in files_param.split(",") if p.strip()]
    files = files[:10]

    stamp_dir = RESULTS_DIR / time.strftime("%Y%m%d_%H%M%S")
    stamp_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for p in files:
        try:
            out_path, dets = run_on_image(p, stamp_dir)
            out.append({
                "input": str(p),
                "output": out_path,
                "download": f"/file?path={out_path}",
                "detections": dets,
                "count": len(dets),
            })
        except Exception as e:
            out.append({
                "input": str(p),
                "error": str(e),
            })

    return jsonify({
        "results_dir": str(stamp_dir),
        "items": out
    })

@app.get("/file")
def get_file():
    p = request.args.get("path")
    if not p:
        return abort(400, "path required")
    path = Path(p)
    if not path.is_file():
        return abort(404, "not found")
    # bezpieczeństwo: zezwól tylko w obrębie RESULTS_DIR
    try:
        path.resolve().relative_to(RESULTS_DIR.resolve())
    except Exception:
        return abort(403, "forbidden")
    return send_file(str(path), mimetype="image/jpeg")

if __name__ == "__main__":
    # python check_model.py (fallback bez FLASK_APP)
    app.run(host="0.0.0.0", port=5001, debug=True)
