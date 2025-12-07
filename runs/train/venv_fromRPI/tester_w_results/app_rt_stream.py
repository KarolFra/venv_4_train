# app_rt_stream.py
import time, subprocess, threading, collections, yaml, warnings
from pathlib import Path
from flask import Flask, Response, jsonify, render_template_string
import numpy as np
import cv2
import onnxruntime as ort
import torch
from ultralytics.utils.nms import non_max_suppression

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- KONFIG ----------
WEIGHTS = "/home/pi/ConveyorBelt-mqtt/Venv/models/weights20251015/epoch50_run_withLedON.pt"  # .pt lub .onnx
WIDTH, HEIGHT, FPS = 640, 480, 15
IMGSZ = 640
CONF_THR, IOU_THR = 0.25, 0.45
INFER_EVERY = 1
JPEG_QUALITY = 85
OUTDIR = Path("/home/pi/ConveyorBelt-mqtt/Venv/results/live_flask")
PORT = 5002
# ----------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)

# ---- utils: przygotuj ONNX jeśli mamy .pt ----
def ensure_onnx(weights_path: str, imgsz: int):
    p = Path(weights_path)
    if p.suffix.lower() == ".onnx":
        return str(p), None
    if p.suffix.lower() == ".pt":
        onnx_out = p.with_suffix(".onnx")
        need_export = (not onnx_out.exists()) or (onnx_out.stat().st_mtime < p.stat().st_mtime)
        names = None
        if need_export:
            from ultralytics import YOLO  # leniwy import
            m = YOLO(str(p))
            # zapisz nazwy klas, jeśli dostępne
            try:
                names = list(m.names.values()) if isinstance(m.names, dict) else list(m.names)
            except Exception:
                names = None
            m.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, verbose=False)
        return str(onnx_out), names
    raise ValueError(f"Nieobsługiwane rozszerzenie: {p.suffix}")

ONNX_PATH, NAMES_FROM_PT = ensure_onnx(WEIGHTS, IMGSZ)

# ---- ONNX session ----
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

# nazwy klas: z eksportu PT lub z pliku YAML obok ONNX
NAMES: list[str] = []
if NAMES_FROM_PT:
    NAMES = [str(n) for n in NAMES_FROM_PT]
else:
    yaml_path = Path(ONNX_PATH).with_suffix(".yaml")
    if yaml_path.exists():
        try:
            with yaml_path.open() as f:
                data_yaml = yaml.safe_load(f) or {}
            names = data_yaml.get("names")
            if isinstance(names, (list, tuple)):
                NAMES = [str(n) for n in names]
        except Exception:
            NAMES = []

# ---- Stan współdzielony ----
lock = threading.Lock()
last_jpeg: bytes | None = None
last_frame = None
frame_counter = 0
infer_times = collections.deque(maxlen=60)
start_time = time.time()

# ---- System ----
def read_cpu_ram():
    with open("/proc/stat") as f:
        a = list(map(int, f.readline().split()[1:8]))
    time.sleep(0.1)
    with open("/proc/stat") as f:
        b = list(map(int, f.readline().split()[1:8]))
    ta, ia = sum(a), a[3]; tb, ib = sum(b), b[3]
    cpu = 0.0 if tb == ta else (1 - (ib - ia) / max(1, (tb - ta))) * 100.0
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v, *_ = line.split()
            info[k[:-1]] = int(v)
    total = info.get("MemTotal", 1); avail = info.get("MemAvailable", 0)
    ram = (total - avail) * 100.0 / total
    return round(cpu, 1), round(ram, 1)

def spawn_libcamera():
    cmd = [
        "libcamera-vid", "--inline", "--timeout", "0",
        "--framerate", str(FPS), "--width", str(WIDTH), "--height", str(HEIGHT),
        "--codec", "mjpeg", "--output", "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

def mjpeg_reader(proc):
    buf = b""; SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    while True:
        chunk = proc.stdout.read(8192)
        if not chunk: break
        buf += chunk
        while True:
            i = buf.find(SOI)
            if i == -1:
                buf = buf[-1024:]
                break
            j = buf.find(EOI, i+2)
            if j == -1:
                break
            jpg = buf[i:j+2]
            buf = buf[j+2:]
            yield jpg

# ---- Letterbox jak w Ultralytics ----
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if new_unpad != shape[::-1]:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)

def infer_onnx(bgr):
    im0 = bgr.copy()
    letterboxed, gain, (dw, dh) = letterbox(bgr, (IMGSZ, IMGSZ))
    im = letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    im = im[None, ...]
    out = sess.run(None, {inp_name: im})[0]  # (1, N, C)
    preds = torch.from_numpy(out)
    det_list = non_max_suppression(preds, conf_thres=CONF_THR, iou_thres=IOU_THR, max_det=300, nc=len(NAMES) or 0)
    det = det_list[0] if det_list else torch.zeros((0, 6))
    return det, im0, gain, (dw, dh)

def postprocess(det, frame0, gain, pad):
    if det is None or len(det) == 0:
        return frame0
    dw, dh = pad
    h0, w0 = frame0.shape[:2]
    det_np = det.cpu().numpy()
    det_np[:, [0, 2]] -= dw
    det_np[:, [1, 3]] -= dh
    det_np[:, :4] /= max(gain, 1e-6)
    det_np[:, [0, 2]] = det_np[:, [0, 2]].clip(0, w0)
    det_np[:, [1, 3]] = det_np[:, [1, 3]].clip(0, h0)
    for x1, y1, x2, y2, conf, cls in det_np:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame0, p1, p2, (0, 255, 0), 2)
        cls = int(cls)
        name = NAMES[cls] if 0 <= cls < len(NAMES) else str(cls)
        cv2.putText(frame0, f"{name} {conf:.2f}", (p1[0], max(0, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame0

# ---- Worker ----
def worker():
    global last_jpeg, last_frame, frame_counter
    proc = spawn_libcamera()
    try:
        for jpg in mjpeg_reader(proc):
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            frame_counter += 1
            if frame_counter % INFER_EVERY == 0:
                t0 = time.perf_counter()
                det, frame0, gain, pad = infer_onnx(frame)
                plot = postprocess(det, frame0, gain, pad)
                infer_times.append(time.perf_counter() - t0)
            else:
                plot = last_frame if last_frame is not None else frame
            ok, enc = cv2.imencode(".jpg", plot, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with lock:
                    last_jpeg = enc.tobytes()
                    last_frame = plot
    finally:
        try:
            proc.terminate(); proc.wait(timeout=2)
        except Exception:
            try: proc.kill()
            except Exception: pass

def stream_generator():
    boundary = b"--frame"
    while True:
        with lock:
            buf = last_jpeg
        if buf is not None:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
        time.sleep(0.03)

# ---- HTTP ----
@app.get("/")
def index():
    return render_template_string("""
<!doctype html><html><head><meta charset="utf-8"><title>RT YOLO ONNX</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:system-ui} .wrap{max-width:960px;margin:24px auto;text-align:center}</style>
</head><body><div class="wrap">
<h3>RT YOLO (ONNX Runtime)</h3>
<img src="/stream" style="max-width:100%;border:1px solid #333"/>
<p><button onclick="snap()">Snapshot</button> <span id="msg"></span></p>
<script>
async function snap(){const r=await fetch('/snapshot',{method:'POST'});const j=await r.json();document.getElementById('msg').textContent= j.ok ? ('Saved: '+j.path) : ('Error: '+j.error);}
</script>
</div></body></html>""")

@app.get("/stream")
def stream():
    return Response(stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/snapshot")
def snapshot():
    with lock:
        img = None if last_frame is None else last_frame.copy()
    if img is None:
        return jsonify({"ok": False, "error": "no_frame"}), 503
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = OUTDIR / f"snapshot_{ts}.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return jsonify({"ok": True, "path": str(path)})

@app.get("/status")
def status():
    cpu, ram = read_cpu_ram()
    fps_infer = 0.0 if not infer_times else 1.0 / float(np.mean(infer_times))
    uptime = int(time.time() - start_time)
    return jsonify({
        "engine": "onnxruntime",
        "model": ONNX_PATH,
        "frames": frame_counter,
        "infer_fps_est": round(fps_infer, 2),
        "cpu_pct": cpu,
        "ram_pct": ram,
        "uptime_s": uptime,
    })

if __name__ == "__main__":
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
