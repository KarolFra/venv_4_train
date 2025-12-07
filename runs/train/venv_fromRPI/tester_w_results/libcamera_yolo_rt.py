# app_rt_stream.py
import os, sys, time, subprocess, threading, collections
from pathlib import Path
from flask import Flask, Response, jsonify
import numpy as np
import cv2
from ultralytics import YOLO

import threading, time

frame_counter = 0
_start_time = time.time()

def _read_cpu_ram():
    # CPU%
    with open("/proc/stat") as f:
        a = list(map(int, f.readline().split()[1:8]))
    time.sleep(0.1)
    with open("/proc/stat") as f:
        b = list(map(int, f.readline().split()[1:8]))
    ta, ia = sum(a), a[3]; tb, ib = sum(b), b[3]
    cpu = 0.0 if tb==ta else (1 - (ib-ia)/max(1, (tb-ta))) * 100.0
    # RAM%
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k,v,*_ = line.split()
            mem[k[:-1]] = int(v)
    total = mem.get("MemTotal",1)
    avail = mem.get("MemAvailable",0)
    ram = (total - avail) * 100.0 / total
    return round(cpu,1), round(ram,1)

def _sys_monitor():
    while True:
        cpu, ram = _read_cpu_ram()
        up = int(time.time() - _start_time)
        print(f"[sys] CPU {cpu:.1f}% | RAM {ram:.1f}% | frames={frame_counter} | uptime={up}s", flush=True)
        time.sleep(3)


# -------- CONFIG --------
WEIGHTS = "/home/pi/ConveyorBelt-mqtt/Venv/models/runs_yolo/pcb-detect/weights/best.pt"
WIDTH, HEIGHT, FPS = 640, 480, 10
CONF, IOU = 0.25, 0.45
INFER_EVERY = 1          # 1 = każda klatka; zwiększ (2/3) by odciążyć CPU
JPEG_QUALITY = 85
OUTDIR = Path("/home/pi/ConveyorBelt-mqtt/Venv/results/live_flask")
# ------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
model = YOLO(WEIGHTS)
device = "cuda" if model.device.type == "cuda" else "cpu"
print(f"[info] YOLO on {device}", flush=True)

# shared state
lock = threading.Lock()
last_annot = None             # bytes (JPEG) do streamu
last_np = None                # ndarray z ostatnią zannotowaną klatką
frame_counter = 0
infer_times = collections.deque(maxlen=30)
start_time = time.time()

def read_cpu_ram():
    # CPU (krótki sampling)
    with open("/proc/stat") as f:
        a = list(map(int, f.readline().split()[1:8]))
    time.sleep(0.1)
    with open("/proc/stat") as f:
        b = list(map(int, f.readline().split()[1:8]))
    ta, ia = sum(a), a[3]; tb, ib = sum(b), b[3]
    cpu = 0.0 if tb==ta else (1 - (ib-ia)/((tb-ta) or 1)) * 100.0
    # RAM
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k,v,*_ = line.split()
            info[k[:-1]] = int(v)
    total = info.get("MemTotal",1); avail = info.get("MemAvailable",0)
    ram = (total - avail) * 100.0 / total
    return round(cpu,1), round(ram,1)

def spawn_libcamera():
    cmd = [
        "libcamera-vid", "--inline", "--timeout", "0",
        "--framerate", str(FPS), "--width", str(WIDTH), "--height", str(HEIGHT),
        "--codec", "mjpeg", "--output", "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

def mjpeg_reader(proc):
    buf = b""; SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    while True:
        chunk = proc.stdout.read(8192)
        if not chunk: break
        buf += chunk
        while True:
            i = buf.find(SOI)
            j = buf.find(EOI, i+2)
            if i == -1 or j == -1: break
            jpg = buf[i:j+2]
            buf = buf[j+2:]
            yield jpg

def worker():
    global last_annot, last_np, frame_counter
    proc = spawn_libcamera()
    try:
        for jpg in mjpeg_reader(proc):
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            frame_counter += 1
            do_infer = (frame_counter % INFER_EVERY) == 0
            if do_infer:
                t0 = time.perf_counter()
                res = model.predict(source=frame, conf=CONF, iou=IOU, verbose=False)[0]
                plot = res.plot()
                infer_times.append(time.perf_counter() - t0)
            else:
                plot = last_np if last_np is not None else frame

            ok, enc = cv2.imencode(".jpg", plot, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with lock:
                    last_annot = enc.tobytes()
                    last_np = plot
    finally:
        try:
            proc.terminate(); proc.wait(timeout=2)
        except Exception:
            try: proc.kill()
            except Exception: pass

def stream_generator():
    while True:
        with lock:
            buf = last_annot
        if buf is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
        time.sleep(0.03)

@app.route("/stream")
def stream():
    return Response(stream_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/snapshot")
def snapshot():
    with lock:
        img = None if last_np is None else last_np.copy()
    if img is None:
        return jsonify({"ok": False, "error": "no_frame"}), 503
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = OUTDIR / f"snapshot_{ts}.jpg"
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return jsonify({"ok": True, "path": str(path)})

@app.get("/status")
def status():
    cpu, ram = read_cpu_ram()
    fps_infer = (0.0 if not infer_times else 1.0/np.mean(infer_times))
    uptime = time.time() - start_time
    return jsonify({
        "device": device,
        "weights": WEIGHTS,
        "frames": frame_counter,
        "infer_fps_est": round(fps_infer, 2),
        "cpu_pct": cpu,
        "ram_pct": ram,
        "uptime_s": int(uptime),
    })

if __name__ == "__main__":
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Odpal na stałym porcie, np. 5002 (żeby nie kolidować)
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)

