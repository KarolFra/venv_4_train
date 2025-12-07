# check_ui.py
import os, glob, time, pathlib
from flask import Flask, request, send_file, render_template_string, jsonify
from ultralytics import YOLO

BASE = pathlib.Path(__file__).resolve().parent
DEFAULT_DIR = "/home/pi/ConveyorBelt-mqtt/Venv/dataset_stream_captures/2/20251007_213029"
WEIGHTS = "/home/pi/ConveyorBelt-mqtt/Venv/models/runs_yolo/pcb-detect/weights/best.pt"  # Twój model
OUTROOT = BASE / "results" / "check_model_ui"

app = Flask(__name__)
model = YOLO(WEIGHTS)
device = "cuda" if model.device.type == "cuda" else "cpu"

INDEX_HTML = """
<!doctype html>
<title>YOLO quick test</title>
<h2>YOLO test (device: {{device}}, weights: {{weights_name}})</h2>
<form method="post" action="/run">
  <label>Directory with images:&nbsp;</label>
  <input type="text" name="dir" size="90" value="{{default_dir}}"/>
  <button type="submit">Run test</button>
</form>
{% if msg %}<p style="color:#c00">{{msg}}</p>{% endif %}
{% if results %}
  <h3>Results saved in: {{outdir}}</h3>
  <ul>
  {% for r in results %}
    <li>
      <div>Input: {{r.input}} &nbsp; | &nbsp; Dets: {{r.count}}</div>
      <div><a href="/file?path={{r.output}}">download</a></div>
      <img src="/file?path={{r.output}}" style="max-width:640px;border:1px solid #ddd;margin:6px 0">
    </li>
  {% endfor %}
  </ul>
{% endif %}
"""

@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML,
        device=device,
        weights_name=os.path.basename(WEIGHTS),
        default_dir=DEFAULT_DIR,
        msg=None, results=None, outdir=None,
    )

from pathlib import Path
import os

@app.route("/run", methods=["POST"])
def run():
    imgdir = request.form.get("dir", "").strip()
    if not imgdir or not os.path.isdir(imgdir):
        return render_template_string(INDEX_HTML, device=device,
            weights_name=os.path.basename(WEIGHTS),
            default_dir=imgdir or DEFAULT_DIR,
            msg="Directory not found.", results=None, outdir=None)

    root = Path(imgdir).expanduser().resolve()
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # pomiń ukryte katalogi
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(Path(dirpath) / fn)

    if not files:
        return render_template_string(INDEX_HTML, device=device,
            weights_name=os.path.basename(WEIGHTS),
            default_dir=str(root),
            msg="No images found (searched recursively).", results=None, outdir=None)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = OUTROOT / stamp
    items = []
    outdir.mkdir(parents=True, exist_ok=True)

    for p in sorted(files):
        res = model.predict(str(p), conf=0.25, iou=0.45, verbose=False)[0]
        plot = res.plot()
        rel = p.relative_to(root)
        out_path = outdir / rel.parent / ("annot_" + p.name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(out_path), plot, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        items.append({
            "input": str(p),
            "output": str(out_path),
            "count": 0 if res.boxes is None else int(res.boxes.shape[0]),
        })

    return render_template_string(
        INDEX_HTML,
        device=device,
        weights_name=os.path.basename(WEIGHTS),
        default_dir=str(root), msg="",
        results=items, outdir=str(outdir)
    )

@app.route("/file")
def file():
    path = request.args.get("path")
    if not path or not os.path.isfile(path):
        return jsonify({"error":"not_found","path":path}), 404
    return send_file(path, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
