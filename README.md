Most interesting results are in `selected_results/` (curated plots, predictions, and final thesis photos—no weights).

# YOLOv8 / YOLOv11 dataset runner

Minimal scripts to train and run Ultralytics YOLOv8/YOLOv11 on your own dataset plus small helpers to prep models for Raspberry Pi 4/5.

Included scripts
- `train.py` / `train_full.py` – wrappers to launch Ultralytics training
- `detect.py` – run inference and optionally save outputs
- `run_models.py` – benchmark multiple pretrained sizes (nano → large)
- `export_onnx.py` – convert `.pt` weights to `.onnx`
- `rpi_infer.py` – ONNX/PyTorch inference helper for Pi 4/5
- `hyp.yaml`, `requirements.txt`, `smoke_test.py`, `list_weights.py`, `validate_labels.py`

Quick start (PowerShell)
```powershell
# activate venv (adjust path if different)
. .\.venv\Scripts\Activate.ps1
python smoke_test.py
python detect.py --weights yolov8n.pt --source test/images --save --device cpu
python train.py --model yolov11s.pt --data data.yaml --epochs 10 --device cpu
```

Raspberry Pi (4/5) tips
- Prefer small models (`yolov8n/s` or `yolov11n/s`) or export to ONNX and quantize.
- Install ARM wheels (PyTorch) or `onnxruntime` aarch64 on the Pi.
- Use smaller `imgsz` (320–416) and batch size 1; set `OMP_NUM_THREADS` to limit CPU contention.

Selected results
- `selected_results/pcbb_v5i_yolo11s_e180` and `selected_results/KICADgood_v5i_yolo11s_e180`: key metrics, confusion matrices, val predictions.
- `selected_results/extra_photos/` contains end-of-thesis reference photos/renders used for reporting.

Future work
- Add ONNX int8 post-training quantization script.
- Optional Pi camera demo loop for live inference.
