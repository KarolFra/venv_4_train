# YOLOv8 dataset runner

This repository contains minimal scripts to train and run Ultralytics YOLOv8 on your dataset and a small set of helpers to prepare models for Raspberry Pi (Pi 4/5).

Files included
- `train.py` - small wrapper to run Ultralytics training
- `train_full.py` - orchestrate training for several model sizes
- `detect.py` - run inference and optionally save results
- `run_models.py` - benchmark different pretrained sizes (nano -> xlarge)
- `export_onnx.py` - export `.pt` weights to `.onnx`
- `rpi_infer.py` - inference helper with ONNX Runtime preferred, fallback to PyTorch
- `hyp.yaml` - suggested hyperparameters
- `smoke_test.py` - verify imports and package versions
- `requirements.txt` - pinned dependencies (Windows/x86 environment)

Quick start (PowerShell)

```powershell
# Activate venv in this project (adjust path if different):
. .\.venv\Scripts\Activate.ps1
python smoke_test.py
python detect.py --weights yolov8n.pt --source test/images --save --device cpu
python train.py --model yolov8n.pt --data data.yaml --epochs 10 --device cpu
```

Raspberry Pi (4 and 5) guidance

Summary:
- For deployment on a Pi you should prefer smaller models (`yolov8n.pt` or `yolov8s.pt`) or convert to ONNX and quantize the model.
- Install ARM-compatible runtime packages on the Pi (PyTorch aarch64 wheels or ONNX Runtime for aarch64).

Checklist to prepare and run on Pi 5

1) Export or pick the trained weights you want to deploy (e.g., `runs/train/exp/weights/best.pt`).

2) Optional: export to ONNX locally (recommended before transfer):

```powershell
python export_onnx.py --weights runs/train/exp/weights/best.pt --output best.onnx --opset 14
```

3) Transfer files to the Pi (example using scp from PowerShell):

```powershell
# copy ONNX and sample images to Pi
scp .\best.onnx pi@<pi-ip>:/home/pi/models/
scp -r .\test\images pi@<pi-ip>:/home/pi/test_images/
```

4) On the Pi, create+activate a venv and install runtime deps (example commands):

```bash
python3 -m venv venv
source venv/bin/activate
# For ONNX Runtime (preferred):
pip install onnxruntime numpy pillow
# OR for PyTorch path (use ARM wheel that matches your Pi OS and Python):
# pip install torch-<ver>-cp3x-none-linux_aarch64.whl torchvision-<ver>.whl
```

5) Run inference on the Pi (ONNX recommended):

```bash
python rpi_infer.py --model /home/pi/models/best.onnx --source /home/pi/test_images --imgsz 320
```

6) If ONNX isn't available, use the PyTorch model:

```bash
python rpi_infer.py --model /home/pi/models/best.pt --source /home/pi/test_images --imgsz 320
```

Performance tips for Pi 5:
- Use `yolov8n.pt` or a quantized ONNX model for best speed/memory trade-off.
- Reduce `imgsz` to 320 or 416 and use batch size = 1.
- Set `OMP_NUM_THREADS=1` if you see CPU contention, or set it to the number of physical cores for throughput tests.
- Consider using on-device accelerators or vendor-provided runtimes if available for your OS.

Next steps I can take for you
- Add an ONNX quantization script (post-training static quantization to int8) and test export locally.
- Add a small Pi camera demo (`rpi_camera_demo.py`) that runs continuous inference and reports FPS.
- Start full training runs (e.g., `yolov8m`, `yolov8l`) using `train_full.py` with the `hyp.yaml` hyperparameters—note: training larger models on your workstation requires a capable GPU.# YOLOv8 dataset runner

Files added:

- `train.py` - wrapper to train using Ultralytics YOLOv8
- `detect.py` - wrapper to run inference/detection
- `run_models.py` - iterate models (nano->xlarge) and measure time on a sample image
- `smoke_test.py` - verify imports and versions in the current venv
- `requirements.txt` - pinned dependencies used in the environment (for Windows/x86)

Quick start (PowerShell):

```powershell
# Activate venv if not already: (adjust path if needed)
. .\.venv\Scripts\Activate.ps1
python smoke_test.py
python detect.py --weights yolov8n.pt --source test/images --save --device cpu
python train.py --model yolov8n.pt --data data.yaml --epochs 10 --device cpu
```

Raspberry Pi 4 notes:
- Reinstall `torch`/`torchvision` on the Pi using ARM-compatible wheels or use pip wheels built for the Pi (these are not the same as Windows x86 wheels).
- Prefer `yolov8n.pt` or `yolov8s.pt` for live inference on the Pi 4. Consider quantization or ONNX with TensorRT/NNAPI where available.
- Reduce `imgsz` and batch size for lower memory use.
