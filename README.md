Most interesting results are in `Final_results/` (curated plots, predictions, and final thesis photos—no weights).

## Thesis / demo visuals

<table>
  <tr>
    <td><b>Dashboard / API</b><br>
      <a href="Final_results/experiment_artifacts/ModelPhoto/Dashboard_PCB.png">
        <img src="Final_results/experiment_artifacts/ModelPhooto/Dashboard_PCB.png" width="420">
      </a>
    </td>
    <td><b>Prototype stand</b><br>
      <a href="Final_results/Results/stanowiskoOswietlone.jpg">
        <img src="Final_results/experiment_artifacts/Results/stanowiskoOswietlone.jpg" width="420">
      </a>
    </td>
  </tr>
  <tr>
    <td><b>KiCad schematic system ESP32</b><br>
      <a href="Final_results/experiment_artifacts/ModelPhoto/schematic_makieta.png">
        <img src="Final_results/experiment_artifacts/ModelPhooto/Kicad3D.png" width="420">
      </a>
    </td>
    <td><b>KiCad 3D model</b><br>
      <a href="Final_results/experiment_artifacts/ModelPhoto/Kicad3D.png">
        <img src="https://raw.githubusercontent.com/KarolFra/venv_4_train/main/Final_results/experiment_artifacts/ModelPhoto/Kicad3D.png" width="420">
      </a>
    </td>
  </tr>
</table>

<b>Final results (2x2)</b><br>
<a href="Final_results/Results/obroconePlytki_2x2.png">
  <img src="Final_results/experiment_artifacts/Results/obroconePlytki_2x2.png" width="700">
</a>
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
- `Final_results/pcbb_v5i_yolo11s_e180` and `Final_results/KICADgood_v5i_yolo11s_e180`: key metrics, confusion matrices, val predictions.
- `Final_results/extra_photos/` contains end-of-thesis reference photos/renders used for reporting.

Future work
- Add ONNX int8 post-training quantization script.
- Optional Pi camera demo loop for live inference.
