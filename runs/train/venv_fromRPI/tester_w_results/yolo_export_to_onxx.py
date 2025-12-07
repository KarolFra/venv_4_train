import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 640])
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--simplify", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.weights)
    model.export(
        format="onnx",
        imgsz=tuple(a.imgsz) if len(a.imgsz)==2 else a.imgsz[0],
        opset=a.opset,
        dynamic=a.dynamic,
        half=a.half,
        simplify=a.simplify,
        verbose=True
    )
    # wynik: obok wag powstanie best.onnx

if __name__ == "__main__":
    main()
