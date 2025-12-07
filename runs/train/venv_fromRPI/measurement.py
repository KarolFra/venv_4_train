from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2


Detection = Dict[str, float]


def _draw_box(frame, det: Detection, color=(56, 189, 248), thickness=2, label_suffix: Optional[str] = None):
    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
    label = det.get('label', 'object')
    text = label
    if label_suffix:
        text = f'{label} {label_suffix}'
    start = (int(x1), int(y1))
    end = (int(x2), int(y2))
    cv2.rectangle(frame, start, end, color, thickness)
    text_origin = (start[0], max(18, start[1] - 8))
    cv2.putText(frame, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def annotate_basic(frame, detections: List[Detection]) -> None:
    for det in detections:
        conf = det.get('confidence')
        suffix = f'{conf:.2f}' if conf is not None else None
        _draw_box(frame, det, label_suffix=suffix)


def annotate_with_measurements(
    frame,
    detections: List[Detection],
    pixels_per_cm: float,
) -> Optional[Tuple[float, float]]:
    best_pair: Optional[Tuple[float, float]] = None
    best_conf = -1.0
    for det in detections:
        conf = float(det.get('confidence', 0.0))
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        width_cm = width_px / pixels_per_cm
        height_cm = height_px / pixels_per_cm
        size_caption = f'{width_cm:.2f}cm x {height_cm:.2f}cm'
        suffix = f'{conf:.2f}'
        _draw_box(frame, det, label_suffix=suffix)
        start = (int(x1), int(y1))
        end = (int(x2), int(y2))
        cv2.putText(frame, size_caption, (start[0], end[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 253, 245), 1, cv2.LINE_AA)
        if conf > best_conf:
            best_conf = conf
            best_pair = (width_cm, height_cm)
    return best_pair


def annotate_frame(
    frame,
    detections: List[Detection],
    pixels_per_cm: float,
    include_measurements: bool,
) -> Tuple:
    if include_measurements:
        best_pair = annotate_with_measurements(frame, detections, pixels_per_cm)
        return frame, best_pair
    annotate_basic(frame, detections)
    return frame, None
