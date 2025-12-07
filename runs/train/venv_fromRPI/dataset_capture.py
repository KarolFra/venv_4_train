#!/usr/bin/env python3
"""
Utility script for collecting dataset images from the conveyor camera.

Features:
  * Streams MJPEG frames via libcamera-vid (matching the main app resolution).
  * Crops each frame to the central 2/3 of the width to reduce background noise.
  * Saves a configurable number of frames into an output directory with timestamps.

Usage examples:
  python dataset_capture.py --output data/raw --label resistor --count 25 --interval 1.5
  python dataset_capture.py --output data/raw --label capacitor --manual
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
import time
from typing import Generator, Optional

import cv2
import numpy as np


def _stream_mjpeg_frames(command: list[str]) -> Generator[np.ndarray, None, None]:
    """Yield decoded frames from an MJPEG stream produced by libcamera-vid."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    buffer = b""
    try:
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("Failed to open libcamera-vid stdout pipe.")
        while True:
            chunk = stdout.read(8192)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")  # JPEG SOI
                end = buffer.find(b"\xff\xd9")    # JPEG EOI
                if start == -1 or end == -1:
                    break
                jpg = buffer[start:end + 2]
                buffer = buffer[end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def _crop_center_fraction(frame: np.ndarray, fraction: float) -> np.ndarray:
    """Return the central horizontal crop covering the requested fraction of width."""
    if fraction >= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(1, int(w * fraction))
    start_x = max(0, (w - crop_w) // 2)
    end_x = min(w, start_x + crop_w)
    return frame[:, start_x:end_x]


def _prepare_output_dir(base: pathlib.Path, label: Optional[str]) -> pathlib.Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if label:
        target = base / label / timestamp
    else:
        target = base / timestamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dataset images from the conveyor camera.")
    parser.add_argument("--output", default="dataset", help="Root directory for captured images.")
    parser.add_argument("--label", help="Optional class label used to create a subdirectory.")
    parser.add_argument("--count", type=int, default=10, help="Number of frames to capture.")
    parser.add_argument("--interval", type=float, default=1.0, help="Delay (seconds) between captures in automatic mode.")
    parser.add_argument("--crop-fraction", type=float, default=2.0 / 3.0,
                        help="Fraction of frame width to keep (default: central 2/3).")
    parser.add_argument("--manual", action="store_true",
                        help="Capture on Enter keypress instead of fixed interval.")
    parser.add_argument("--libcamera-command", nargs="*", default=None,
                        help="Override libcamera-vid command (advanced use).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_root = pathlib.Path(args.output)
    output_dir = _prepare_output_dir(output_root, args.label)
    print(f"[capture] Saving images to: {output_dir}")

    if args.libcamera_command:
        command = args.libcamera_command
    else:
        command = [
            "libcamera-vid",
            "--inline",
            "--timeout", "0",
            "--framerate", "10",
            "--width", "640",
            "--height", "480",
            "--codec", "mjpeg",
            "--output", "-"
        ]

    frame_iter = _stream_mjpeg_frames(command)

    captured = 0
    try:
        for frame in frame_iter:
            cropped = _crop_center_fraction(frame, args.crop_fraction)
            filename = output_dir / f"{args.label or 'capture'}_{captured:04d}.jpg"
            if args.manual:
                print("Press Enter to capture this frame, or Ctrl+C to exit.", end=" ", flush=True)
                try:
                    input()
                except KeyboardInterrupt:
                    print("\n[capture] Interrupted by user.")
                    break
            else:
                time.sleep(max(0.0, args.interval))

            success = cv2.imwrite(str(filename), cropped)
            if not success:
                print(f"[capture] Failed to save frame {captured} to {filename}")
                continue
            print(f"[capture] Saved {filename}")
            captured += 1
            if captured >= args.count:
                break
    except KeyboardInterrupt:
        print("\n[capture] Interrupted by user.")

    print(f"[capture] Finished. Total images saved: {captured}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
