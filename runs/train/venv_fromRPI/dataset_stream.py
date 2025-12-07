#!/usr/bin/env python3
"""
Lightweight Flask app that streams the conveyor camera with a central 2/3 crop.

Usage:
  python dataset_stream.py --host 0.0.0.0 --port 8080
  python dataset_stream.py --crop-fraction 0.7
  python dataset_stream.py --libcamera-command libcamera-vid --inline ... --output -
"""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request, url_for
from flask_mqtt import Mqtt

from dataset_capture import _crop_center_fraction, _stream_mjpeg_frames


class MJPEGStreamer:
    """Background worker that keeps the latest cropped JPEG frame ready for Flask."""

    def __init__(self, command: Iterable[str], crop_fraction: float):
        self._command = list(command)
        self._crop_fraction = crop_fraction
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                for frame in _stream_mjpeg_frames(self._command):
                    if self._stop_event.is_set():
                        break
                    cropped = _crop_center_fraction(frame, self._crop_fraction)
                    ok, encoded = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if not ok:
                        continue
                    with self._lock:
                        self._latest_jpeg = encoded.tobytes()
                if self._stop_event.is_set():
                    break
                # If the stream ended unexpectedly, pause briefly before retrying.
                time.sleep(0.5)
            except Exception as exc:
                print(f"[streamer] Error while capturing frames: {exc}")
                time.sleep(1.0)

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg


class CaptureManager:
    """Persist captured frames to disk with per-session numbering."""

    def __init__(self, output_root: Path, default_label: str):
        self._output_root = output_root
        self._default_label = default_label
        self._session_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def save(self, frame_bytes: bytes, label: Optional[str]) -> Path:
        label_clean = (label or self._default_label or "capture").strip() or "capture"
        target_dir = self._output_root / label_clean / self._session_stamp
        target_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            index = self._counters[label_clean]
            self._counters[label_clean] += 1
        filename = target_dir / f"{label_clean}_{index:04d}.jpg"
        with open(filename, "wb") as fh:
            fh.write(frame_bytes)
        return filename


def resolve_mqtt_host(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    local_ip = socket.gethostbyname(socket.gethostname())
    if local_ip.startswith("192.168.1."):
        return "192.168.1.21"
    if local_ip.startswith("192.168.0."):
        return "192.168.0.121"
    return "localhost"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a cropped conveyor camera stream over Flask.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP for Flask.")
    parser.add_argument("--port", type=int, default=8080, help="Port for Flask.")
    parser.add_argument("--crop-fraction", type=float, default=2.0 / 3.0,
                        help="Horizontal fraction of the frame to keep (0-1].")
    parser.add_argument("--libcamera-command", nargs="*", default=None,
                        help="Override the libcamera-vid command (advanced).")
    parser.add_argument("--output-root", default="dataset_stream_captures",
                        help="Directory root where captured frames will be stored.")
    parser.add_argument("--default-label", default="capture",
                        help="Fallback label when none is provided in capture requests.")
    parser.add_argument("--mqtt-broker", help="Override MQTT broker hostname/IP.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-username", default="", help="MQTT username if required.")
    parser.add_argument("--mqtt-password", default="", help="MQTT password if required.")
    parser.add_argument("--light-topic", default="/home/control/light", help="MQTT topic used to toggle the light.")
    parser.add_argument("--light-on-state", default="ON", help="Payload published to switch the light on.")
    parser.add_argument("--light-off-state", default="OFF", help="Payload published to switch the light off.")
    parser.add_argument("--light-warmup", type=float, default=0.4,
                        help="Seconds to wait after turning the light on before capturing.")
    parser.add_argument("--light-cooldown", type=float, default=0.2,
                        help="Seconds to wait before publishing the light-off payload.")
    parser.add_argument("--no-light-off", action="store_true",
                        help="Skip sending the light-off payload after capture.")
    return parser.parse_args()


def default_libcamera_command() -> list[str]:
    return [
        "libcamera-vid",
        "--inline",
        "--timeout", "0",
        "--framerate", "10",
        "--width", "640",
        "--height", "480",
        "--codec", "mjpeg",
        "--output", "-",
    ]


def register_routes(app: Flask, streamer: MJPEGStreamer, capturer: CaptureManager, mqtt: Mqtt, args: argparse.Namespace) -> None:

    @app.route("/")
    def index():
        return render_template_string(
            """
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <title>Dataset Stream</title>
              <style>
                body { font-family: sans-serif; background: #222; color: #eee; text-align: center; }
                main { max-width: 720px; margin: 0 auto; padding: 20px; }
                img { border: 4px solid #444; border-radius: 8px; margin-top: 20px; max-width: 100%; height: auto; }
                .controls { margin-top: 20px; display: flex; gap: 12px; justify-content: center; align-items: center; flex-wrap: wrap; }
                input[type="text"] { padding: 8px; border-radius: 4px; border: 1px solid #666; background: #111; color: #eee; }
                button { padding: 10px 16px; border: none; border-radius: 4px; background: #28a745; color: #fff; cursor: pointer; font-size: 1rem; }
                button:disabled { background: #3a3a3a; cursor: not-allowed; }
                #status { margin-top: 14px; min-height: 1.2em; }
              </style>
            </head>
            <body>
              <main>
                <h1>Conveyor Stream (cropped)</h1>
                <p>Central portion of the 640px feed for quick dataset review and capture.</p>
                <img src="{{ video_url }}" alt="Video stream">
                <section class="controls">
                  <label for="labelInput">Label:</label>
                  <input id="labelInput" type="text" placeholder="{{ default_label }}">
                  <button id="captureBtn">Capture Frame</button>
                </section>
                <div id="status"></div>
              </main>
              <script>
                const captureBtn = document.getElementById('captureBtn');
                const labelInput = document.getElementById('labelInput');
                const statusEl = document.getElementById('status');

                async function captureFrame() {
                  const label = labelInput.value.trim();
                  captureBtn.disabled = true;
                  statusEl.textContent = 'Capturing...';
                  try {
                    const response = await fetch('{{ capture_url }}', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify(label ? { label } : {})
                    });
                    if (!response.ok) {
                      const msg = await response.text();
                      throw new Error(msg || 'Capture failed');
                    }
                    const payload = await response.json();
                    statusEl.textContent = `Saved: ${payload.saved_path}`;
                  } catch (error) {
                    statusEl.textContent = `Error: ${error.message}`;
                  } finally {
                    captureBtn.disabled = false;
                  }
                }

                captureBtn.addEventListener('click', captureFrame);
              </script>
            </body>
            </html>
            """,
            video_url=url_for("video_feed"),
            capture_url=url_for("capture_frame"),
            default_label=args.default_label,
        )

    def frame_generator():
        while True:
            frame = streamer.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    @app.route("/video_feed")
    def video_feed():
        return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/capture", methods=["POST"])
    def capture_frame():
        payload = request.get_json(silent=True) or {}
        label = payload.get("label")

        try:
            mqtt.publish(args.light_topic, args.light_on_state)
        except Exception as exc:
            print(f"[capture] Failed to publish light ON: {exc}")
        if args.light_warmup > 0:
            time.sleep(args.light_warmup)

        frame_bytes: Optional[bytes] = None
        for _ in range(10):
            frame_bytes = streamer.get_frame()
            if frame_bytes is not None:
                break
            time.sleep(0.05)

        if frame_bytes is None:
            return jsonify({"status": "error", "message": "No frame available yet."}), 503

        saved_path = capturer.save(frame_bytes, label)
        print(f"[capture] Saved {saved_path}")

        if not args.no_light_off:
            if args.light_cooldown > 0:
                time.sleep(args.light_cooldown)
            try:
                mqtt.publish(args.light_topic, args.light_off_state)
            except Exception as exc:
                print(f"[capture] Failed to publish light OFF: {exc}")

        return jsonify({"status": "ok", "saved_path": str(saved_path)})


def main() -> int:
    args = parse_args()
    command = args.libcamera_command or default_libcamera_command()
    streamer = MJPEGStreamer(command, args.crop_fraction)
    streamer.start()

    app = Flask(__name__)
    app.config["MQTT_BROKER_URL"] = resolve_mqtt_host(args.mqtt_broker)
    app.config["MQTT_BROKER_PORT"] = args.mqtt_port
    app.config["MQTT_USERNAME"] = args.mqtt_username
    app.config["MQTT_PASSWORD"] = args.mqtt_password
    app.config["MQTT_KEEPALIVE"] = 60
    app.config["MQTT_TLS_ENABLED"] = False

    mqtt = Mqtt(app)
    capture_manager = CaptureManager(Path(args.output_root), args.default_label)
    register_routes(app, streamer, capture_manager, mqtt, args)

    def _shutdown_handler(signum, frame):
        print(f"[stream] Signal {signum} received. Shutting down.")
        streamer.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        streamer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
