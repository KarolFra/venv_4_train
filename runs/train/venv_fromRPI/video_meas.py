# video_meas.py
import atexit
import binascii
import os
import json
import shutil
import queue
import secrets
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from multiprocessing import Event
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import psutil  # <-- monitoring CPU/RAM
from flask import Blueprint, Response, jsonify, current_app, request
from ultralytics import YOLO

from measurement import annotate_frame

# ---------------- robust path helpers ----------------
import logging
import platform

log = logging.getLogger(__name__)
_DEBUG_ENV = os.getenv('VIDEO_MEAS_DEBUG', '')
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[video_meas] %(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(handler)
if _DEBUG_ENV.strip():
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

def _base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception as e:
        log.warning("Cannot resolve __file__. Using CWD. Details: %s\n%s",
                    e, traceback.format_exc())
        return Path.cwd()

def _ensure_dir(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        debug = {
            "path": str(p),
            "cwd": str(Path.cwd()),
            "argv0": sys.argv[0] if sys.argv else None,
            "platform": platform.platform(),
        }
        raise RuntimeError(f"Failed to create directory: {p}\ndebug={debug}") from e
    if not p.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {p}")
    return p

def _ensure_parent(p: Path) -> None:
    _ensure_dir(p.parent)

def _report_missing(label: str, p: Path) -> None:
    msg = f"{label} not found at {p}"
    log.error(msg)
# -----------------------------------------------------

# tymczasowe: ostatnie detekcje do podglądu JSON
_snapshot_last_detections: Optional[list] = None

video_bp = Blueprint('video_bp', __name__)

# Shared frame buffer consumed by Flask while worker processes keep running.
latest_frame: Optional[bytes] = None
latest_raw_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()
raw_frame_lock = threading.Lock()
_ai_recognition_lock = threading.Lock()
_ai_recognition_enabled = True
_last_stream_wait_log = 0.0
_inference_state_lock = threading.Lock()
_inference_state = {
    'mode': 'initializing',
    'detail': 'Initialising inference pipeline…',
    'error': '',
    'model_path': None,
    'timestamp': time.time(),
}
_local_model_error = ''
AI_SNAPSHOT_DURATION = float(os.getenv('AI_SNAPSHOT_DURATION', '15'))
_ai_detection_lock = threading.Lock()
_ai_detection_requested = False
_snapshot_payload_bytes: Optional[bytes] = None
_snapshot_best_pair: Optional[Tuple[float, float]] = None
_snapshot_expiry = 0.0
_stream_active = True
_stream_stop_event_proxy: Optional[Any] = None
_halt_stream_after_snapshot = threading.Event()

FRAME_DEBUG_INTERVAL = int(os.getenv('VIDEO_FRAME_DEBUG_INTERVAL', '60'))
QUEUE_EMPTY_LOG_INTERVAL = float(os.getenv('VIDEO_QUEUE_WARN_INTERVAL', '5'))
DETECTION_DEBUG_INTERVAL = float(os.getenv('VIDEO_DETECTION_DEBUG_INTERVAL', '15'))

_frame_stats_lock = threading.Lock()
_frame_counter = 0
_detection_counter = 0
_snapshot_counter = 0
_last_queue_log = 0.0
_last_detection_log = 0.0

# Callback injected from app.py to push detections into the conveyor logic.
measurement_callback = None

# ---------- paths with validation/auto-create ----------
_BASE = _base_dir()

# Directories
DEFAULT_MODELS_DIR = _ensure_dir(Path(os.getenv('MODELS_DIR', _BASE / 'models')))
DETECTION_SNAPSHOT_DIR = _ensure_dir(Path(os.getenv('AI_SNAPSHOT_DIR', _BASE / 'detections')))
DETECTION_RESULTS_DIR = _ensure_dir(Path(os.getenv('DETECTION_RESULTS_DIR', _BASE / 'results_detection')))
YOLO_RESULTS_ROOT = _ensure_dir(Path(os.getenv('YOLO_RESULTS_DIR', _BASE / 'results')))
BEST_RESULTS_DIR = _ensure_dir(Path(os.getenv('BEST_RESULTS_DIR', YOLO_RESULTS_ROOT / 'best_results')))
AUTO_STOP_STREAM_ON_ENABLE = os.getenv('AUTO_STOP_STREAM_AFTER_ENABLE', 'true').strip().lower() not in {'0', 'false', 'no', 'on'}
PREFERRED_BEST_MODEL = Path(os.getenv('BEST_MODEL_WEIGHTS', _BASE / 'models' / 'runs_yolo' / 'pcb-detect' / 'weights' / 'best.pt'))

# Files
WORKER_SCRIPT = Path(os.getenv('VIDEO_WORKER', _BASE / 'video_worker.py'))
_ensure_parent(WORKER_SCRIPT)
if not WORKER_SCRIPT.exists():
    _report_missing("Video worker script", WORKER_SCRIPT)

DEFAULT_YOLO_WEIGHTS = Path(os.getenv('YOLO_FALLBACK_WEIGHTS', _BASE / 'yolo11n.pt'))

YOLO_MODEL_PATH_ENV = os.getenv('YOLO_MODEL_PATH')
YOLO_MODEL_DIR_ENV = os.getenv('YOLO_MODEL_DIR')
YOLO_CONFIDENCE = float(os.getenv('YOLO_CONFIDENCE', '0.25'))
YOLO_IOU = float(os.getenv('YOLO_IOU', '0.45'))

_yolo_model_lock = threading.Lock()
_yolo_model: Optional[YOLO] = None
_yolo_model_path: Optional[Path] = None
_failed_weight_paths: set[Path] = set()
_result_counter_lock = threading.Lock()
_result_counter: Optional[int] = None
_snapshot_duration_override: Optional[float] = None
_snapshot_auto_disable = True
_snapshot_saved_path: Optional[Path] = None
_snapshot_saved_timestamp = 0.0

CONFIDENCE_THRESHOLD = 0.35  # kept for potential future use
PIXELS_PER_CM = 75.0
FRAME_QUEUE_SIZE = int(os.getenv('VIDEO_FRAME_QUEUE_SIZE', '6'))
RESULT_QUEUE_SIZE = int(os.getenv('VIDEO_RESULT_QUEUE_SIZE', '6'))
ENABLE_MEASUREMENT = os.getenv('ENABLE_MEASUREMENT', 'true').lower() not in {'0', 'false', 'no', 'on'}

AI_RECOGNITION_DEFAULT = os.getenv('ENABLE_AI_RECOGNITION', 'true').lower() not in {'0', 'false', 'no', 'on'}

_ai_recognition_enabled = AI_RECOGNITION_DEFAULT

if _ai_recognition_enabled:
    _inference_state.update({
        'mode': 'initializing',
        'detail': 'Preparing inference pipeline…',
        'error': '',
        'model_path': None,
        'timestamp': time.time(),
    })
else:
    print('[video_meas] AI recognition disabled on startup; streaming raw frames only.', flush=True)
    _inference_state.update({
        'mode': 'disabled',
        'detail': 'AI recognition disabled.',
        'error': '',
        'model_path': None,
        'timestamp': time.time(),
    })

_ai_enabled_event_proxy: Optional[Any] = None
_capture_pause_event_proxy: Optional[Any] = None

WORKER_SCRIPT = Path(os.getenv('VIDEO_WORKER', _BASE / 'video_worker.py'))

class _VideoManager(BaseManager):
    """Manager class for sharing queues and events between video processing workers."""
    pass

def _get_frame_queue():
    return _manager_state['frame_queue']

def _get_result_queue():
    return _manager_state['result_queue']

def _get_stop_event():
    return _manager_state['stop_event']

def _get_ai_enabled_event():
    return _manager_state['ai_enabled_event']

def _get_capture_pause_event():
    return _manager_state['capture_pause_event']

def _get_stream_stop_event():
    return _manager_state['stream_stop_event']

_VideoManager.register('get_frame_queue', callable=_get_frame_queue)
_VideoManager.register('get_result_queue', callable=_get_result_queue)
_VideoManager.register('get_stop_event', callable=_get_stop_event)
_VideoManager.register('get_ai_enabled_event', callable=_get_ai_enabled_event)
_VideoManager.register('get_capture_pause_event', callable=_get_capture_pause_event)
_VideoManager.register('get_stream_stop_event', callable=_get_stream_stop_event)

_manager_state = {
    'frame_queue': None,
    'result_queue': None,
    'stop_event': None,
    'ai_enabled_event': None,
    'capture_pause_event': None,
    'stream_stop_event': None,
}

_manager: Optional[_VideoManager] = None
_manager_address: Optional[Tuple[str, int]] = None
_manager_authkey: Optional[bytes] = None
_capture_queue = None
_result_queue = None
_capture_process: Optional[subprocess.Popen] = None
_detection_process: Optional[subprocess.Popen] = None
_result_thread: Optional[threading.Thread] = None
_shutdown_event = None
_app_shutdown = threading.Event()
_pipeline_lock = threading.Lock()

def _update_stream_stop_event():
    event = _stream_stop_event_proxy or _manager_state.get('stream_stop_event')
    if event is None:
        return
    try:
        if _halt_stream_after_snapshot.is_set():
            event.set()
        else:
            event.clear()
    except Exception:
        pass

REFERENCE_EMPTY_DIR = Path(__file__).with_name('reference_empty')
REFERENCE_MATCH_THRESHOLD = float(os.getenv('REFERENCE_MATCH_THRESHOLD', '0.92'))
REFERENCE_RESIZE = (160, 120)
_reference_histograms: List[np.ndarray] = []
_reference_notice_emitted = False
_reference_lock = threading.Lock()

def _compute_histogram(image: np.ndarray) -> np.ndarray:
    """Return a normalised colour histogram for quick frame similarity checks."""
    resized = cv2.resize(image, REFERENCE_RESIZE, interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def _load_reference_histograms():
    """Preload histograms for frames that represent an empty conveyor."""
    global _reference_notice_emitted
    with _reference_lock:
        if _reference_histograms:
            return
        if not REFERENCE_EMPTY_DIR.exists():
            if not _reference_notice_emitted:
                _reference_notice_emitted = True
            return
        loaded = 0
        for extension in ('*.jpg', '*.jpeg', '*.png'):
            for path in REFERENCE_EMPTY_DIR.glob(extension):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                _reference_histograms.append(_compute_histogram(image))
                loaded += 1
        if loaded and not _reference_notice_emitted:
            _reference_notice_emitted = True

def _frame_matches_reference(frame: np.ndarray) -> bool:
    """Return True if the frame is close to any configured empty-line reference."""
    if not _reference_histograms:
        return False
    frame_hist = _compute_histogram(frame)
    for ref_hist in _reference_histograms:
        similarity = cv2.compareHist(frame_hist.astype(np.float32), ref_hist.astype(np.float32), cv2.HISTCMP_CORREL)
        if similarity >= REFERENCE_MATCH_THRESHOLD:
            return True
    return False

def _start_manager() -> bool:
    global _manager, _manager_address, _manager_authkey
    global _ai_enabled_event_proxy, _capture_pause_event_proxy, _stream_stop_event_proxy
    if _manager is not None:
        return True
    authkey = secrets.token_bytes(16)
    _manager_state['frame_queue'] = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    _manager_state['result_queue'] = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
    _manager_state['stop_event'] = Event()
    _manager_state['ai_enabled_event'] = Event()
    _manager_state['capture_pause_event'] = Event()
    _manager_state['stream_stop_event'] = Event()
    if is_ai_recognition_enabled():
        _manager_state['ai_enabled_event'].set()
    else:
        _manager_state['ai_enabled_event'].clear()
    _ai_enabled_event_proxy = _manager_state['ai_enabled_event']
    _capture_pause_event_proxy = _manager_state['capture_pause_event']
    _stream_stop_event_proxy = _manager_state['stream_stop_event']
    _update_stream_stop_event()
    try:
        manager = _VideoManager(address=('127.0.0.1', 0), authkey=authkey)
        manager.start()
    except Exception as exc:
        print(f'[video_meas] failed to start IPC manager: {exc}', flush=True)
        traceback.print_exc()
        _manager_state['frame_queue'] = None
        _manager_state['result_queue'] = None
        _manager_state['stop_event'] = None
        _manager_state['ai_enabled_event'] = None
        _manager_state['capture_pause_event'] = None
        _manager_state['stream_stop_event'] = None
        _ai_enabled_event_proxy = None
        _capture_pause_event_proxy = None
        _stream_stop_event_proxy = None
        return False
    address = manager.address
    host: str
    port: int
    if isinstance(address, tuple) and len(address) == 2:
        host = str(address[0])
        port = int(address[1])
    else:
        print(f'[video_meas] unexpected manager address type: {address}', flush=True)
        try:
            manager.shutdown()
        except Exception:
            pass
        _manager_state['frame_queue'] = None
        _manager_state['result_queue'] = None
        _manager_state['stop_event'] = None
        _manager_state['ai_enabled_event'] = None
        _manager_state['capture_pause_event'] = None
        _manager_state['stream_stop_event'] = None
        return False
    _manager = manager
    _manager_address = (host, port)
    _manager_authkey = authkey
    return True

def _stop_manager():
    global _manager, _manager_address, _manager_authkey
    global _ai_enabled_event_proxy, _capture_pause_event_proxy, _stream_stop_event_proxy
    if _manager is None:
        return
    try:
        _manager.shutdown()
    except Exception:
        pass
    _manager = None
    _manager_address = None
    _manager_authkey = None
    _manager_state['frame_queue'] = None
    _manager_state['result_queue'] = None
    _manager_state['stop_event'] = None
    _manager_state['ai_enabled_event'] = None
    _manager_state['capture_pause_event'] = None
    _manager_state['stream_stop_event'] = None
    _ai_enabled_event_proxy = None
    _capture_pause_event_proxy = None
    _stream_stop_event_proxy = None

def _launch_worker(role: str) -> Optional[subprocess.Popen]:
    if not WORKER_SCRIPT.exists():
        print(f'[video_meas] worker script not found at {WORKER_SCRIPT}', flush=True)
        return None
    if _manager_address is None or _manager_authkey is None:
        print('[video_meas] manager not initialised; cannot spawn workers', flush=True)
        return None
    host, port = _manager_address
    if not host or not port:
        print(f'[video_meas] invalid manager address: {host}:{port}', flush=True)
        return None
    auth_hex = binascii.hexlify(_manager_authkey).decode('ascii')
    command = [
        sys.executable,
        '-u',
        str(WORKER_SCRIPT),
        role,
        '--host', host,
        '--port', str(port),
        '--authkey', auth_hex,
    ]
    try:
        return subprocess.Popen(command, cwd=str(WORKER_SCRIPT.parent))
    except Exception as exc:
        print(f'[video_meas] failed to launch {role} worker: {exc}', flush=True)
        traceback.print_exc()
        return None

def set_measurement_callback(callback):
    """Register a callback that receives (width_cm, height_cm)."""
    global measurement_callback
    measurement_callback = callback

def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off'}
    return bool(value)

def _set_inference_state(mode: str, detail: str = '', *, error: str = '', model_path: Optional[Path] = None):
    with _inference_state_lock:
        _inference_state.update({
            'mode': mode,
            'detail': detail,
            'error': error,
            'model_path': str(model_path) if model_path else None,
            'timestamp': time.time(),
        })

def get_inference_status() -> dict:
    with _inference_state_lock:
        snapshot = dict(_inference_state)
    snapshot['ai_enabled'] = is_ai_recognition_enabled()
    snapshot['local_model_error'] = _local_model_error
    snapshot['roboflow_ready'] = False
    snapshot['workspace'] = None
    snapshot['workflow'] = None
    with _ai_detection_lock:
        active = _snapshot_payload_bytes is not None
        remaining = max(0.0, _snapshot_expiry - time.time()) if active else 0.0
        saved_path = str(_snapshot_saved_path) if _snapshot_saved_path else None
        saved_time = datetime.fromtimestamp(_snapshot_saved_timestamp).isoformat() if _snapshot_saved_timestamp else None
        auto_disable = _snapshot_auto_disable
    snapshot['snapshot_active'] = active
    snapshot['snapshot_seconds_remaining'] = remaining
    snapshot['snapshot_saved_path'] = saved_path
    snapshot['snapshot_saved_time'] = saved_time
    snapshot['snapshot_auto_disable'] = auto_disable
    with _frame_stats_lock:
        snapshot['frame_counter'] = _frame_counter
        snapshot['detection_counter'] = _detection_counter
        snapshot['snapshot_requests'] = _snapshot_counter
    return snapshot

def is_ai_recognition_enabled() -> bool:
    event = _ai_enabled_event_proxy
    if event is not None:
        try:
            return bool(event.is_set())
        except Exception:
            pass
    with _ai_recognition_lock:
        return _ai_recognition_enabled

def set_ai_recognition_enabled(enabled: bool) -> bool:
    global _ai_recognition_enabled, _local_model_error, _ai_detection_requested
    global _snapshot_payload_bytes, _snapshot_best_pair, _snapshot_expiry
    global _snapshot_duration_override, _snapshot_auto_disable
    global _stream_active
    changed = False
    with _ai_recognition_lock:
        if _ai_recognition_enabled != enabled:
            _ai_recognition_enabled = enabled
            changed = True
    with _ai_detection_lock:
        if enabled:
            _ai_detection_requested = True
            _stream_active = True
            if AUTO_STOP_STREAM_ON_ENABLE:
                _halt_stream_after_snapshot.set()
            else:
                _halt_stream_after_snapshot.clear()
        else:
            _ai_detection_requested = False
            _stream_active = True  # resume stream immediately on disable
            _halt_stream_after_snapshot.clear()
        _snapshot_payload_bytes = None
        _snapshot_best_pair = None
        _snapshot_expiry = 0.0
        if not enabled:
            _snapshot_duration_override = None
            _snapshot_auto_disable = True
    _update_stream_stop_event()
    event = _ai_enabled_event_proxy
    if event is not None:
        try:
            if enabled:
                event.set()
            else:
                event.clear()
        except Exception:
            pass
    if changed:
        if enabled:
            _local_model_error = ''
            _set_inference_state('initializing', 'Preparing inference pipeline…')
            _preload_local_model()
        else:
            _set_inference_state('disabled', 'AI recognition disabled.')
        state = 'enabled' if enabled else 'disabled'
        log.info('AI recognition %s', state)
    if enabled and changed:
        try:
            status = request_yolo_snapshot(force=True, auto_disable=True)
            log.debug('Snapshot scheduled on enable: %s', status)
        except Exception as exc:
            log.exception('Failed to schedule snapshot after enabling AI: %s', exc)
    return is_ai_recognition_enabled()

# ---------------- YOLO loader and detections ----------------
_YOLO_PRIORITY_PREFIXES = ('yolov12', 'yolo12', 'yolov11', 'yolo11', 'best', 'last')

def _collect_weight_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    candidates: List[Path] = []
    try:
        for path in directory.rglob('*.pt'):
            if path.is_file():
                candidates.append(path)
        for path in directory.rglob('*.pth'):
            if path.is_file():
                candidates.append(path)
    except Exception as exc:
        log.exception('Failed to scan %s for YOLO weights: %s', directory, exc)
    return candidates

def _select_weight_candidate(candidates: List[Path]) -> Optional[Path]:
    ranked: List[Tuple[int, float, Path]] = []
    for path in candidates:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        except Exception as exc:
            log.exception('Failed to stat %s: %s', path, exc)
            continue
        name = path.name.lower()
        priority = len(_YOLO_PRIORITY_PREFIXES)
        for idx, prefix in enumerate(_YOLO_PRIORITY_PREFIXES):
            if name.startswith(prefix):
                priority = idx
                break
        ranked.append((priority, -stat.st_mtime, path))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]

def _resolve_yolo_weights() -> Path:
    candidates: List[Path] = []
    if PREFERRED_BEST_MODEL.exists() and PREFERRED_BEST_MODEL.is_file():
        if PREFERRED_BEST_MODEL not in _failed_weight_paths:
            return PREFERRED_BEST_MODEL
    specific_path = Path('/home/pi/ConveyorBelt-mqtt/Venv/models/best_model.pt')
    if specific_path.exists() and specific_path.is_file():
        if specific_path not in _failed_weight_paths:
            return specific_path  # Prioritize the specific path
    if YOLO_MODEL_PATH_ENV:
        configured = Path(YOLO_MODEL_PATH_ENV).expanduser()
        if configured.is_file():
            if configured.suffix.lower() in {'.pt', '.pth'}:
                if configured not in _failed_weight_paths:
                    return configured
            log.error('YOLO_MODEL_PATH is not a .pt or .pth file: %s', configured)
        elif configured.is_dir():
            candidates.extend(_collect_weight_files(configured))
        else:
            log.error('YOLO_MODEL_PATH does not exist: %s', configured)
    search_dirs: List[Path] = []
    if YOLO_MODEL_DIR_ENV:
        search_dirs.append(Path(YOLO_MODEL_DIR_ENV).expanduser())
    search_dirs.append(DEFAULT_MODELS_DIR)
    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        for candidate in _collect_weight_files(directory):
            if candidate in _failed_weight_paths:
                continue
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    selected = _select_weight_candidate(candidates)
    if selected:
        return selected
    if DEFAULT_YOLO_WEIGHTS.exists() and DEFAULT_YOLO_WEIGHTS not in _failed_weight_paths:
        return DEFAULT_YOLO_WEIGHTS
    raise FileNotFoundError('No YOLO weights (.pt or .pth) found in configured directories.')

def _load_yolo_model() -> YOLO:
    global _yolo_model, _yolo_model_path, _local_model_error, _failed_weight_paths
    with _yolo_model_lock:
        if _yolo_model is not None:
            return _yolo_model
        model: Optional[YOLO] = None
        weights_path: Optional[Path] = None
        while model is None:
            try:
                weights_path = _resolve_yolo_weights()
            except Exception as exc:
                _record_model_failure(exc)
                raise
            try:
                model = YOLO(str(weights_path))
            except KeyError as e:
                if str(e) == "'model'":
                    log.warning('Skipping non-Ultralytics checkpoint at %s; attempting fallback weights.', weights_path)
                    _failed_weight_paths.add(weights_path)
                    model = None
                    continue
                _record_model_failure(e)
                raise
            except Exception as exc:
                log.exception('Failed to load YOLO model from %s: %s', weights_path, exc)
                _record_model_failure(exc)
                raise
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model.to(device)
        except Exception:
            try:
                model.model.to(device)  # older Ultralytics builds
            except Exception:
                pass
        _yolo_model = model
        _yolo_model_path = weights_path
        _local_model_error = ''
        _set_inference_state('local', f'YOLO model: {weights_path.name}', model_path=weights_path)
        log.info('YOLO model loaded from %s (device=%s)', weights_path, device)
        return model

def _preload_local_model(detail_prefix: str = 'YOLO model') -> None:
    try:
        _load_yolo_model()
    except Exception as exc:
        log.debug('Model preload skipped: %s', exc)
        return
    path = _current_model_path()
    if path is not None:
        _set_inference_state('local', f'{detail_prefix}: {path.name}', model_path=path)
        log.debug('Model preloaded for snapshot flow: %s', path)
    else:
        _set_inference_state('local', detail_prefix)


def _record_model_failure(exc: Exception):
    global _local_model_error
    message = str(exc)
    if message != _local_model_error:
        log.error('Local YOLO model unavailable: %s', message)
    _local_model_error = message
    _set_inference_state('error', 'Local YOLO model unavailable.', error=message)

def _get_yolo_detections(frame: np.ndarray) -> Tuple[List[dict], Optional[Path]]:
    model = _load_yolo_model()
    weights_path = _yolo_model_path
    detections: List[dict] = []
    frame_input = np.ascontiguousarray(frame)
    start = time.perf_counter()
    try:
        results = model.predict(source=frame_input, conf=YOLO_CONFIDENCE, iou=YOLO_IOU, verbose=False)
    except Exception as exc:
        log.exception('YOLO inference failed: %s', exc)
        raise RuntimeError(f'YOLO inference failed: {exc}') from exc
    if not results:
        log.debug('YOLO inference returned no results (time=%.3fs)', time.perf_counter() - start)
        return detections, weights_path
    result = results[0]
    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        log.debug('YOLO inference returned no boxes (time=%.3fs)', time.perf_counter() - start)
        return detections, weights_path
    names = result.names or getattr(model, 'names', None)
    xyxy = getattr(boxes, 'xyxy', None)
    confs = getattr(boxes, 'conf', None)
    classes = getattr(boxes, 'cls', None)
    if xyxy is None or confs is None or classes is None:
        return detections, weights_path
    xyxy_np = xyxy.cpu().numpy()
    confs_np = confs.cpu().numpy()
    classes_np = classes.cpu().numpy()
    for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy_np, confs_np, classes_np):
        confidence = float(conf)
        if confidence < YOLO_CONFIDENCE:
            continue
        label: Optional[str] = None
        if isinstance(names, dict):
            label = names.get(int(cls_idx))
        elif isinstance(names, list):
            idx = int(cls_idx)
            if 0 <= idx < len(names):
                label = names[idx]
        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": confidence,
            "label": label or f"class_{int(cls_idx)}",
        })
    log.debug('YOLO detections=%d (time=%.3fs, weights=%s)', len(detections), time.perf_counter() - start, weights_path)
    return detections, weights_path
# ---------------------------------------------------------------

def _prepare_detections(frame: np.ndarray) -> List[dict]:
    if not is_ai_recognition_enabled():
        _set_inference_state('disabled', 'AI recognition disabled.')
        return []
    try:
        detections, weights_path = _get_yolo_detections(frame)
        detail = f'YOLO model: {weights_path.name}' if weights_path else 'YOLO model'
        if detections:
            _set_inference_state('local', detail, model_path=weights_path)
        else:
            _set_inference_state('local', f'{detail} (no detections)', model_path=weights_path)
        return detections
    except Exception as exc:
        _record_model_failure(exc)
        return []

def _current_model_path() -> Optional[Path]:
    return _yolo_model_path

def _parse_result_index(name: str) -> Optional[int]:
    if not name.startswith('result_yolo_'):
        return None
    suffix = name.split('result_yolo_')[-1]
    if suffix.isdigit():
        return int(suffix)
    return None

def _next_result_directory() -> Tuple[Path, int]:
    global _result_counter
    with _result_counter_lock:
        if _result_counter is None:
            max_existing = 0
            try:
                for entry in YOLO_RESULTS_ROOT.iterdir():
                    if not entry.is_dir():
                        continue
                    parsed = _parse_result_index(entry.name)
                    if parsed is not None:
                        max_existing = max(max_existing, parsed)
            except FileNotFoundError:
                pass
            _result_counter = max_existing + 1
        index = _result_counter
        _result_counter += 1
    destination = YOLO_RESULTS_ROOT / f'result_yolo_{index:02d}'
    destination.mkdir(parents=True, exist_ok=True)
    return destination, index

def _store_yolo_snapshot(image: np.ndarray) -> Optional[Path]:
    try:
        destination, _ = _next_result_directory()
    except Exception as exc:
        log.exception('Failed to allocate results directory: %s', exc)
        return None
    filename = datetime.now().strftime('annotated_%Y%m%d_%H%M%S.jpg')
    path = destination / filename
    ok = cv2.imwrite(str(path), image)
    if not ok:
        log.error('Failed to write snapshot image to %s', path)
        return None
    log.info('YOLO snapshot saved to %s', path)
    return path

def _snapshot_state() -> dict:
    with _ai_detection_lock:
        active = _snapshot_payload_bytes is not None
        expiry = _snapshot_expiry
        duration_override = _snapshot_duration_override
        auto_disable = _snapshot_auto_disable
        saved_path = _snapshot_saved_path
        saved_timestamp = _snapshot_saved_timestamp
    with _frame_stats_lock:
        frame_count = _frame_counter
        detection_count = _detection_counter
        snapshot_count = _snapshot_counter
    remaining = max(0.0, expiry - time.time()) if active else 0.0
    return {
        'active': active,
        'seconds_remaining': remaining,
        'duration_override': duration_override,
        'auto_disable': auto_disable,
        'saved_path': str(saved_path) if saved_path else None,
        'saved_time': datetime.fromtimestamp(saved_timestamp).isoformat() if saved_timestamp else None,
        'frame_counter': frame_count,
        'detection_counter': detection_count,
        'snapshot_counter': snapshot_count,
    }


def _persist_snapshot_summary(snapshot_path: Optional[Path], detections: List[dict], best_pair: Optional[Tuple[float, float]], weights_path: Optional[Path]) -> None:
    summary = {
        'timestamp': datetime.now().isoformat(),
        'snapshot_path': str(snapshot_path) if snapshot_path else None,
        'weights_path': str(weights_path) if weights_path else None,
        'detection_count': len(detections),
        'detections': detections,
        'measurement_cm': {
            'width': best_pair[0],
            'height': best_pair[1],
        } if best_pair else None,
        'snapshot_state': _snapshot_state(),
        'inference_state': get_inference_status(),
    }
    try:
        results_path = YOLO_RESULTS_ROOT / 'ai_snapshot_now.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open('w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
        log.debug('Updated snapshot summary at %s', results_path)
    except Exception as exc:
        log.exception('Failed to write snapshot summary: %s', exc)

def request_yolo_snapshot(*, duration: Optional[float] = None, auto_disable: bool = True, force: bool = False) -> dict:
    now = time.time()
    global _snapshot_payload_bytes, _snapshot_best_pair, _snapshot_expiry
    global _snapshot_duration_override, _snapshot_auto_disable
    global _snapshot_saved_path, _snapshot_saved_timestamp, _ai_detection_requested
    log.info('Snapshot request received (duration=%s, auto_disable=%s, force=%s)', duration, auto_disable, force)
    with _ai_detection_lock:
        active = _snapshot_payload_bytes is not None and _snapshot_expiry > now
        if active and not force:
            status = {
                'accepted': False,
                'reason': 'snapshot_already_active',
                'status': _snapshot_state(),
            }
            log.warning('Snapshot request rejected: snapshot already active.')
            return status
        if duration is not None:
            try:
                duration = max(1.0, float(duration))
            except (TypeError, ValueError):
                log.warning('Invalid snapshot duration provided (%s); falling back to default.', duration)
                duration = None
        _snapshot_payload_bytes = None
        _snapshot_best_pair = None
        _snapshot_expiry = 0.0
        _snapshot_duration_override = duration
        _snapshot_auto_disable = bool(auto_disable)
        _snapshot_saved_path = None
        _snapshot_saved_timestamp = 0.0
        _ai_detection_requested = True
        status = _snapshot_state()
        with _frame_stats_lock:
            global _snapshot_counter
            _snapshot_counter += 1
            status['request_id'] = _snapshot_counter
    if not is_ai_recognition_enabled():
        set_ai_recognition_enabled(True)
    else:
        with _ai_detection_lock:
            _ai_detection_requested = True
    log.info('Snapshot scheduled (request_id=%s)', status.get('request_id'))
    status.update({'accepted': True, 'reason': 'scheduled'})
    return status

def cancel_active_snapshot(*, disable_ai: bool = False) -> dict:
    global _snapshot_payload_bytes, _snapshot_best_pair, _snapshot_expiry
    global _snapshot_duration_override
    log.info('Cancel snapshot requested (disable_ai=%s)', disable_ai)
    canceled = False
    with _ai_detection_lock:
        if _snapshot_payload_bytes is not None:
            canceled = True
        _snapshot_payload_bytes = None
        _snapshot_best_pair = None
        _snapshot_expiry = 0.0
        _snapshot_duration_override = None
    if disable_ai:
        set_ai_recognition_enabled(False)
    status = _snapshot_state()
    status.update({'canceled': canceled})
    if canceled:
        log.info('Active snapshot canceled.')
    else:
        log.info('No active snapshot to cancel.')
    return status

def _annotate_with_detections(frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    _load_reference_histograms()
    if _reference_histograms and _frame_matches_reference(frame):
        annotated_empty = frame.copy()
        cv2.putText(
            annotated_empty,
            "Line clear (reference)",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated_empty, None
    detections = _prepare_detections(frame)
    annotated = frame.copy()
    if not is_ai_recognition_enabled():
        cv2.putText(
            annotated,
            "AI recognition disabled",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated, None
    if not detections:
        return annotated, None
    annotated, best_pair = annotate_frame(annotated, detections, PIXELS_PER_CM, ENABLE_MEASUREMENT)
    if not ENABLE_MEASUREMENT:
        cv2.putText(
            annotated,
            "Measurements disabled",
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )
    return annotated, best_pair

def _capture_worker(frame_queue: Any, stop_event: Any, pause_event: Any = None):
    log.info('Capture worker starting (pid=%s)', os.getpid())
    command = [
        'libcamera-vid', '--inline', '--timeout', '0', '--framerate', '10',
        '--width', '640', '--height', '480', '--codec', 'mjpeg', '--output', '-'
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    bytes_buffer = b''
    paused_logged = False
    try:
        while not stop_event.is_set():
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            bytes_buffer += chunk
            while True:
                start = bytes_buffer.find(b'\xff\xd8')
                end = bytes_buffer.find(b'\xff\xd9')
                if start == -1 or end == -1:
                    break
                jpg = bytes_buffer[start:end + 2]
                bytes_buffer = bytes_buffer[end + 2:]
                if pause_event is not None:
                    try:
                        paused = pause_event.is_set()
                    except Exception:
                        paused = False
                    if paused:
                        if not paused_logged:
                            log.info('Capture paused while YOLO detection running.')
                            paused_logged = True
                        continue
                    if paused_logged:
                        log.info('Capture resumed after YOLO detection.')
                        paused_logged = False
                try:
                    frame_queue.put_nowait(jpg)
                except queue.Full:
                    log.warning('Capture queue full; dropping frame.')
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_queue.put_nowait(jpg)
                    except queue.Full:
                        log.error('Capture queue still full after drop.')
    except Exception as exc:
        log.exception('Capture worker error: %s', exc)
    finally:
        stop_event.set()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        rc = process.poll()
        stderr_output = ''
        if process.stderr is not None:
            try:
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore').strip()
            except Exception:
                stderr_output = ''
        log.info('Capture worker exiting (code=%s)', rc)
        if stderr_output:
            log.warning('Capture stderr:\n%s', stderr_output)

def _detection_worker(frame_queue: Any, result_queue: Any, stop_event: Any, ai_enabled_event: Any, capture_pause_event: Any, stream_stop_event: Any):
    log.info('Detection worker starting (pid=%s)', os.getpid())
    global latest_raw_frame, _ai_enabled_event_proxy, _ai_detection_requested
    global _snapshot_payload_bytes, _snapshot_best_pair, _snapshot_expiry
    global _snapshot_duration_override, _snapshot_auto_disable
    global _snapshot_saved_path, _snapshot_saved_timestamp
    global _frame_counter, _detection_counter, _snapshot_counter
    global _last_queue_log, _last_detection_log
    global _stream_active, _stream_stop_event_proxy
    _ai_enabled_event_proxy = ai_enabled_event
    global _capture_pause_event_proxy
    _capture_pause_event_proxy = capture_pause_event
    _stream_stop_event_proxy = stream_stop_event

    def _set_capture_pause(active: bool):
        event = _capture_pause_event_proxy
        if event is None:
            return
        try:
            if active:
                event.set()
            else:
                event.clear()
        except Exception:
            pass

    def _drain_frame_queue() -> int:
        drained = 0
        if frame_queue is None:
            return drained
        while True:
            try:
                frame_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            log.debug('Drained %d queued frames before detection.', drained)
        return drained

    def _should_halt_stream() -> bool:
        event = _stream_stop_event_proxy
        if event is None:
            return False
        try:
            return bool(event.is_set())
        except Exception:
            return False

    def _mark_stream_halted() -> None:
        global _stream_active
        with _ai_detection_lock:
            _stream_active = False

    while not stop_event.is_set():
        try:
            jpg = frame_queue.get(timeout=0.5)
        except queue.Empty:
            now = time.time()
            if QUEUE_EMPTY_LOG_INTERVAL > 0:
                if (_last_queue_log == 0.0) or (now - _last_queue_log) >= QUEUE_EMPTY_LOG_INTERVAL:
                    log.warning('Detection worker waiting for frames... ai_enabled=%s requested=%s',
                                is_ai_recognition_enabled(), _ai_detection_requested)
                    _last_queue_log = now
            continue

        try:
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                log.error('Failed to decode frame (bytes=%s)', len(jpg) if jpg else 0)
                continue
            with raw_frame_lock:
                latest_raw_frame = frame.copy()
            with _frame_stats_lock:
                _frame_counter += 1
                frame_index = _frame_counter
            if FRAME_DEBUG_INTERVAL and frame_index % FRAME_DEBUG_INTERVAL == 0:
                log.debug('Frame %d decoded (%dx%d)', frame_index, frame.shape[1], frame.shape[0])

            now = time.time()
            with _ai_detection_lock:
                detection_requested = _ai_detection_requested and is_ai_recognition_enabled()
                snapshot_bytes = _snapshot_payload_bytes
                snapshot_best = _snapshot_best_pair
                snapshot_expiry = _snapshot_expiry
                duration_override = _snapshot_duration_override
                auto_disable = _snapshot_auto_disable

            if snapshot_bytes is not None and now >= snapshot_expiry:
                with _ai_detection_lock:
                    _snapshot_payload_bytes = None
                    _snapshot_best_pair = None
                    _snapshot_expiry = 0.0
                    expired_auto_disable = _snapshot_auto_disable
                if expired_auto_disable:
                    set_ai_recognition_enabled(False)
                log.info('Snapshot expired (auto_disable=%s)', expired_auto_disable)
                snapshot_bytes = None
                snapshot_best = None

            payload_bytes: Optional[bytes] = None
            best_pair: Optional[Tuple[float, float]] = None

            if detection_requested:
                halt_stream = _should_halt_stream()
                detect_start = time.perf_counter()
                try:
                    _set_capture_pause(True)
                    _drain_frame_queue()
                    detections, weights_path = _get_yolo_detections(frame)
                    with _ai_detection_lock:
                        global _snapshot_last_detections
                        _snapshot_last_detections = detections
                except Exception as exc:
                    log.exception('YOLO detection failed: %s', exc)
                    detections = []
                    weights_path = _current_model_path()
                finally:
                    if not halt_stream:
                        _set_capture_pause(False)
                annotated = frame.copy()
                if detections:
                    annotated, best_pair = annotate_frame(annotated, detections, PIXELS_PER_CM, ENABLE_MEASUREMENT)
                else:
                    log.info('YOLO snapshot produced no detections.')
                    cv2.putText(
                        annotated,
                        "YOLO: no components detected",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (14, 165, 233),
                        2,
                        cv2.LINE_AA,
                    )
                stamp = datetime.now().strftime('YOLO snapshot %H:%M:%S')
                cv2.putText(annotated, stamp, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2, cv2.LINE_AA)
                saved_path = _store_yolo_snapshot(annotated)
                fallback_path: Optional[Path] = None
                if saved_path:
                    log.info('Snapshot stored at %s', saved_path)
                ok, encoded = cv2.imencode('.jpg', annotated)
                if not ok:
                    log.error('Failed to encode annotated snapshot frame.')
                    continue
                payload_bytes = encoded.tobytes()
                duration = duration_override if duration_override is not None else AI_SNAPSHOT_DURATION
                expiry = time.time() + duration
                with _ai_detection_lock:
                    _ai_detection_requested = False
                    _snapshot_payload_bytes = payload_bytes
                    _snapshot_best_pair = best_pair
                    _snapshot_expiry = expiry
                    _snapshot_duration_override = None
                    resolved_path = saved_path
                    if resolved_path is not None:
                        global _snapshot_saved_path, _snapshot_saved_timestamp
                        _snapshot_saved_path = resolved_path
                        _snapshot_saved_timestamp = time.time()
                detect_time = time.perf_counter() - detect_start
                with _frame_stats_lock:
                    _detection_counter += 1
                    detection_index = _detection_counter
                    _last_detection_log = time.time()
                log.info('YOLO snapshot #%d ready (detections=%d, time=%.3fs, duration=%.1fs, auto_disable=%s)',
                         detection_index, len(detections), detect_time, duration, auto_disable)
                detail = f'YOLO snapshot active ({duration:.0f}s)'
                _set_inference_state('snapshot', detail, model_path=weights_path)
                try:
                    DETECTION_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    snap_name = datetime.now().strftime('snapshot_%Y%m%d_%H%M%S.jpg')
                    snapshot_path = DETECTION_SNAPSHOT_DIR / snap_name
                    if cv2.imwrite(str(snapshot_path), annotated):
                        fallback_path = snapshot_path
                    else:
                        log.error('Failed to store snapshot at %s', snapshot_path)
                except Exception as exc:
                    log.exception('Failed to store snapshot copy: %s', exc)
                if fallback_path is not None and saved_path is None:
                    with _ai_detection_lock:
                        _snapshot_saved_path = fallback_path
                        _snapshot_saved_timestamp = time.time()
                    log.info('Fallback snapshot stored at %s', fallback_path)
                final_snapshot_path = saved_path or fallback_path
                result_record = {
                    'timestamp': datetime.now().isoformat(),
                    'weights_path': str(weights_path) if weights_path else None,
                    'detection_count': len(detections),
                    'detections': detections,
                    'measurement_cm': {
                        'width': best_pair[0],
                        'height': best_pair[1],
                    } if best_pair else None,
                    'snapshot_path': str(final_snapshot_path) if final_snapshot_path else None,
                }
                try:
                    DETECTION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    result_filename = datetime.now().strftime('detections_%Y%m%d_%H%M%S.json')
                    result_path = DETECTION_RESULTS_DIR / result_filename
                    with result_path.open('w', encoding='utf-8') as fh:
                        json.dump(result_record, fh, indent=2)
                    log.info('Detection results written to %s', result_path)
                except Exception as exc:
                    log.exception('Failed to write detection results: %s', exc)
                _persist_snapshot_summary(final_snapshot_path, detections, best_pair, weights_path)
                if final_snapshot_path is not None:
                    try:
                        BEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                        best_path = BEST_RESULTS_DIR / Path(final_snapshot_path).name
                        shutil.copy2(final_snapshot_path, best_path)
                        log.info('Best results snapshot saved to %s', best_path)
                    except Exception as exc:
                        log.exception('Failed to copy snapshot to best results: %s', exc)
                if halt_stream:
                    log.info('Halting stream after snapshot per configuration.')
                    _mark_stream_halted()
                    summary_detail = f'Snapshot captured ({len(detections)} detections); stream halted. Disable AI or set AUTO_STOP_STREAM_AFTER_ENABLE=0 to resume.'
                    _set_inference_state('halted', summary_detail, model_path=weights_path)
                    try:
                        stop_event.set()
                    except Exception:
                        pass
                    break
            elif snapshot_bytes is not None:
                remaining = max(0.0, snapshot_expiry - now)
                _set_inference_state('snapshot', f'YOLO snapshot ({remaining:.0f}s left)', model_path=_current_model_path())
                payload_bytes = snapshot_bytes
                best_pair = None  # avoid repeated measurement callbacks
                log.debug('Snapshot stream refresh (remaining=%.2fs)', remaining)
            else:
                ok, encoded = cv2.imencode('.jpg', frame)
                if not ok:
                    log.error('Failed to encode live frame for streaming.')
                    continue
                payload_bytes = encoded.tobytes()
                if is_ai_recognition_enabled():
                    _set_inference_state('local', 'Awaiting YOLO snapshot trigger…', model_path=_current_model_path())
                    if FRAME_DEBUG_INTERVAL and frame_index % FRAME_DEBUG_INTERVAL == 0:
                        log.debug('AI enabled, awaiting snapshot trigger (frame %d).', frame_index)
                else:
                    _set_inference_state('disabled', 'AI recognition idle.')

            if payload_bytes is None:
                continue
            payload = (payload_bytes, best_pair)
            try:
                result_queue.put(payload, timeout=0.5)
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put(payload, timeout=0.1)
                except queue.Full:
                    pass
        except Exception as exc:
            log.exception('Detection worker error: %s', exc)
    _set_capture_pause(False)
    log.info('Detection worker exiting')

def _result_collector(result_queue: Any, stop_event: Any):
    global latest_frame
    while True:
        try:
            should_stop = stop_event.is_set()
        except (EOFError, ConnectionError, BrokenPipeError):
            break
        try:
            encoded, best_pair = result_queue.get(timeout=0.5)
        except queue.Empty:
            if should_stop:
                try:
                    if result_queue.empty():
                        break
                except (EOFError, ConnectionError, BrokenPipeError):
                    break
            continue
        with frame_lock:
            latest_frame = encoded
        if measurement_callback is not None and best_pair is not None:
            try:
                measurement_callback(*best_pair)
            except Exception as exc:
                print(f'[video_meas] measurement callback failed: {exc}', flush=True)

def _stop_pipeline_locked():
    global _capture_queue, _result_queue, _capture_process, _detection_process, _result_thread, _shutdown_event
    if _shutdown_event is not None:
        try:
            _shutdown_event.set()
        except Exception:
            pass
    if _capture_process is not None:
        try:
            _capture_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _capture_process.terminate()
            try:
                _capture_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _capture_process.kill()
        _capture_process = None
    if _detection_process is not None:
        try:
            _detection_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _detection_process.terminate()
            try:
                _detection_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _detection_process.kill()
        _detection_process = None
    if _result_thread is not None:
        _result_thread.join(timeout=2)
        _result_thread = None
    _capture_queue = None
    _result_queue = None
    _shutdown_event = None
    _stop_manager()

def _start_pipeline_locked() -> bool:
    global _capture_queue, _result_queue, _capture_process, _detection_process, _result_thread, _shutdown_event
    if _app_shutdown.is_set():
        return False
    if _capture_process and _capture_process.poll() is None and _detection_process and _detection_process.poll() is None:
        return True
    _stop_pipeline_locked()
    if not _start_manager():
        return False
    try:
        _capture_queue = _manager.get_frame_queue()
        _result_queue = _manager.get_result_queue()
        _shutdown_event = _manager.get_stop_event()
    except Exception as exc:
        print(f'[video_meas] failed to acquire IPC handles: {exc}', flush=True)
        traceback.print_exc()
        _stop_pipeline_locked()
        return False
    _capture_process = _launch_worker('capture')
    _detection_process = _launch_worker('detection')
    if _capture_process is None or _detection_process is None:
        _stop_pipeline_locked()
        return False
    _result_thread = threading.Thread(target=_result_collector, args=(_result_queue, _shutdown_event), daemon=True)
    _result_thread.start()
    print(f'[video_meas] pipeline started (capture_pid={_capture_process.pid}, detect_pid={_detection_process.pid})', flush=True)
    return True


def _start_pipeline() -> bool:
    with _pipeline_lock:
        return _start_pipeline_locked()

def _stop_pipeline():
    with _pipeline_lock:
        _stop_pipeline_locked()
        print('[video_meas] pipeline stopped', flush=True)


def restart_pipeline(reason: str = '') -> bool:
    suffix = f' ({reason})' if reason else ''
    with _pipeline_lock:
        _stop_pipeline_locked()
        time.sleep(0.2)
        started = _start_pipeline_locked()
    if started:
        print(f'[video_meas] pipeline restarted{suffix}', flush=True)
    else:
        print(f'[video_meas] pipeline restart failed{suffix}', flush=True)
    return started

def _at_exit_cleanup():
    _app_shutdown.set()
    _stop_pipeline()

def capture_measured_video():
    global _stream_active
    while not _app_shutdown.is_set():
        if not _start_pipeline():
            log.warning('Failed to start pipeline; retrying in 2s.')
            time.sleep(2.0)
            continue
        try:
            while not _app_shutdown.is_set():
                time.sleep(1.0)
                capture_dead = _capture_process is None or _capture_process.poll() is not None
                detection_dead = _detection_process is None or _detection_process.poll() is not None
                if capture_dead or detection_dead:
                    log.warning('Pipeline workers stopped (capture_dead=%s detection_dead=%s); restarting.',
                                capture_dead, detection_dead)
                    break
        except KeyboardInterrupt:
            _app_shutdown.set()
        except Exception as exc:
            log.exception('Capture supervisor error: %s', exc)
        finally:
            _stop_pipeline()
        if _halt_stream_after_snapshot.is_set():
            log.info('Stream halt requested; capture supervisor waiting for resume signal.')
            while _halt_stream_after_snapshot.is_set() and not _app_shutdown.is_set():
                time.sleep(0.5)
            with _ai_detection_lock:
                _stream_active = True
            continue
    log.info('Capture supervisor exiting')

def generate_measured_stream():
    global _last_stream_wait_log
    while True:
        frame_copy = None
        with frame_lock:
            if latest_frame is not None:
                frame_copy = latest_frame
        if frame_copy is not None:
            _last_stream_wait_log = 0.0
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_copy + b'\r\n'
            )
            time.sleep(0.03)
        else:
            now = time.time()
            if _last_stream_wait_log == 0.0 or (now - _last_stream_wait_log) >= 5.0:
                log.warning('Waiting for frames to stream...')
                _last_stream_wait_log = now
            time.sleep(0.05)
        # Check if stream should stop
        with _ai_detection_lock:
            if not _stream_active:
                break

@video_bp.route('/inference_status')
def inference_status():
    return jsonify(get_inference_status())

@video_bp.route('/ai_snapshot', methods=['GET'])
def ai_snapshot_status():
    status = _snapshot_state()
    status['ai_enabled'] = is_ai_recognition_enabled()
    status['model_loaded'] = _yolo_model is not None
    status['model_path'] = str(_yolo_model_path) if _yolo_model_path else None
    status['inference_state'] = get_inference_status()
    return jsonify(status)

@video_bp.route('/ai_snapshot', methods=['POST'])
def ai_snapshot_trigger():
    payload = request.get_json(silent=True) or {}
    log.debug('POST /ai_snapshot payload: %s', payload)
    try:
        duration = payload.get('duration')
        auto_disable = _coerce_bool(payload.get('auto_disable'), True)
        force = _coerce_bool(payload.get('force'), False)
        status = request_yolo_snapshot(duration=duration, auto_disable=auto_disable, force=force)
        code = 202 if status.get('accepted') else 409
        return jsonify(status), code
    except Exception as exc:
        log.exception('Snapshot trigger failed: %s', exc)
        return jsonify({'accepted': False, 'reason': 'internal_error', 'message': str(exc)}), 500

@video_bp.route('/ai_snapshot', methods=['DELETE'])
def ai_snapshot_cancel():
    payload = request.get_json(silent=True) or {}
    log.debug('DELETE /ai_snapshot payload: %s', payload)
    try:
        disable_ai = _coerce_bool(payload.get('disable_ai'), False)
        status = cancel_active_snapshot(disable_ai=disable_ai)
        return jsonify(status)
    except Exception as exc:
        log.exception('Snapshot cancel failed: %s', exc)
        return jsonify({'canceled': False, 'reason': 'internal_error', 'message': str(exc)}), 500

@video_bp.route('/ai_snapshot/image')
def ai_snapshot_image():
    with _ai_detection_lock:
        payload = _snapshot_payload_bytes
    if payload is None:
        log.debug('Snapshot image requested but no snapshot available.')
        return jsonify({'error': 'no_snapshot', 'status': _snapshot_state()}), 404
    response = current_app.response_class(payload, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store'
    return response

@video_bp.route('/ai_snapshot/debug')
def ai_snapshot_debug():
    status = _snapshot_state()
    status['ai_enabled'] = is_ai_recognition_enabled()
    status['model_loaded'] = _yolo_model is not None
    status['model_path'] = str(_yolo_model_path) if _yolo_model_path else None
    status['inference_state'] = get_inference_status()
    status['queue_last_log'] = _last_queue_log
    status['detection_last_log'] = _last_detection_log
    return jsonify(status)

@video_bp.route("/measured_video")
def measured_video():
    """Flask route streaming MJPEG frames (annotated when AI recognition is enabled)."""
    return Response(generate_measured_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@video_bp.route('/ai_snapshot/detections')
def ai_snapshot_detections():
    with _ai_detection_lock:
        dets = _snapshot_last_detections
    if not dets:
        return jsonify({"error": "no_snapshot_or_no_detections"}), 404
    return jsonify({"detections": dets})

@video_bp.route('/measured_view')
def measured_view():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Electronics Detection Stream</title>
    </head>
    <body>
      <h1>Electronics Detection Stream</h1>
      <img src="/measured_video" alt="Live Detection" style="width:640px; height:480px;">
    </body>
    </html>
    """
    return html_content

@video_bp.route('/enable_ai', methods=['POST'])
def enable_ai():
    """Route to enable AI detection after clicking 'Enable AI' button."""
    enabled = set_ai_recognition_enabled(True)
    return jsonify({'status': 'enabled' if enabled else 'failed'})

@video_bp.route('/disable_ai', methods=['POST'])
def disable_ai():
    """Route to disable AI detection."""
    enabled = set_ai_recognition_enabled(False)
    return jsonify({'status': 'disabled' if not enabled else 'failed'})

def capture_reference_frame() -> Optional[Path]:
    """Persist the most recent raw frame (without annotations) for empty-line calibration."""
    with raw_frame_lock:
        if latest_raw_frame is None:
            return None
        frame_copy = latest_raw_frame.copy()
    REFERENCE_EMPTY_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    destination = REFERENCE_EMPTY_DIR / filename
    ok = cv2.imwrite(str(destination), frame_copy)
    if not ok:
        return None
    return destination

@video_bp.route('/capture_reference', methods=['POST'])
def capture_reference():
    from flask import jsonify as _jsonify
    path = capture_reference_frame()
    if path is None:
        return _jsonify({"status": "error", "message": "No raw frame available yet."}), 503
    return _jsonify({"status": "ok", "saved_as": path.name})

# -------------- lightweight system monitor (CPU/RAM) --------------
def _sys_monitor():
    while not _app_shutdown.is_set():
        cpu = psutil.cpu_percent(interval=5)
        mem = psutil.virtual_memory()
        log.info("[SYS] CPU %.1f%% | RAM %.1f%% (%.0fMB free)",
                 cpu, mem.percent, mem.available/1048576)

def _start_sys_monitor_once():
    t = threading.Thread(target=_sys_monitor, daemon=True)
    t.start()
# ------------------------------------------------------------------

_start_sys_monitor_once()
atexit.register(_at_exit_cleanup)
