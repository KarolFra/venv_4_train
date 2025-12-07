import signal #for sending messages after ^C to: /home/control/light & DCpwm
import sys #provides functions and variables used by the interpreter; used e.g. 4 sys.exit.
import json #enables works with text formatted in json
import time
import os
import threading #allows to create and manage threads - performing parallel tasks.
from threading import Lock #lock ensures synchro between parrarel data used by threading
from pathlib import Path
import subprocess #enable camera use
import logging
from flask import Flask, jsonify, Response, render_template, request #4 API
from flask import current_app
from flask_mqtt import Mqtt
from db_models import db, SensorData, SystemMetric # to generate charts
# from flask_sqlalchemy import SQLAlchemy
#from sqlalchemy import create_engine
#from sqlalchemy.orm import sessionmaker
#from imutils import perspective, contours
#from scipy.spatial.distance import euclidean
#import os
#from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

app = Flask(__name__, static_folder='static') #in static is located .css file 

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('werkzeug').disabled = True
logging.getLogger('werkzeug').propagate = False
app.logger.disabled = True

#  Import functions from viedo_meas.py 
import video_meas

from video_meas import (
    video_bp,
    capture_measured_video,
    is_ai_recognition_enabled,
    set_ai_recognition_enabled,
    set_measurement_callback,
)

app.register_blueprint(video_bp)  # REQUIRED for routes from video_meas
# if ^^ wouldn t be here, Flask will never know about your blueprints routes.

# global measurement, updated whenever video_meas sees something
captured_measurement = None
captured_timestamp = 0
lock = threading.Lock()
element_state_lock = Lock()
element_present = False
last_element_seen = 0.0
LINE_CLEAR_TIMEOUT = float(os.getenv('LINE_CLEAR_TIMEOUT', '3.0'))
SYSTEM_DEBUG_INTERVAL = float(os.getenv('SYSTEM_DEBUG_INTERVAL', '10'))
CONVEYOR_VERBOSE = os.getenv('CONVEYOR_VERBOSE_LOGS', 'false').lower() not in {'0', 'false', 'no', 'off'}

# Database Setup
from flask_migrate import Migrate

#!!IF DEBUG MQTT COMMENT THIS BELOW!!
#Place this before you call app.run(...). This will stop printing the standard request logs, 
# but also hides all other default logs from the Flask server.
"""import logging
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True
"""
# Configure the app
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sensor_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# create SQLAlchemy instance with the app
db.init_app(app)
# Initialize Flask-Migrate with the app and database
migrate = Migrate(app, db)

with app.app_context():
    print("Creating tables...")
    db.create_all()
    print("Tables created!")

# Global sensor data
sensor_data = {
    'temperature': None,  # DHT11 temperature
    'humidity': None,     # DHT11 humidity
    'distance': None,      # HC-SR04 distance
    'cpu_temp': None       #  Rpi CPU temperature 
}
lock = Lock()
system_metrics = {}
system_metrics_lock = Lock()
is_subscribed = False

# ---- AI model auto-switch strategy ----
CONNECTOR_MODEL_PATH = Path("/home/pi/ConveyorBelt-mqtt/Venv/models/weights20251015/epoch5_connectr.pt")
PRODUCTION_MODEL_PATH = Path("/home/pi/ConveyorBelt-mqtt/Venv/models/weights20251015/epoch50_run_withLedON.pt")
MODEL_STATE_LOCK = Lock()
_model_state = {
    "stage": "waiting_connectors",
    "current_weights": str(CONNECTOR_MODEL_PATH),
    "stage_started_at": time.time(),
    "switched_at": None,
    "switch_reason": "startup",
}
connectors_detected_event = threading.Event()


def _ensure_weight_path(path: Path) -> None:
    if not path.exists():
        print(f"[MODEL] Warning: weight file not found at {path}", flush=True)


def _configure_yolo_weights(weights_path: Path) -> None:
    path_str = str(weights_path)
    os.environ['BEST_MODEL_WEIGHTS'] = path_str
    os.environ['YOLO_MODEL_PATH'] = path_str
    os.environ['YOLO_FORCE_MODEL_PATH'] = path_str
    video_meas.PREFERRED_BEST_MODEL = weights_path
    video_meas.YOLO_MODEL_PATH_ENV = path_str
    failed_set = getattr(video_meas, '_failed_weight_paths', None)
    if isinstance(failed_set, set):
        failed_set.clear()
    lock_obj = getattr(video_meas, '_yolo_model_lock', None)
    if lock_obj is None:
        lock_obj = Lock()
    with lock_obj:
        if hasattr(video_meas, '_yolo_model'):
            video_meas._yolo_model = None
        if hasattr(video_meas, '_yolo_model_path'):
            video_meas._yolo_model_path = None


def initialize_connector_stage() -> None:
    _ensure_weight_path(CONNECTOR_MODEL_PATH)
    _configure_yolo_weights(CONNECTOR_MODEL_PATH)
    connectors_detected_event.clear()
    with MODEL_STATE_LOCK:
        _model_state.update({
            "stage": "waiting_connectors",
            "current_weights": str(CONNECTOR_MODEL_PATH),
            "stage_started_at": time.time(),
            "switched_at": None,
            "switch_reason": "startup",
        })


def get_model_stage() -> str:
    with MODEL_STATE_LOCK:
        return _model_state.get("stage", "unknown")


def is_waiting_for_connectors() -> bool:
    return get_model_stage() == "waiting_connectors"


def switch_to_production_model(reason: str = "manual") -> bool:
    reason = reason or "manual"
    with MODEL_STATE_LOCK:
        if _model_state.get("stage") == "production":
            return False
        _model_state["stage"] = "production"
        _model_state["current_weights"] = str(PRODUCTION_MODEL_PATH)
        _model_state["stage_started_at"] = time.time()
        _model_state["switched_at"] = _model_state["stage_started_at"]
        _model_state["switch_reason"] = reason
    connectors_detected_event.set()
    _ensure_weight_path(PRODUCTION_MODEL_PATH)
    print(f"[MODEL] Connectors detected via {reason}; switching to {PRODUCTION_MODEL_PATH.name}", flush=True)
    _configure_yolo_weights(PRODUCTION_MODEL_PATH)
    success = False
    try:
        success = video_meas.restart_pipeline(f"model_switch:{reason}")
    except AttributeError:
        success = False
    except Exception as exc:
        print(f"[MODEL] restart_pipeline failed: {exc}", flush=True)
    if not success:
        try:
            video_meas._stop_pipeline()
            time.sleep(0.5)
            success = video_meas._start_pipeline()
        except Exception as exc:
            print(f"[MODEL] Manual pipeline restart failed: {exc}", flush=True)
    if success:
        print("[MODEL] Pipeline restarted with production weights.", flush=True)
    set_ai_recognition_enabled(True)
    return True


def get_model_state_snapshot() -> dict:
    with MODEL_STATE_LOCK:
        snapshot = dict(_model_state)
    snapshot["connectors_detected"] = connectors_detected_event.is_set()
    for key in ("stage_started_at", "switched_at"):
        ts = snapshot.get(key)
        if ts:
            snapshot[f"{key}_iso"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
    return snapshot


initialize_connector_stage()

# This function will be called whenever video_meas detects a measurement
def on_new_measurement(width_cm, height_cm):
    now = time.time()
    print(f"[APP] New measurement: {width_cm:.2f} x {height_cm:.2f}")

    global captured_measurement, captured_timestamp, element_present, last_element_seen
    with lock:
        captured_measurement = (width_cm, height_cm)
        captured_timestamp = now

    first_detection = False
    with element_state_lock:
        first_detection = not element_present
        element_present = True
        last_element_seen = now

    if first_detection:
        print(f"[LINE] Element detected ({width_cm:.2f} x {height_cm:.2f} cm) -> stopping conveyor.")
        request_conveyor_stop("element detected")
        switch_to_production_model("measurement")

#"esp32" ~= auto 
control_mode = "esp32"  # Default mode to control the process, ESP32 (bcs at beginning i wanted to make calculations in one of esp)


#temporary set pwm on conveyor for specified time, bcs distance sensor doesnt work correctly
def auto_send_conveyor_pwm():
    """
    If control_mode != "esp32" function exit immediately
    works only for captured_measurement  (width, height) (from video!)
    If older than 15s or missing capture default dimensions = (1,1)
    Map the average dimension [1..3](obj) -> [50..90](pwm's)
    Publish that PWM mapped value, hold it for 10 seconds, then publish 0
    """
    if control_mode != "esp32":
        return  # skip if in manual mode
    #import time
    global mqtt, captured_measurement, captured_timestamp, element_present

    # Immediately set conveyor PWM to 0
    try:
        mqtt.publish('/home/control/conveyorPWM', '0')
    except Exception as exc:
        if CONVEYOR_VERBOSE:
            print(f"[auto_send_conveyor_pwm] failed to enforce idle PWM: {exc}")

    with element_state_lock:
        if element_present:
            # element still on the line – keep conveyor stopped
            return

    now = time.time()
    # If no measurement or older than 15s -> default to 1x1
    if captured_measurement is None or (now - captured_timestamp) > 15:
        width, height = 1, 1
    else:
        width, height = captured_measurement

    # Clamp each dimension to [1 - 3]
    width = max(1, min(width, 3))
    height = max(1, min(height, 3))

    # Compute average dimension
    avg_dim = (width + height) / 2.0

    # map [1 - 3] -> [50 - 90]
    conveyor_pwm = int((avg_dim - 1) * (90 - 50) / (3 - 1) + 50)

    # wait 2 seconds before applying (optionally)
    time.sleep(2)

    # publish the final PWM, 0 for now later implement
    mqtt.publish('/home/control/conveyorPWM', 0)
    if CONVEYOR_VERBOSE:
        print(f"[auto_send_conveyor_pwm] set to {conveyor_pwm} (width={width:.2f}, height={height:.2f}) ignoring distance")

    # hold that PWM for 10 seconds
    time.sleep(10)

    # reset to 0 after 20s, after that time u can put another item on conveyor
    mqtt.publish('/home/control/conveyorPWM', '0')
    if CONVEYOR_VERBOSE:
        print("[auto_send_conveyor_pwm] Held 10s, now reset to 0.")


#  BACKGROUND THREAD FOR PERIODIC UPDATES 
def pwm_update_loop():
    """
    Periodically call auto-send function to hold conveyor PWM values
    """
    while True:
        auto_send_conveyor_pwm()
        time.sleep(1)  # adjust as needed

def handle_sigint(signum, frame):
    print("CTRL+C pressed! Sending 'default' messages to revert ESP32 to automatic mode.")
    try:
        mqtt.publish('/home/control/light', 'OFF')     # Revert LED to a safe state in esp32
        mqtt.publish('/home/control/DCpwm', '0')  # Revert DC motors to sensor based
    except Exception as e:
        print("Error publishing default messages on exit:", e)
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# MQTT Configuration
import socket
local_ip = socket.gethostbyname(socket.gethostname())

if local_ip.startswith('192.168.1.'):
    app.config['MQTT_BROKER_URL'] = '192.168.1.21'
elif local_ip.startswith('192.168.0.'):
    app.config['MQTT_BROKER_URL'] = '192.168.0.121'
else:
    # Default or fallback
    app.config['MQTT_BROKER_URL'] = 'localhost'

print(f"Using MQTT Broker: {app.config['MQTT_BROKER_URL']}")

app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_USERNAME'] = ''
app.config['MQTT_PASSWORD'] = ''
app.config['MQTT_KEEPALIVE'] = 60
app.config['MQTT_TLS_ENABLED'] = False

mqtt = Mqtt(app) #mqtt app init

def send_distance(distance):
    """
    Publishes the provided distance value as a JSON payload to the MQTT topic /home/sensors/distance
    :paramaters distance: float, the distance value to send.
    """
    payload = json.dumps({"distance": distance})
    try:
        mqtt.publish("/home/sensors/distance", payload)
        print(f"Sent distance: {payload} to /home/sensors/distance")
    except Exception as e:
        print(f"Error publishing distance: {e}")

def publish_distance_periodically():
    while True:
        dist = sensor_data.get('distance', None)
        if dist is not None:
            send_distance(dist)
        time.sleep(1)  # Wait 1 second between publishes (adjust as needed)


def request_conveyor_stop(reason: str = ""):
    """Publish a stop command to the conveyor with optional debug context."""
    message = "[CONVEYOR] Stop requested"
    if reason:
        message += f" ({reason})"
    print(message)
    try:
        mqtt.publish('/home/control/conveyorPWM', '0')
    except Exception as exc:
        print(f"[CONVEYOR] Failed to send stop command: {exc}")


def line_state_monitor():
    """Detect when the conveyor becomes empty again and reset state."""
    global element_present, captured_measurement, captured_timestamp
    while True:
        time.sleep(0.5)
        cleared = False
        with element_state_lock:
            if element_present and (time.time() - last_element_seen) > LINE_CLEAR_TIMEOUT:
                element_present = False
                cleared = True
        if cleared:
            with lock:
                captured_measurement = None
                captured_timestamp = 0
            print("[LINE] Line clear - ready for next element.")


def _read_meminfo_snapshot():
    """Return memory totals in kB from /proc/meminfo when psutil is unavailable."""
    try:
        data = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as memfile:
            for line in memfile:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(':')
                if key in {"MemTotal", "MemAvailable"}:
                    data[key] = float(parts[1])  # value in kB
                if len(data) == 2:
                    break
        if {"MemTotal", "MemAvailable"} <= data.keys():
            return data
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"[SYSTEM] Failed to read /proc/meminfo: {exc}")
    return None


def _collect_system_metrics():
    metrics = {
        'cpu_temperature': None,
        'cpu_percent': None,
        'ram_percent': None,
        'ram_available_mb': None,
        'timestamp': time.time(),
    }

    # CPU temperature
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as f:
            metrics['cpu_temperature'] = float(f.read()) / 1000.0
    except Exception:
        if sensor_data.get('cpu_temp') is not None:
            metrics['cpu_temperature'] = sensor_data.get('cpu_temp')

    if psutil:
        try:
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            metrics['ram_percent'] = vm.percent
            metrics['ram_available_mb'] = vm.available / (1024 * 1024)
        except Exception as exc:
            print(f"[SYSTEM] psutil metrics failed: {exc}")
    else:
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            metrics['cpu_percent'] = (load1 / cpu_count) * 100.0
        except (AttributeError, OSError):
            pass
        meminfo = _read_meminfo_snapshot()
        if meminfo:
            total = meminfo.get("MemTotal")
            available = meminfo.get("MemAvailable")
            if total and available:
                metrics['ram_percent'] = (1.0 - (available / total)) * 100.0
                metrics['ram_available_mb'] = available / 1024.0

    return metrics


def system_debug_loop():
    """Periodically print temperature and resource usage for the host."""
    if psutil:
        # Prime the internal counters so the first cpu_percent call is meaningful.
        psutil.cpu_percent(interval=None)
    while True:
        metrics = _collect_system_metrics()
        entries = []
        if metrics['cpu_temperature'] is not None:
            entries.append(f"CPU Temp {metrics['cpu_temperature']:.2f}C")
        if metrics['cpu_percent'] is not None:
            entries.append(f"CPU {metrics['cpu_percent']:.1f}%")
        if metrics['ram_percent'] is not None:
            if metrics['ram_available_mb'] is not None:
                entries.append(f"RAM {metrics['ram_percent']:.1f}% ({metrics['ram_available_mb']:.0f}MB free)")
            else:
                entries.append(f"RAM {metrics['ram_percent']:.1f}%")
        if entries:
            print("[SYSTEM] " + " | ".join(entries))

        with lock:
            if metrics['cpu_temperature'] is not None:
                sensor_data['cpu_temp'] = metrics['cpu_temperature']

        with system_metrics_lock:
            system_metrics.update(metrics)

        if any(metrics.get(key) is not None for key in ('cpu_percent', 'ram_percent', 'cpu_temperature')):
            try:
                with app.app_context():
                    db.session.add(SystemMetric(
                        cpu_percent=metrics.get('cpu_percent'),
                        ram_percent=metrics.get('ram_percent'),
                        ram_available_mb=metrics.get('ram_available_mb'),
                        cpu_temperature=metrics.get('cpu_temperature'),
                    ))
                    db.session.commit()
            except Exception as exc:
                print(f"[SYSTEM] Failed to persist metrics: {exc}")
        time.sleep(SYSTEM_DEBUG_INTERVAL)

@app.route('/set_conveyor_speed/<int:speed>', methods=['POST'])
def set_conveyor_speed(speed):
    if control_mode == "manual":
        print(f"Manual mode: Conveyor speed set to {speed}")
        time.sleep(0.3)  # 300 milliseconds delay before sending the message
        # Publish to /home/control/conveyorPWM, for now 0
        mqtt.publish('/home/control/conveyorPWM', '0')
        time.sleep(0.6)  # 300 milliseconds delay before sending second message, for safety, because one of esp is in deepleep
        mqtt.publish('/home/control/conveyorPWM', '0')
    return jsonify({"status": "ok", "conveyor_speed": speed})

@app.route('/set_control_mode/<mode>', methods=['POST'])
def set_control_mode(mode):
    global control_mode

    # Only allow "esp32" or "manual"
    if mode in ["esp32", "manual"]:
        control_mode = mode
        print(f"Control mode switched to: {mode}")

        # Publish the same raw string to the ESP32
        # so it matches your callback logic:
        mqtt.publish('/set_control_mode', mode)

        return jsonify({"status": "ok", "control_mode": control_mode})
    else:
        return jsonify({"error": "Invalid mode. Use 'esp32' or 'manual'."}), 400

# Thread to read RPi CPU Temp & store in DB
def read_rpi_temp_forever():
    while True:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_c = float(f.read()) / 1000.0
            with lock:
                sensor_data['cpu_temp'] = temp_c

            # Optionally store CPU temp in DB
            """with app.app_context():
                entry = SensorData(temperature=temp_c, humidity=None, sensor_source='cpu')
                db.session.add(entry)
                db.session.commit()
            """
        except Exception as e:
            print(f"Error reading RPi temp: {e}")

        time.sleep(5)

# MQTT Handlers
@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    global is_subscribed
    print("Connected to MQTT broker")
    if not is_subscribed:
        try:
            topics = ['/home/sensor_data', '/home/control/light', '/home/sensors/distance']
            for t in topics:
                mqtt.subscribe(t)
            is_subscribed = True
            print(f"Subscribed to topics: {topics}")
        except Exception as e:
            print(f"Error subscribing to topics: {e}")

@mqtt.on_disconnect()
def handle_disconnect():
    print("MQTT broker disconnected â€” resetting manual modes.")
    # Publish messages to inform ESP32 to return to sensor mode
    mqtt.publish("/home/control/light", "OFF")
    mqtt.publish('/set_control_mode', '0')

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    global sensor_data
    try:
        payload_text = message.payload.decode('utf-8').strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = payload_text
        print(f"MQTT received on {message.topic}: {payload}")

        if message.topic == '/home/sensor_data' and isinstance(payload, dict):
            t = float(payload.get('t', 0.0))
            h = float(payload.get('h', 0.0))
            sensor_data['temperature'] = t
            sensor_data['humidity'] = h
            # Store DHT sensor data with sensor_source ='dht'
            with app.app_context():
                entry = SensorData(temperature=t, humidity=h, sensor_source='dht')
                db.session.add(entry)
                db.session.commit()

        elif message.topic == '/home/sensors/distance' and isinstance(payload, dict):
            sensor_data['distance'] = float(payload.get('distance', 0.0))
        elif message.topic == '/home/control/light':
            print("Light Control message:", payload)
        elif message.topic == '/set_control_mode':
            print("Control mode message:", payload)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except Exception as e:
        print(f"Error processing MQTT message: {e}")


def _coerce_to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off'}
    return bool(value)

# Routes
#video routes in separate file video_meas.py

@app.route('/ai_recognition', methods=['GET', 'POST'])
def ai_recognition_route():
    if request.method == 'GET':
        return jsonify({'enabled': is_ai_recognition_enabled()})
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {}
    if 'enabled' in payload:
        target_state = _coerce_to_bool(payload.get('enabled'))
    elif 'toggle' in payload:
        target_state = not is_ai_recognition_enabled()
    else:
        target_state = not is_ai_recognition_enabled()
    new_state = set_ai_recognition_enabled(target_state)
    return jsonify({'enabled': new_state})

# Return the distance as JSON
@app.route('/distance', methods=['GET'])
def get_distance():
    dist = sensor_data.get('distance', 'No data')
    return jsonify({'distance': dist})

# Return the DHT sensor data as JSON
@app.route('/sensor_data', methods=['GET'])
def get_sensor_data():
    t = sensor_data.get('temperature', 'No data')
    h = sensor_data.get('humidity', 'No data')
    return jsonify({'temperature': t, 'humidity': h})


@app.route('/model_state', methods=['GET'])
def model_state_route():
    return jsonify(get_model_state_snapshot())


@app.route('/system_resources', methods=['GET'])
def system_resources():
    with system_metrics_lock:
        data = dict(system_metrics)
    data.setdefault('cpu_temperature', None)
    data.setdefault('cpu_percent', None)
    data.setdefault('ram_percent', None)
    data.setdefault('ram_available_mb', None)
    data.setdefault('timestamp', time.time())
    return jsonify(data)

# Return CPU temperature as JSON //(unchanged)
@app.route('/rpi_temperature', methods=['GET'])
def rpi_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0
        return jsonify({"temperature": f"{temp:.3f} \u00B0C"})
    except Exception as e:
        return jsonify({"error": "Could not fetch RPi temperature", "message": str(e)}), 500

@app.route('/control/light', methods=['POST'])
def control_light():
    """Publish ON/OFF commands for the light over MQTT."""
    try:
        state = None
        payload = request.get_json(silent=True)
        if payload and isinstance(payload, dict):
            state = payload.get('state') or payload.get('value')
        if not state:
            state = request.form.get('state') or request.args.get('state')
        if not state:
            return jsonify({"error": "Missing light state"}), 400
        normalized = state.strip().upper()
        if normalized not in {'ON', 'OFF'}:
            return jsonify({"error": "Invalid light state. Use 'ON' or 'OFF'."}), 400
        mqtt.publish('/home/control/light', normalized)
        print(f"Sent light command: {normalized} to /home/control/light")
        return jsonify({"status": "success", "light": normalized})
    except Exception as exc:
        print(f"Error publishing light command: {exc}")
        return jsonify({"error": "Internal Server Error"}), 500

# Chart for 'temperature' or 'humidity'

@app.route('/chart/<data_type>')
def chart(data_type):
    try:
        if data_type == 'temperature':
            data = SensorData.query.filter_by(sensor_source='dht').with_entities(
                SensorData.timestamp, SensorData.temperature).all()
        elif data_type == 'humidity':
            data = SensorData.query.with_entities(
                SensorData.timestamp, SensorData.humidity).all()
        else:
            return "Invalid data type", 400

        chart_data = {
            "timestamps": [
                record[0].strftime('%Y-%m-%d %H:%M:%S')
                for record in data if record[0] is not None
            ],
            "values": [record[1] for record in data if record[1] is not None]
        }

        print(f"Chart Data for {data_type}: {chart_data}")
        return render_template('chart.html', data_type=data_type, chart_data=chart_data)
    except Exception as e:
        print(f"Error generating chart: {e}")
        return f"Error generating chart: {e}", 500

# Combined Chart
@app.route('/chart/combined')
def combined_chart():
    """
    Returns a combined chart of DHT temperature and humidity,
    excluding CPU data (sensor_source='cpu').
    """
    try:
        # Only fetch rows where sensor_source='dht'
        data = (SensorData.query
                .filter(SensorData.sensor_source == 'dht')
                .with_entities(SensorData.timestamp,
                               SensorData.temperature,
                               SensorData.humidity)
                .all()) 
# Prepare data for the template
        chart_data = {
            "timestamps": [
                r[0].strftime('%Y-%m-%d %H:%M:%S') for r in data if r[0]
            ],
            "temperature_values": [r[1] for r in data if r[1] is not None],
            "humidity_values":    [r[2] for r in data if r[2] is not None]
        }

        print(f"Combined Chart Data: {chart_data}")
        return render_template('combined_chart.html', chart_data=chart_data)

    except Exception as e:
        print(f"Error generating combined chart: {e}")
        return f"Error generating combined chart: {e}", 500


@app.route('/chart/system')
def chart_system():
    try:
        try:
            limit = int(request.args.get('limit', '500'))
        except ValueError:
            limit = 500
        metrics = (SystemMetric.query
                   .order_by(SystemMetric.timestamp.desc())
                   .limit(limit)
                   .all())
        metrics = list(reversed(metrics))
        chart_data = {
            "timestamps": [m.timestamp.strftime('%Y-%m-%d %H:%M:%S') for m in metrics],
            "cpu_percent": [m.cpu_percent for m in metrics],
            "ram_percent": [m.ram_percent for m in metrics],
            "ram_available_mb": [m.ram_available_mb for m in metrics],
            "cpu_temperature": [m.cpu_temperature for m in metrics],
        }
        if request.args.get('format') == 'json':
            return jsonify(chart_data)
        return render_template('system_chart.html', chart_data=chart_data, sample_count=len(metrics), sample_limit=limit)
    except Exception as exc:
        print(f"Error generating system chart: {exc}")
        return f"Error generating system chart: {exc}", 500

@app.route('/chart/rpi_temp') # CHART for RPI Temp
def chart_rpi_temp():

    #Query sensor_source='cpu' from DB and pass data to a template for charting CPU temperature.
    
    try:
        data = SensorData.query.filter_by(sensor_source='cpu') \
                               .with_entities(SensorData.timestamp, SensorData.temperature).all()
        chart_data = {
            "timestamps": [record[0].strftime('%Y-%m-%d %H:%M:%S') for record in data if record[0]],
            "values":     [record[1] for record in data if record[1] is not None]
        }
        return render_template('chart_rpi_temp.html', chart_data=chart_data)
    except Exception as e:
        print(f"Error generating RPi temp chart: {e}")
        return f"Error generating chart: {e}", 500


# Home & index
@app.route('/')
def index():
    return render_template('index.html')

# Main
if __name__ == '__main__':
    # Start CPU temp reading in background
    threading.Thread(target=read_rpi_temp_forever, daemon=True).start()
    set_measurement_callback(on_new_measurement)
    set_ai_recognition_enabled(True)
    print(f"[MODEL] Initial stage: waiting for connectors using {CONNECTOR_MODEL_PATH.name}", flush=True)
    capture_thread = threading.Thread(target=capture_measured_video, daemon=True)
    capture_thread.start()
    threading.Thread(target=pwm_update_loop, daemon=True).start()
    threading.Thread(target=line_state_monitor, daemon=True).start()
    threading.Thread(target=system_debug_loop, daemon=True).start()
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
