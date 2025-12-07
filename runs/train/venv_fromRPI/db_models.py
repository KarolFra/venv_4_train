from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    sensor_source = db.Column(db.String(50), default="dht")  # New column
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)


class SystemMetric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    cpu_percent = db.Column(db.Float, nullable=True)
    ram_percent = db.Column(db.Float, nullable=True)
    ram_available_mb = db.Column(db.Float, nullable=True)
    cpu_temperature = db.Column(db.Float, nullable=True)
