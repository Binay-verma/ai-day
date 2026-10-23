"""MQTT bridge: sensor ingestion (ESP32 nodes) and alert publishing.

Sensor nodes publish JSON to ``campus/<zone>/sensors``::

    {"smoke_ppm": 412.0, "temp_c": 31.2, "sound_db": 71.5}

The zone is taken from the topic (``+`` wildcard) unless the payload names
one.  Payloads are validated (numbers only, physical ranges, size limit) and
time-stamped on arrival with the server clock, so a node with a wrong RTC
cannot inject stale or future readings.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("edgecopilot.mqtt")

RANGES = {"smoke_ppm": (0.0, 10000.0), "temp_c": (-40.0, 125.0), "sound_db": (0.0, 160.0)}
MAX_PAYLOAD = 2048


def parse_sensor_payload(topic: str, payload: bytes, now: Optional[float] = None) -> Optional[dict]:
    """Validated reading dict or ``None`` (never raises on hostile input)."""
    if len(payload) > MAX_PAYLOAD:
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    parts = topic.split("/")
    zone = data.get("zone") if isinstance(data.get("zone"), str) else (parts[1] if len(parts) >= 3 else "default")
    zone = zone.replace("_", " ")[:60]
    out = {"zone": zone, "t": now if now is not None else time.time()}
    for k, (lo, hi) in RANGES.items():
        v = data.get(k)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            return None
        out[k] = float(v)
    return out if len(out) > 2 else None


class MqttBridge:
    def __init__(self, host: str, port: int, sensor_topic: str, on_reading: Callable[[dict], None],
                 username: Optional[str] = None, password: Optional[str] = None, tls: bool = False,
                 clock: Callable[[], float] = time.time):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edgecopilot-{os.getpid()}")
        if username:
            self.client.username_pw_set(username, password)
        if tls:
            self.client.tls_set()
        self.host, self.port, self.topic = host, port, sensor_topic
        self.on_reading = on_reading
        self.clock = clock
        self.connected = threading.Event()
        self.received = 0
        self.rejected = 0
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = lambda *a, **k: self.connected.clear()
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    @classmethod
    def from_config(cls, mcfg: dict, on_reading, clock=time.time) -> "MqttBridge":
        user = os.environ.get(mcfg["username_env"]) if mcfg.get("username_env") else None
        pw = os.environ.get(mcfg["password_env"]) if mcfg.get("password_env") else None
        return cls(mcfg.get("host", "localhost"), mcfg.get("port", 1883), mcfg.get("sensor_topic", "campus/+/sensors"),
                   on_reading, user, pw, mcfg.get("tls", False), clock)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.topic, qos=1)
        self.connected.set()
        log.info("MQTT connected to %s:%s, subscribed %s", self.host, self.port, self.topic)

    def _on_message(self, client, userdata, msg):
        r = parse_sensor_payload(msg.topic, msg.payload, self.clock())
        if r is None:
            self.rejected += 1
            return
        self.received += 1
        self.on_reading(r)

    def start(self) -> None:
        self.client.connect_async(self.host, self.port, keepalive=30)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload: str) -> None:
        self.client.publish(topic, payload, qos=1)
