"""Shared MQTT transport for face-centering motor commands and status."""
from __future__ import annotations

import json
import threading
from typing import Optional


class MQTTMotorClient:
    """Publish absolute motor angles without blocking the camera/UI loop.

    The matching ESP8266 subscriber expects JSON containing an ``angle`` field
    in degrees. Paho's background network loop handles reconnects and delivery.
    """

    def __init__(self, host: str, port: int, topic: str, status_topic: str, client_id: str,
                 keepalive: int = 30):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT support needs paho-mqtt; install it from requirements.txt") from exc

        self.topic = topic
        self.status_topic = status_topic
        self.connected = threading.Event()
        self.connection_count = 0
        self.last_status: Optional[str] = None
        self.last_error: Optional[str] = None
        self._mqtt = mqtt
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_connect_fail = self._on_connect_fail
        self.client.on_message = self._on_message
        self.client.connect_async(host, port, keepalive)
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            self.last_error = f"MQTT connect rejected: {reason_code}"
            self.connected.clear()
        else:
            self.last_error = None
            self.connection_count += 1
            self.connected.set()
            client.subscribe(self.status_topic, qos=0)

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            status = payload.get("status")
            angle = payload.get("angle")
            self.last_status = f"ESP8266 {status}" + (f" at {float(angle):.0f}°" if angle is not None else "")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            self.last_status = "ESP8266 sent an invalid status message"

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected.clear()
        if reason_code.is_failure:
            self.last_error = f"MQTT disconnected: {reason_code}"

    def _on_connect_fail(self, client, userdata):
        self.connected.clear()
        self.last_error = "MQTT broker unavailable; reconnecting"

    def publish_angle(self, angle: float, **metadata) -> bool:
        if not self.connected.is_set():
            return False
        payload = {"angle": round(float(angle), 1), **metadata}
        info = self.client.publish(self.topic, json.dumps(payload), qos=0, retain=False)
        return info.rc == self._mqtt.MQTT_ERR_SUCCESS

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()
