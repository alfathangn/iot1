# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
# from datetime import datetime
import plotly.graph_objs as go
import paho.mqtt.client as mqtt
from streamlit_autorefresh import st_autorefresh  # auto refresh helper
# import pytz
from datetime import datetime, timezone, timedelta

# -----------------------
# CONFIG (ubah bila perlu)
# -----------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "iot/class/session5/sensor"
TOPIC_OUTPUT = "iot/class/session5/output"
MODEL_PATH = "iot_temp_model.pkl"

# UI constants
MAX_POINTS = 200
ANOMALY_Z_THRESHOLD = 3.0  # default z-score threshold

# -----------------------
# Session state init
# -----------------------
if "mqtt_in_q" not in st.session_state:
    st.session_state.mqtt_in_q = queue.Queue()
if "mqtt_out_q" not in st.session_state:
    st.session_state.mqtt_out_q = queue.Queue()
if "logs" not in st.session_state:
    st.session_state.logs = []   # list of dict rows
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_worker_started" not in st.session_state:
    st.session_state.mqtt_worker_started = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "anomaly_window" not in st.session_state:
    st.session_state.anomaly_window = 30

# -----------------------
# Load model (safe)
# -----------------------
@st.cache_resource
def load_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model '{path}': {e}")
        return None

model = load_model(MODEL_PATH)
st.session_state.model_loaded = model is not None

# -----------------------
# MQTT worker (background thread)
# - reads st.session_state.mqtt_out_q for outgoing publishes
# - publishes incoming messages into st.session_state.mqtt_in_q
# -----------------------
def mqtt_worker(broker, port, topic_sensor, topic_output, in_q, out_q):
    client = mqtt.Client()
    # optional: set client.on_log = lambda c, userdata, level, buf: print(buf)
    def _on_connect(c, userdata, flags, rc):
        print("MQTT connected with rc:", rc)
        if rc == 0:
            c.subscribe(topic_sensor)
        else:
            print("MQTT connect failed rc:", rc)
    
    def _on_message(c, userdata, msg):
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
            # push raw message into queue for main thread to process
            in_q.put({
               
                "ts" : (datetime.utcnow() + timedelta(hours=7)).isoformat(),
                # "ts": datetime.utcnow().isoformat(),
               

                "topic": msg.topic,
                "payload": data
            })
        except Exception as e:
            print("Failed parse incoming msg:", e, getattr(msg, "payload", None))

    client.on_connect = _on_connect
    client.on_message = _on_message

    # try to connect with reconnect loop
    while True:
        try:
            client.connect(broker, port, keepalive=60)
            client.loop_start()
            print("Connected to MQTT broker:", broker, port)
            # run until error; meanwhile check outgoing queue periodically
            while True:
                try:
                    try:
                        item = out_q.get(timeout=0.5)
                    except queue.Empty:
                        item = None
                    if item is not None:
                        # expected item: {"topic": str, "payload": str}
                        topic = item.get("topic")
                        payload = item.get("payload")
                        qos = int(item.get("qos", 0))
                        retain = bool(item.get("retain", False))
                        client.publish(topic, payload, qos=qos, retain=retain)
                except Exception as e:
                    print("MQTT worker inner error:", e)
                    # keep running
            # client.loop_stop()  # unreachable
        except Exception as e:
            print("MQTT worker connection error:", e)
            try:
                client.loop_stop()
                client.disconnect()
            except:
                pass
            time.sleep(2)  # backoff then retry


# start mqtt worker once
if not st.session_state.mqtt_worker_started:
    t = threading.Thread(
        target=mqtt_worker,
        args=(MQTT_BROKER, MQTT_PORT, TOPIC_SENSOR, TOPIC_OUTPUT,
              st.session_state.mqtt_in_q, st.session_state.mqtt_out_q),
        daemon=True)
    t.start()
    st.session_state.mqtt_worker_started = True
    time.sleep(0.1)

# -----------------------
# Helper: process incoming queue items (main thread)
# -----------------------
def process_incoming():
    updated = False
    q = st.session_state.mqtt_in_q
    while not q.empty():
        item = q.get()
        payload = item["payload"]
        ts = item["ts"]
        # expected payload contains temp & hum
        temp = None
        hum = None
        try:
            temp = float(payload.get("temp"))
        except:
            temp = None
        try:
            hum = float(payload.get("hum"))
        except:
            hum = None

        row = {"ts": ts, "temp": temp, "hum": hum}

        # ML inference
        pred = None
        conf = None
        anomaly = False
        if model is not None and temp is not None and hum is not None:
            X = [[temp, hum]]
            try:
                pred = model.predict(X)[0]
            except Exception:
                pred = None
            # predict_proba if available
            try:
                conf = float(np.max(model.predict_proba(X)))
            except Exception:
                conf = None

            # anomaly detection:
            # 1) low confidence => suspicious
            if conf is not None and conf < 0.6:
                anomaly = True
            # 2) z-score on temp using recent window
            temps = [r["temp"] for r in st.session_state.logs if r.get("temp") is not None]
            window = temps[-st.session_state.anomaly_window:] if len(temps) > 0 else []
            if len(window) >= 5:
                mean = float(np.mean(window))
                std = float(np.std(window, ddof=0))
                if std > 0:
                    z = abs((temp - mean) / std)
                    if z >= ANOMALY_Z_THRESHOLD:
                        anomaly = True
            # attach info
        row.update({"pred": pred, "conf": conf, "anomaly": anomaly})
        st.session_state.last = row
        st.session_state.logs.append(row)
        # keep logs bounded
        if len(st.session_state.logs) > 5000:
            st.session_state.logs = st.session_state.logs[-5000:]
        updated = True
    return updated

# Poll incoming queue now
process_incoming()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="IoT ML Realtime Dashboard — ML Enhanced", layout="wide")
st.title("🔥 IoT ML Realtime Dashboard — ML Enhanced Version")

# show model status
if st.session_state.model_loaded:
    st.success(f"Model loaded: {MODEL_PATH}")
else:
    st.warning("Model not loaded. Place a valid pickle file (iot_temp_model.pkl) next to app.py")

# Auto-refresh small helper (useful to fetch new queue messages)
st_autorefresh(interval=2000, limit=None, key="autorefresh")

left, right = st.columns([1, 2])

with left:
    st.header("Connection Status")
    st.write("**MQTT Broker**")
    st.write(f"{MQTT_BROKER}:{MQTT_PORT}")
    # show simple connected status: check if any messages recently received
    connected = "Yes" if len(st.session_state.logs) > 0 else "No"
    st.metric("MQTT Connected", connected)
    st.write("Topic:", TOPIC_SENSOR)

    st.markdown("### Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"**Time (UTC):** {last['ts']}")
        st.write(f"**Temp:** {last['temp']} °C")
        st.write(f"**Hum :** {last['hum']} %")
        st.write(f"**Prediction:** {last.get('pred')}")
        st.write(f"**Confidence:** {last.get('conf')}")
        st.write(f"**Anomaly Flag:** {last.get('anomaly')}")
    else:
        st.info("Waiting for data...")

    st.markdown("### Manual Output Control")
    col1, col2 = st.columns(2)
    if col1.button("Send ALERT_ON"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_ON"})
        st.success("Published ALERT_ON")
    if col2.button("Send ALERT_OFF"):
        st.session_state.mqtt_out_q.put({"topic": TOPIC_OUTPUT, "payload": "ALERT_OFF"})
        st.success("Published ALERT_OFF")

    st.markdown("### Anomaly & Window Settings")
    w = st.slider("anomaly window (history points used for z-score)", 5, 200, st.session_state.anomaly_window)
    st.session_state.anomaly_window = w
    zthr = st.number_input("z-score threshold for anomaly", value=float(ANOMALY_Z_THRESHOLD))
    # store usage
    ANOMALY_Z_THRESHOLD = float(zthr)

    st.markdown("### Download Logs")
    if st.button("Download CSV"):
        if st.session_state.logs:
            df = pd.DataFrame(st.session_state.logs)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV file", csv, file_name=f"iot_logs_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")
        else:
            st.warning("No logs yet")

with right:
    st.header(f"Live Chart (last {MAX_POINTS} points)")
    df_plot = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])
    if not df_plot.empty and "temp" in df_plot.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["temp"], mode="lines+markers", name="Temp (°C)"))
        if "hum" in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Humidity (%)", yaxis="y2"))
            # add secondary y-axis for humidity
            fig.update_layout(
                yaxis=dict(title="Temp (°C)"),
                yaxis2=dict(title="Humidity (%)", overlaying="y", side="right", showgrid=False)
            )
        # color markers by prediction / anomaly
        colors = []
        for idx, r in df_plot.iterrows():
            if r.get("anomaly"):
                colors.append("red")
            else:
                if r.get("pred") == "Panas":
                    colors.append("orange")
                elif r.get("pred") == "Normal":
                    colors.append("green")
                elif r.get("pred") == "Dingin":
                    colors.append("blue")
                else:
                    colors.append("grey")
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="markers"))
        fig.update_layout(height=500, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")

    st.markdown("### Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(50))
    else:
        st.write("—")

# done UI: ensure we process any incoming messages that arrived during UI work
process_incoming()
