import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt

# =========================
# MQTT
# =========================

MQTT_BROKER = "157.230.214.127"
MQTT_TOPIC = "cmqtt_sdesi"

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

# =========================
# CARGAR MODELO
# =========================

model = load_model("keras_model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

# =========================
# STREAMLIT
# =========================

st.title("🐱 Detector Inteligente de Michis")

# =========================
# PROCESADOR DE VIDEO
# =========================

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # =========================
        # PREPROCESAMIENTO
        # =========================

        image = cv2.resize(img, (224, 224))

        image_array = np.asarray(image)

        normalized_image_array = (
            image_array.astype(np.float32) / 127.5
        ) - 1

        data = np.ndarray(
            shape=(1, 224, 224, 3),
            dtype=np.float32
        )

        data[0] = normalized_image_array

        # =========================
        # PREDICCIÓN
        # =========================

        prediction = model.predict(data, verbose=0)

        index = np.argmax(prediction)

        class_name = class_names[index]

        confidence_score = prediction[0][index]

        class_name = class_name[2:].strip()

        # =========================
        # TEXTO EN PANTALLA
        # =========================

        if class_name == "Michi_correcto":

            text = f"MICHI CORRECTO ({confidence_score:.2f})"

            color = (0, 255, 0)

            client.publish(
                MQTT_TOPIC,
                '{"Act1":"ON"}'
            )

        else:

            text = f"INTRUSO ({confidence_score:.2f})"

            color = (0, 0, 255)

            client.publish(
                MQTT_TOPIC,
                '{"Act1":"OFF"}'
            )

        # =========================
        # DIBUJAR TEXTO
        # =========================

        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# =========================
# INICIAR WEBCAM
# =========================

webrtc_streamer(
    key="michi-detector",
    video_processor_factory=VideoProcessor
)
