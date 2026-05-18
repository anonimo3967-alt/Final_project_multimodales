import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import json

# LIBRERÍAS PARA EL CONTROL DE VOZ
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io

st.set_page_config(page_title="Control por Voz e IA - Comedero", layout="centered")
st.title("🐱 Sistema de Comedero por Voz e IA")
st.write("La IA mide la confianza en pantalla, pero tú decides cuándo abrir con tu voz.")

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN MQTT Y MODELO IA
# -------------------------------------------------------------------------
BROKER_IP = "157.230.214.127"
PORT = 1883
TOPIC_DIGITAL = "cmqtt_sdesi"
CLIENT_ID = "stream_client_michi_voice_99"

@st.cache_resource
def inicializar_recursos():
    try:
        modelo_keras = tf.keras.models.load_model('keras_model.h5', compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"Error al cargar 'keras_model.h5': {e}")
        
    cliente_mqtt = mqtt.Client(CLIENT_ID)
    try:
        cliente_mqtt.connect(BROKER_IP, PORT, 60)
        cliente_mqtt.loop_start()
    except Exception as e:
        st.error(f"No se pudo conectar al Broker MQTT ({BROKER_IP}): {e}")
        
    return modelo_keras, cliente_mqtt

model, client1 = inicializar_recursos()

# Ajusta el orden de estas etiquetas según tu modelo de Teachable Machine
ETIQUETAS = ["Gato Permitido", "Gato Intruso", "Nadie"]

# -------------------------------------------------------------------------
# 2. PROCESAMIENTO DE IMAGEN (TEACHABLE MACHINE)
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model == None: return "Nadie", 0.0
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction[0])
    return ETIQUETAS[index], prediction[0][index]

# -------------------------------------------------------------------------
# 3. PIPELINE DE LA CÁMARA (CON BARRA DE CONFIANZA INCLUIDA)
# -------------------------------------------------------------------------
@st.fragment
def pipeline_camara():
    imagen_feed = camera_input_live(width=420, height=315, debounce=800)
    if imagen_feed:
        st.image(imagen_feed, caption="Monitoreo del Comedero", use_container_width=True)
        img_pil = Image.open(imagen_feed).convert("RGB")
        resultado, confianza = procesar_y_clasificar(img_pil)
        
        # --- AQUÍ ESTÁ LA BARRA DE CONFIANZA QUE SE MUEVE ---
        st.write(f"Identificación actual de la IA: **{resultado}**")
        st.progress(float(confianza), text=f"Nivel de confianza: {confianza*100:.2f}%")
        
        # Alertas informativas en base a la detección
        if confianza > 0.75 and resultado != "Nadie":
            st.info(f"🚨 La IA detecta en la cámara a: **{resultado}**")
        elif resultado == "Nadie":
            st.success("✨ Zona del comedero despejada.")

pipeline_camara()

# -------------------------------------------------------------------------
# 4. MÓDULO: RECONOCIMIENTO DE COMANDOS DE VOZ
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎙️ Control por Comando de Voz")
st.write("Presiona el micrófono, di tu comando claramente en español y espera a que se procese.")

audio_grabado = mic_recorder(
    start_prompt="Presiona para Hablar 🎤",
    stop_prompt="Detener Grabación 🟥",
    just_once=True,
    format="wav",
    key="grabador_voz"
)

if audio_grabado:
    audio_bytes = audio_grabado['bytes']
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            texto_detectado = recognizer.recognize_google(audio_data, language="es-ES")
            
            st.write(f"Transcripción de voz: *\"{texto_detectado}\"*")
            comando_voz = texto_detectado.lower()
            
            # --- EVALUACIÓN DE COMANDOS DE VOZ ---
            payload = None
            
            if "abrir plato a" in comando_voz or "abrir plato 1" in comando_voz:
                payload = json.dumps({"Act1": "GATO_A"})
                st.success("Comando detectado: Abriendo Plato A")
                
            elif "abrir plato b" in comando_voz or "abrir plato 2" in comando_voz:
                payload = json.dumps({"Act1": "GATO_B"})
                st.success("Comando detectado: Abriendo Plato B")
                
            elif "cerrar" in comando_voz or "quitar comida" in comando_voz:
                payload = json.dumps({"Act1": "NADIE"})
                st.error("Comando detectado: Cerrando todos los platos")
                
            else:
                st.warning("Comando no reconocido. Prueba diciendo: 'abrir plato a', 'abrir plato b' o 'cerrar'.")
            
            if payload:
                client1.publish(TOPIC_DIGITAL, payload)
                st.toast(f"Enviado por voz a Wokwi: {payload}", icon="📡")
                
    except sr.UnknownValueError:
        st.error("No pude entender el audio. Asegúrate de hablar claro y cerca del micrófono.")
    except sr.RequestError as e:
        st.error(f"Error con el servicio de reconocimiento de voz: {e}")
