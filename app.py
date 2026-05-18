import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import json

# LIBRERÍAS REQUERIDAS PARA EL CONTROL DE VOZ
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io

# Configuración estética de la página web
st.set_page_config(page_title="Comedero Inteligente Coco y Canela", layout="centered")
st.title("🐱 Comedero Inteligente de Coco y Canela")
st.write("La IA identifica pasivamente al gato en pantalla, pero tú decides cuándo abrir con tu voz.")

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN MQTT Y CARGA DEL MODELO DE IA (CACHEADO)
# -------------------------------------------------------------------------
BROKER_IP = "157.230.214.127"
PORT = 1883
TOPIC_DIGITAL = "cmqtt_sdesi"
CLIENT_ID = "stream_client_michi_voice_99"

@st.cache_resource
def inicializar_recursos():
    # Intenta cargar el modelo de Teachable Machine
    try:
        modelo_keras = tf.keras.models.load_model('keras_model.h5', compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"Error crítico: No se pudo cargar 'keras_model.h5'. Asegúrate de subirlo a tu GitHub: {e}")
        
    # Inicializa el cliente MQTT y arranca el loop asíncrono
    cliente_mqtt = mqtt.Client(CLIENT_ID)
    try:
        cliente_mqtt.connect(BROKER_IP, PORT, 60)
        cliente_mqtt.loop_start()
    except Exception as e:
        st.error(f"No se pudo establecer conexión con el Broker MQTT ({BROKER_IP}): {e}")
        
    return modelo_keras, cliente_mqtt

model, client1 = inicializar_recursos()

# Orden de las clases según entrenaste tu modelo (Clase 0, Clase 1, Clase 2)
ETIQUETAS = ["Coco", "Canela", "Nadie"]

# Variables de estado en memoria para el filtro antispam de la cámara
if "ultimo_michi_visto" not in st.session_state:
    st.session_state.ultimo_michi_visto = None

# -------------------------------------------------------------------------
# 2. PROCESAMIENTO MATEMÁTICO DE LA IMAGEN
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model is None: 
        return "Nadie", 0.0
        
    # Escalar la imagen al tamaño estricto de entrada (224, 224)
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalización exacta de Teachable Machine (Rango de -1.0 a 1.0)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    
    # Empaquetar en la dimensión de lote (Batch)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Realizar la predicción en la red neuronal
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction[0])
    return ETIQUETAS[index], prediction[0][index]

# -------------------------------------------------------------------------
# 3. PIPELINE DE LA CÁMARA (MONITOREO PASIVO Y ACTUALIZACIÓN DE LCD)
# -------------------------------------------------------------------------
@st.fragment
def pipeline_camara():
    # Capturador optimizado para la nube (un frame cada 800 milisegundos)
    imagen_feed = camera_input_live(width=420, height=315, debounce=800)
    
    if imagen_feed:
        # Dibujar feed de la webcam en la página web
        st.image(imagen_feed, caption="Monitoreo del Comedero en Vivo", use_container_width=True)
        
        # Convertir a PIL y procesar con la IA
        img_pil = Image.open(imagen_feed).convert("RGB")
        resultado, confianza = procesar_y_clasificar(img_pil)
        
        # Muestra dinámicamente la barra de progreso con el nivel de confianza
        st.write(f"Identificación actual de la IA: **{resultado}**")
        st.progress(float(confianza), text=f"Nivel de confianza del modelo: {confianza*100:.2f}%")
        
        # Filtrar por confianza mínima del 75%
        michi_actual = resultado if confianza > 0.75 else "Nadie"
        
        # Control antispam: Solo enviamos datos por MQTT si el gato cambió frente a la cámara
        if michi_actual != st.session_state.ultimo_michi_visto:
            st.session_state.ultimo_michi_visto = michi_actual
            
            # JSON exclusivo para cambiar el texto de la pantalla LCD sin tocar servos
            payload_pantalla = json.dumps({"Pantalla": michi_actual})
            try:
                client1.publish(TOPIC_DIGITAL, payload_pantalla)
            except Exception as e:
                st.error(f"Error al enviar datos de pantalla a MQTT: {e}")
        
        # Cuadros de alerta visual en Streamlit
        if confianza > 0.75 and resultado != "Nadie":
            st.info(f"🚨 La IA detecta en la cámara a: **{resultado}**")
        elif resultado == "Nadie":
            st.success("✨ Zona del comedero despejada.")

pipeline_camara()

# -------------------------------------------------------------------------
# 4. RECONOCIMIENTO DE COMANDOS DE VOZ (CONTROL ACTIVO DE SERVOS)
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎙️ Control por Comando de Voz")
st.write("Di comandos claros en español como: *'abrir coco'*, *'abre el plato de canela'* o *'cerrar comederos'*.")

# Capturador de audio configurado estrictamente en formato WAV para evitar errores binarios
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
        # Analizar el archivo virtual de sonido
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            texto_detectado = recognizer.recognize_google(audio_data, language="es-ES")
            
            st.write(f"Transcripción de voz obtenida: *\"{texto_detectado}\"*")
            comando_voz = texto_detectado.lower()
            
            payload_motores = None
            
            # Evaluación e interpretación de palabras clave
            if "coco" in comando_voz:
                payload_motores = json.dumps({"Act1": "GATO_A"})
                st.success("Comando aceptado: Abriendo el plato de Coco 🐱")
                
            elif "canela" in comando_voz:
                payload_motores = json.dumps({"Act1": "GATO_B"})
                st.success("Comando aceptado: Abriendo el plato de Canela 🐱")
                
            elif "cerrar" in comando_voz or "quitar" in comando_voz or "nadie" in comando_voz:
                payload_motores = json.dumps({"Act1": "NADIE"})
                st.error("Comando aceptado: Cerrando todos los comederos")
                
            else:
                st.warning("Comando no reconocido. Recuerda pronunciar claramente 'Coco', 'Canela' o 'Cerrar'.")
            
            # Si la orden de voz fue válida, se publica al tópico digital del ESP32
            if payload_motores:
                client1.publish(TOPIC_DIGITAL, payload_motores)
                st.toast(f"Mensaje de acción enviado a Wokwi: {payload_motores}", icon="📡")
                
    except sr.UnknownValueError:
        st.error("El motor de voz no pudo entender el audio. Intenta hablar con mayor claridad.")
    except sr.RequestError as e:
        st.error(f"Error de comunicación con el servicio de voz de Google: {e}")
