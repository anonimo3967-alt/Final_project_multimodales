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
    try:
        modelo_keras = tf.keras.models.load_model('keras_model.h5', compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"Error crítico: No se pudo cargar 'keras_model.h5': {e}")
        
    cliente_mqtt = mqtt.Client(CLIENT_ID)
    try:
        cliente_mqtt.connect(BROKER_IP, PORT, 60)
        cliente_mqtt.loop_start()
    except Exception as e:
        st.error(f"No se pudo conectar al Broker MQTT ({BROKER_IP}): {e}")
        
    return modelo_keras, cliente_mqtt

model, client1 = inicializar_recursos()

# Orden de las clases según Teachable Machine
ETIQUETAS = ["Coco", "Canela", "Nadie"]

# --- VARIABLES DE SESIÓN (INICIALIZACIÓN) ---
if "ultimo_michi_visto" not in st.session_state:
    st.session_state.ultimo_michi_visto = "Nadie"
if "contador_estabilidad" not in st.session_state:
    st.session_state.contador_estabilidad = 0
if "michi_candidato" not in st.session_state:
    st.session_state.michi_candidato = "Nadie"

# -------------------------------------------------------------------------
# 2. PROCESAMIENTO MATEMÁTICO DE LA IMAGEN
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model is None: 
        return "Nadie", 0.0
        
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalización exacta de Teachable Machine (-1.0 a 1.0)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction[0])
    return ETIQUETAS[index], prediction[0][index]

# -------------------------------------------------------------------------
# 3. PIPELINE DE LA CÁMARA (MONITOREO CON FILTRO DE ESTABILIDAD)
# -------------------------------------------------------------------------
@st.fragment
def pipeline_camara():
    # Debounce ajustado a 900ms para darle un respiro óptimo a Wokwi
    imagen_feed = camera_input_live(width=420, height=315, debounce=900)
    
    if imagen_feed:
        st.image(imagen_feed, caption="Monitoreo del Comedero en Vivo", use_container_width=True)
        
        img_pil = Image.open(imagen_feed).convert("RGB")
        resultado, confianza = procesar_y_clasificar(img_pil)
        
        # Muestra dinámicamente la barra de progreso
        st.write(f"Identificación actual de la IA: **{resultado}**")
        st.progress(float(confianza), text=f"Nivel de confianza del modelo: {confianza*100:.2f}%")
        
        # Filtro inicial por confianza (mínimo 75%)
        michi_detectado_ahora = resultado if confianza > 0.75 else "Nadie"
        
        # LÓGICA DEL FILTRO DE ESTABILIDAD
        if michi_detectado_ahora == st.session_state.michi_candidato:
            st.session_state.contador_estabilidad += 1
        else:
            st.session_state.michi_candidato = michi_detectado_ahora
            st.session_state.contador_estabilidad = 0
            
        # El michi debe mantenerse por lo menos 2 frames seguidos para enviar el MQTT
        if st.session_state.contador_estabilidad >= 2:
            if michi_detectado_ahora != st.session_state.ultimo_michi_visto:
                st.session_state.ultimo_michi_visto = michi_detectado_ahora
                
                # Payload exclusivo para cambiar el texto de la LCD
                payload_pantalla = json.dumps({"Pantalla": michi_detectado_ahora})
                try:
                    client1.publish(TOPIC_DIGITAL, payload_pantalla, qos=1)
                except Exception as e:
                    st.error(f"Error al enviar datos de pantalla: {e}")
        
        # Bloque de alertas corregido y limpio en base al estado de la sesión
        if st.session_state.ultimo_michi_visto != "Nadie":
            st.info(f"🚨 La IA detecta en la cámara a: **{st.session_state.ultimo_michi_visto}**")
        else:
            st.success("✨ Zona del comedero despejada.")

pipeline_camara()

# -------------------------------------------------------------------------
# 4. RECONOCIMIENTO DE COMANDOS DE VOZ (CONTROL ACTIVO DE SERVOS)
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎙️ Control por Comando de Voz")
st.write("Di comandos claros en español como: *'abrir coco'*, *'abre el plato de canela'* o *'cerrar comederos'*.")

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
            
            st.write(f"Transcripción de voz obtenida: *\"{texto_detectado}\"*")
            comando_voz = texto_detectado.lower()
            
            payload_motores = None
            
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
                st.warning("Comando no reconocido. Intenta diciendo claramente 'Coco' o 'Canela'.")
            
            if payload_motores:
                client1.publish(TOPIC_DIGITAL, payload_motores)
                st.toast(f"Mensaje de acción enviado a Wokwi: {payload_motores}", icon="📡")
                
    except sr.UnknownValueError:
        st.error("El motor de voz no pudo entender el audio. Intenta hablar con mayor claridad.")
    except sr.RequestError as e:
        st.error(f"Error con el servicio de voz de Google: {e}")
