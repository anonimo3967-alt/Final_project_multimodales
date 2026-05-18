import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import json
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io

st.set_page_config(page_title="Comedero Inteligente Coco y Canela", layout="centered")
st.title("🐱 Comedero Inteligente de Coco y Canela")
st.write("Sistema IoT Bidireccional en Bucle Cerrado.")

# --- VARIABLES DE ESTADO UNIFICADAS ---
if "ultimo_michi_visto" not in st.session_state:
    st.session_state.ultimo_michi_visto = "Nadie"
if "contador_estabilidad" not in st.session_state:
    st.session_state.contador_estabilidad = 0
if "michi_candidato" not in st.session_state:
    st.session_state.michi_candidato = "Nadie"
if "estado_motor_actual" not in st.session_state:
    st.session_state.estado_motor_actual = "NADIE"
if "confirmacion_hardware" not in st.session_state:
    st.session_state.confirmacion_hardware = "Esperando conexión..."

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN MQTT Y CARGA DEL MODELO DE IA (REPARADO)
# -------------------------------------------------------------------------
BROKER_IP = "157.230.214.127"
PORT = 1883
TOPIC_CONTROL = "cmqtt_sdesi"        # Lo que Streamlit envía
TOPIC_STATUS = "cmqtt_sdesi_status"   # Lo que Streamlit recibe de Wokwi
CLIENT_ID = "stream_client_michi_voice_99"

# Función que se ejecuta automáticamente cuando Wokwi publica algo
def al_recibir_mensaje(client, userdata, message):
    texto_recibido = str(message.payload.decode("utf-8"))
    st.session_state.confirmacion_hardware = texto_recibido

@st.cache_resource
def inicializar_recursos():
    try:
        modelo_keras = tf.keras.models.load_model('keras_model.h5', compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"Error crítico al cargar modelo: {e}")
        
    # CORRECCIÓN AQUÍ: Pasamos el CLIENT_ID de forma correcta sin usar parámetros inválidos
    cliente_mqtt = mqtt.Client(client_id=CLIENT_ID)
    
    # Asignamos la función de escucha
    cliente_mqtt.on_message = al_recibir_mensaje
    
    try:
        cliente_mqtt.connect(BROKER_IP, PORT, 60)
        cliente_mqtt.subscribe(TOPIC_STATUS) # Nos suscribimos al reporte de Wokwi
        cliente_mqtt.loop_start()
    except Exception as e:
        st.error(f"No se pudo conectar al Broker MQTT: {e}")
        
    return modelo_keras, cliente_mqtt

model, client1 = inicializar_recursos()
ETIQUETAS = ["Coco", "Canela", "Nadie"]

def enviar_estado_sistema():
    payload = json.dumps({
        "Pantalla": st.session_state.ultimo_michi_visto,
        "Act1": st.session_state.estado_motor_actual
    })
    try:
        client1.publish(TOPIC_CONTROL, payload, qos=1)
    except Exception as e:
        st.error(f"Error al enviar datos: {e}")

# -------------------------------------------------------------------------
# INDICADOR VISUAL DE TELEMETRÍA REAL (PANEL DE FEEDBACK)
# -------------------------------------------------------------------------
st.sidebar.markdown("### 🛰️ Telemetría del Circuito (Wokwi)")

estado_real = st.session_state.confirmacion_hardware

if estado_real == "COCO_ABIERTO":
    st.sidebar.success("🔓 Confirmado: Compuerta de COCO Abierta")
elif estado_real == "CANELA_ABIERTO":
    st.sidebar.success("🔓 Confirmado: Compuerta de CANELA Abierta")
elif estado_real == "TODO_CERRADO":
    st.sidebar.error("🔒 Confirmado: Todo Cerrado Seguro")
else:
    st.sidebar.warning(f"⏳ Estado: {estado_real}")

if st.sidebar.button("🔄 Refrescar Telemetría"):
    st.rerun()

# -------------------------------------------------------------------------
# 2. PROCESAMIENTO MATEMÁTICO DE LA IMAGEN
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model is None: return "Nadie", 0.0
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
# 3. PIPELINE DE LA CÁMARA
# -------------------------------------------------------------------------
@st.fragment
def pipeline_camara():
    imagen_feed = camera_input_live(width=420, height=315, debounce=900)
    if imagen_feed:
        st.image(imagen_feed, caption="Monitoreo en Vivo", use_container_width=True)
        img_pil = Image.open(imagen_feed).convert("RGB")
        resultado, confianza = procesar_y_clasificar(img_pil)
        
        st.write(f"Identificación actual de la IA: **{resultado}**")
        st.progress(float(confianza), text=f"Confianza: {confianza*100:.2f}%")
        
        michi_detectado_ahora = resultado if confianza > 0.75 else "Nadie"
        
        if michi_detectado_ahora == st.session_state.michi_candidato:
            st.session_state.contador_estabilidad += 1
        else:
            st.session_state.michi_candidato = michi_detectado_ahora
            st.session_state.contador_estabilidad = 0
            
        if st.session_state.contador_estabilidad >= 2:
            if michi_detectado_ahora != st.session_state.ultimo_michi_visto:
                st.session_state.ultimo_michi_visto = michi_detectado_ahora
                enviar_estado_sistema()
        
        if st.session_state.ultimo_michi_visto != "Nadie":
            st.info(f"🚨 La IA detecta en la cámara a: **{st.session_state.ultimo_michi_visto}**")
        else:
            st.success("✨ Zona del comedero despejada.")

pipeline_camara()

# -------------------------------------------------------------------------
# 4. RECONOCIMIENTO DE COMANDOS DE VOZ
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎙️ Control por Comando de Voz")
st.write("Di comandos como: *'abrir coco'*, *'abre el plato de canela'* o *'cerrar comederos'*.")

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
            
            st.write(f"Transcripción: *\"{texto_detectado}\"*")
            comando_voz = texto_detectado.lower()
            comando_valido = False
            
            if "coco" in comando_voz:
                st.session_state.estado_motor_actual = "GATO_A"
                st.success("Comando aceptado: Abriendo el plato de Coco 🐱")
                comando_valido = True
            elif "canela" in comando_voz:
                st.session_state.estado_motor_actual = "GATO_B"
                st.success("Comando aceptado: Abriendo el plato de Canela 🐱")
                comando_valido = True
            elif "cerrar" in comando_voz or "quitar" in comando_voz or "nadie" in comando_voz:
                st.session_state.estado_motor_actual = "NADIE"
                st.error("Comando aceptado: Cerrando todos los comederos")
                comando_valido = True
            else:
                st.warning("Comando no reconocido.")
            
            if comando_valido:
                enviar_estado_sistema()
                st.toast("¡Comando enviado!", icon="📡")
                
    except sr.UnknownValueError:
        st.error("El motor de voz no pudo entender el audio.")
    except sr.RequestError as e:
        st.error(f"Error con el servicio de voz: {e}")
