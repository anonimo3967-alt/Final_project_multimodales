import streamlit as st
import paho.mqtt.client as mqtt
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf  # Cambiado de tflite a TensorFlow normal
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import json
import base64

# -------------------------------------------------------------------------
# CONTEXTO DEL PROYECTO: COMEDERO AUTOMATIZADO (COCO Y CANELA)
# -------------------------------------------------------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_TXT_ESP = "universidad/multimodal/esp32/control"  
TOPIC_JSON_ST = "universidad/multimodal/streamlit/estado" 
CLIENT_ID = "streamlit_michi_feeder_client"

ETIQUETAS = ["Coco", "Canela", "Nadie"]

st.set_page_config(page_title="Comedero Inteligente - Coco & Canela", page_icon="🐾", layout="centered")

# -------------------------------------------------------------------------
# 1. INICIALIZACIÓN DE RECURSOS GLOBALES (MQTT Y MODELO KERAS/H5)
# -------------------------------------------------------------------------
@st.cache_resource
def inicializar_recursos():
    try:
        # Intenta cargar tu modelo de Keras. Asegúrate de cambiar "model.keras" 
        # por el nombre exacto de tu archivo (por ejemplo, "keras_model.h5" o "model.h5")
        modelo_keras = tf.keras.models.load_model("keras_model.h5", compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"⚠️ Error al cargar el modelo de Keras. Verifica el nombre del archivo en tu GitHub. Detalle: {e}")
        
    cliente_mqtt = mqtt.Client(client_id=CLIENT_ID)
    try:
        cliente_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
        cliente_mqtt.loop_start()
    except Exception as e:
        st.error(f"⚠️ Error MQTT: {e}")
        
    return modelo_keras, cliente_mqtt

model, client = inicializar_recursos()

# -------------------------------------------------------------------------
# 2. GESTIÓN DEL ESTADO DE LA SESIÓN (SESSION STATE)
# -------------------------------------------------------------------------
if "ultimo_michi_visto" not in st.session_state:
    st.session_state.ultimo_michi_visto = "Nadie"
if "michi_candidato" not in st.session_state:
    st.session_state.michi_candidato = "Nadie"
if "contador_estabilidad" not in st.session_state:
    st.session_state.contador_estabilidad = 0

if "hub_comunicacion" not in st.session_state:
    st.session_state.hub_comunicacion = ""

def enviar_estado_sistema():
    payload = {
        "michi_detectado": st.session_state.ultimo_michi_visto,
        "accion_sugerida": "ABRIR" if st.session_state.ultimo_michi_visto != "Nadie" else "CERRAR"
    }
    try:
        client.publish(TOPIC_JSON_ST, json.dumps(payload), qos=1)
    except Exception as e:
        print(f"Error MQTT: {e}")

# -------------------------------------------------------------------------
# 3. MÓDULO DE PROCESAMIENTO DE IMÁGENES (TERNSORFLOW K_ERAS)
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model is None:
        return "Nadie", 0.0
        
    # Redimensionar la imagen al tamaño estándar de entrada (Teachable Machine usa 224x224)
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalización idéntica a la que genera Teachable Machine / Keras
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    input_data = np.expand_dims(normalized_image_array, axis=0)
    
    # Inferencia directa con el modelo clásico cargado
    prediccion = model.predict(input_data, verbose=0)
    
    indice_maximo = np.argmax(prediccion[0])
    return ETIQUETAS[indice_maximo], float(prediccion[0][indice_maximo])

# -------------------------------------------------------------------------
# 4. INTERFAZ GRÁFICA PRINCIPAL (UI / UX)
# -------------------------------------------------------------------------
st.title("🐾 Panel del Comedero Inteligente")
st.write("Monitoreo automático asistido por TensorFlow / Keras y control de voz.")

pestana_camara, pestana_voz = st.tabs(["📸 Visión Artificial Auto", "🎙️ Control por Voz"])

# --- PESTAÑA A: CÁMARA AUTOMÁTICA EN TIEMPO REAL (JS STREAMING) ---
with pestana_camara:
    st.header("Video del Comedero en Tiempo Real")
    
    # Marcador de posición dinámico para colocar las métricas arriba del video
    contenedor_video = st.empty()
    
    # INYECCIÓN DE COMPONENTE WEB JAVASCRIPT
    js_camera_code = """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <video id="webcam" autoplay playsinline width="400" height="300" style="border-radius: 10px; background-color: #222;"></video>
        <canvas id="canvas_oculto" width="224" height="224" style="display:none;"></canvas>
        <p style="color: #888; font-size: 13px; margin-top: 5px;">Transmisión activa con TensorFlow 🟢</p>
    </div>
    
    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas_oculto');
        const ctx = canvas.getContext('2d');
        
        navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 300 } })
            .then((stream) => {
                video.srcObject = stream;
            })
            .catch((err) => {
                console.error("Error al acceder a la webcam: ", err);
            });
            
        setInterval(() => {
            if(video.videoWidth > 0) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/jpeg');
                
                const streamLitInput = window.parent.document.querySelector('input[aria-label="transfer_frame_hub"]');
                if (streamLitInput) {
                    streamLitInput.value = dataURL;
                    streamLitInput.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        }, 900);
    </script>
    """
    
    st.components.v1.html(js_camera_code, height=350)
    
    # Input invisible puente
    captura_base64 = st.text_input("transfer_frame_hub", label_visibility="collapsed", key="hub_comunicacion")
    
    if captura_base64 and captura_base64.startswith("data:image/jpeg;base64,"):
        try:
            datos_limpios = captura_base64.replace("data:image/jpeg;base64,", "")
            bytes_imagen = base64.b64decode(datos_limpios)
            img_pil = Image.open(io.BytesIO(bytes_imagen)).convert("RGB")
            
            # Ejecutar inferencia con Keras
            resultado, confianza = procesar_y_clasificar(img_pil)
            
            with contenedor_video.container():
                st.metric(label="IA Identificó a:", value=resultado, delta=f"Confianza: {confianza * 100:.1f}%")
                st.progress(confianza)
            
            # Filtro de estabilidad
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
                st.info(f"🚨 Servomotores en Wokwi ordenados para abrir el plato de **{st.session_state.ultimo_michi_visto}**.")
            else:
                st.success("✨ Zona despejada. Todos los platos permanecen resguardados.")
                
        except Exception as e:
            pass

# --- PESTAÑA B: INTERFAZ DE CONTROL POR VOZ ---
with pestana_voz:
    st.header("Comandos de Voz del Sistema")
    st.write("Presiona el botón para grabar un comando de voz (Ej: *'abrir plato'*, *'cerrar comedero'*).")
    
    audio_grabado = mic_recorder(
        start_prompt="🎙️ Iniciar grabación",
        stop_prompt="🛑 Detener y procesar",
        key="grabadora_michi",
        format="wav"
    )
    
    if audio_grabado:
        bytes_audio = audio_grabado['bytes']
        st.audio(bytes_audio, format="audio/wav")
        
        reconocedor = sr.Recognizer()
        archivo_audio = io.BytesIO(bytes_audio)
        
        try:
            with sr.AudioFile(archivo_audio) as origen:
                datos_audio = reconocedor.record(origen)
                texto_comando = reconocedor.recognize_google(datos_audio, language="es-ES").lower()
                
                st.subheader("Texto interpretado:")
                st.code(texto_comando)
                
                if "abrir" in texto_comando or "abre" in texto_comando:
                    client.publish(TOPIC_TXT_ESP, "ABRIR", qos=1)
                    st.success("🛰️ Comando enviado por MQTT: **ABRIR**.")
                elif "cerrar" in texto_comando or "cierra" in texto_comando:
                    client.publish(TOPIC_TXT_ESP, "CERRAR", qos=1)
                    st.warning("🛰️ Comando enviado por MQTT: **CERRAR**.")
                else:
                    st.error("⚠️ Comando de voz no reconocido.")
                    
        except sr.UnknownValueError:
            st.error("❌ No logramos entender el audio.")
        except sr.RequestError as error_api:
            st.error(f"❌ Error en el servicio de voz: {error_api}")
