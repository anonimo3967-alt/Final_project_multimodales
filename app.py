import streamlit as st
import paho.mqtt.client as mqtt
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import io
import json

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
# 1. INICIALIZACIÓN DE RECURSOS GLOBALES (MQTT Y MODELO TENSORFLOW / KERAS)
# -------------------------------------------------------------------------
@st.cache_resource
def inicializar_recursos():
    try:
        from tensorflow.keras.models import load_model
        # Carga del modelo omitiendo compilación estricta de Keras 3
        modelo_keras = load_model("keras_model.h5", compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"⚠️ Error al cargar el modelo 'keras_model.h5'. Detalle: {e}")
        
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
# 3. INTERFAZ GRÁFICA PRINCIPAL (UI / UX)
# -------------------------------------------------------------------------
st.title("🐾 Panel del Comedero Inteligente")
st.write("Monitoreo automático asistido por TensorFlow / Keras y control de voz adaptativo.")

pestana_camara, pestana_voz = st.tabs(["📸 Visión Artificial Auto", "🎙️ Control por Voz"])

# --- PESTAÑA A: CÁMARA AUTOMÁTICA EN TIEMPO REAL (NATIVA CON WEBRTC) ---
with pestana_camara:
    st.header("Video del Comedero en Tiempo Real")
    
    contenedor_metricas = st.empty()
    contenedor_alertas = st.empty()

    # Mostramos el estado actual del sistema en la parte superior
    with contenedor_metricas.container():
        st.metric(label="🐾 Última Identificación de la IA:", value=st.session_state.ultimo_michi_visto)
        if st.session_state.ultimo_michi_visto != "Nadie":
            st.info(f"🚨 Servomotores ordenados para abrir el plato de **{st.session_state.ultimo_michi_visto}**.")
        else:
            st.success("✨ Zona despejada. Todos los platos permanecen resguardados.")

    # Clase encargada de procesar los frames de video directamente desde la WebRTC nativa
    class AnalizadorMichis(VideoTransformerBase):
        def transform(self, frame):
            # Convertimos el frame nativo a una imagen de Pillow
            img = frame.to_ndarray(format="bgr24")
            img_rgb = Image.fromarray(img)
            
            if model is not None:
                try:
                    # Ajuste dimensional exacto para Teachable Machine (224x224 RGB)
                    size = (224, 224)
                    image = ImageOps.fit(img_rgb, size, Image.Resampling.LANCZOS)
                    image_array = np.asarray(image)
                    
                    # Normalización idéntica al entrenamiento
                    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
                    input_data = np.expand_dims(normalized_image_array, axis=0)
                    
                    # Inferencia
                    prediccion = model.predict(input_data, verbose=0)
                    indice_maximo = np.argmax(prediccion[0])
                    resultado = ETIQUETAS[indice_maximo]
                    confianza = float(prediccion[0][indice_maximo])
                    
                    # Lógica de estabilidad para el envío MQTT
                    michi_detectado_ahora = resultado if confianza > 0.75 else "Nadie"
                    
                    if michi_detectado_ahora == st.session_state.michi_candidato:
                        st.session_state.contador_estabilidad += 1
                    else:
                        st.session_state.michi_candidato = michi_detectado_ahora
                        st.session_state.contador_estabilidad = 0
                        
                    if st.session_state.contador_estabilidad >= 5: # 5 cuadros estables
                        if michi_detectado_ahora != st.session_state.ultimo_michi_visto:
                            st.session_state.ultimo_michi_visto = michi_detectado_ahora
                            enviar_estado_sistema()
                except Exception as e:
                    print(f"Error en inferencia: {e}")
            
            # Devolvemos el frame modificado o igual para que se pinte en pantalla (añadimos efecto espejo)
            return np.flip(img, axis=1)

    # Invocamos el streamer oficial WebRTC de Streamlit
    webrtc_streamer(
        key="michi-streamer", 
        video_transformer_factory=AnalizadorMichis,
        media_stream_constraints={"video": True, "audio": False}
    )

# --- PESTAÑA B: INTERFAZ DE CONTROL POR VOZ ---
with pestana_voz:
    st.header("Comandos de Voz del Sistema")
    st.write("Presiona el botón para grabar un comando de voz directo hacia los servomotores (Ej: *'abrir plato'*, *'cerrar comedero'*).")
    
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
                datos_audio = reconnaissace = reconocedor.record(origen)
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
                    st.error("⚠️ Comando de voz no reconocido. Intenta incluir palabras como 'abrir' o 'cerrar'.")
                    
        except sr.UnknownValueError:
            st.error("❌ No logramos entender el audio. Asegúrate de hablar claro y cerca del micrófono.")
        except sr.RequestError as error_api:
            st.error(f"❌ Error técnico en el servicio de voz: {error_api}")
