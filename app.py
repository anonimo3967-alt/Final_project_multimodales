import streamlit as st
import paho.mqtt.client as mqtt
from PIL import Image, ImageOps
import numpy as np
import tflite_runtime.interpreter as tflite
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import json
import time

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
# 1. INICIALIZACIÓN DE RECURSOS GLOBALES (MQTT Y MODELO TFLITE)
# -------------------------------------------------------------------------
@st.cache_resource
def inicializar_recursos():
    # Carga optimizada y ligera usando TensorFlow Lite para evitar caídas de RAM
    try:
        # Asegúrate de tener el archivo "model.tflite" subido en la raíz de tu GitHub
        interprete = tflite.Interpreter(model_path="model.tflite")
        interprete.allocate_tensors()
    except Exception as e:
        interprete = None
        st.error(f"⚠️ Error al cargar el modelo 'model.tflite'. Verifica que esté en el repositorio. Detalle: {e}")
        
    cliente_mqtt = mqtt.Client(client_id=CLIENT_ID)
    
    try:
        cliente_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
        cliente_mqtt.loop_start()
    except Exception as e:
        st.error(f"⚠️ No se pudo conectar al servidor MQTT: {e}")
        
    return interprete, cliente_mqtt

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
        print(f"Error al publicar en MQTT: {e}")

# -------------------------------------------------------------------------
# 3. MÓDULO DE PROCESAMIENTO DE IMÁGENES (TFLITE)
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model is None:
        return "Nadie", 0.0
        
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    input_data = np.expand_dims(normalized_image_array, axis=0)
    
    # Inferencia ligera con arreglos de punteros nativos de TFLite
    detalles_entrada = model.get_input_details()
    detalles_salida = model.get_output_details()
    
    model.set_tensor(detalles_entrada[0]['index'], input_data)
    model.invoke()
    prediccion = model.get_tensor(detalles_salida[0]['index'])
    
    indice_maximo = np.argmax(prediccion[0])
    return ETIQUETAS[indice_maximo], prediccion[0][indice_maximo]

# -------------------------------------------------------------------------
# 4. INTERFAZ GRÁFICA PRINCIPAL (UI / UX CON AUTOMACIÓN LOOP)
# -------------------------------------------------------------------------
st.title("🐾 Panel del Comedero Inteligente")
st.write("Monitoreo en tiempo real asistido por IA Ligera y control de voz adaptativo.")

pestana_camara, pestana_voz = st.tabs(["📸 Visión Artificial", "🎙️ Control por Voz"])

# --- PESTAÑA A: CÁMARA CON BUCLE DE RERUN AUTOMÁTICO ---
with pestana_camara:
    st.header("Monitoreo del Comedero en Vivo")
    
    # Componente nativo del ecosistema de Streamlit
    imagen_feed = st.camera_input("Enfoque la cámara hacia el plato del comedero")
    
    if imagen_feed:
        img_pil = Image.open(imagen_feed).convert("RGB")
        resultado, confianza = procesar_y_clasificar(img_pil)
        
        st.write(f"Identificación actual de la IA: **{resultado}**")
        st.progress(float(confianza), text=f"Nivel de confianza: {confianza * 100:.2f}%")
        
        michi_detectado_ahora = resultado if confianza > 0.75 else "Nadie"
        
        # Filtro de estabilización matemática contra parpadeos de cuadros
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
            st.info(f"🚨 Servomotores accionados: Se abrió el compartimiento para **{st.session_state.ultimo_michi_visto}**.")
        else:
            st.success("✨ Zona despejada. Todos los platos permanecen resguardados.")
            
    # BUCLE SÍNCRONO PARA TIEMPO REAL CONTINUO
    # Espera 1 segundo y relanza la lectura de la cámara de manera autónoma
    time.sleep(1.0)
    st.rerun()

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
