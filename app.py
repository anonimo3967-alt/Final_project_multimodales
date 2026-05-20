import streamlit as st
import paho.mqtt.client as mqtt
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
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
# 1. INICIALIZACIÓN DE RECURSOS GLOBALES (MQTT Y MODELO TENSORFLOW / KERAS)
# -------------------------------------------------------------------------
@st.cache_resource
def inicializar_recursos():
    try:
        from tensorflow.keras.models import load_model
        # Asegúrate de que el archivo se llame exactamente "keras_model.h5" en tu GitHub
        modelo_keras = load_model("keras_model.h5", compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"⚠️ Error al cargar el modelo 'keras_model.h5'. Verifica tu repositorio. Detalle: {e}")
        
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
# 3. MÓDULO DE PROCESAMIENTO DE IMÁGENES (TENSORFLOW / KERAS)
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model is None:
        return "Nadie", 0.0
        
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalización idéntica a Teachable Machine
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    input_data = np.expand_dims(normalized_image_array, axis=0)
    
    # Inferencia con Keras
    prediccion = model.predict(input_data, verbose=0)
    indice_maximo = np.argmax(prediccion[0])
    
    return ETIQUETAS[indice_maximo], float(prediccion[0][indice_maximo])

# -------------------------------------------------------------------------
# 4. INTERFAZ GRÁFICA PRINCIPAL (UI / UX)
# -------------------------------------------------------------------------
st.title("🐾 Panel del Comedero Inteligente")
st.write("Monitoreo automático asistido por TensorFlow / Keras y control de voz adaptativo.")

pestana_camara, pestana_voz = st.tabs(["📸 Visión Artificial Auto", "🎙️ Control por Voz"])

# --- PESTAÑA A: CÁMARA AUTOMÁTICA EN TIEMPO REAL (ZONA CONTROLADA) ---
with pestana_camara:
    st.header("Cámara del Comedero en Tiempo Real")
    
    # Marcadores dinámicos arriba del video
    contenedor_metricas = st.empty()
    contenedor_alertas = st.empty()

    # -------------------------------------------------------------------------
    # EL HUB DE COMUNICACIÓN (Ahora visible para depuración)
    # -------------------------------------------------------------------------
    # Definimos la función que Python ejecutará cuando el hub cambie de valor
    # Esto ocurre automáticamente cuando JavaScript le entrega un frame
    def procesar_cuadro_camara():
        captura_base64 = st.session_state.hub_comunicacion
        
        if captura_base64 and "base64," in captura_base64:
            try:
                # Decodificar la imagen Base64 a bytes y Pillow
                datos_limpios = captura_base64.split("base64,")[1].strip().replace(" ", "+")
                bytes_imagen = base64.b64decode(datos_limpios)
                img_pil = Image.open(io.BytesIO(bytes_imagen)).convert("RGB")
                
                # Ejecutar inferencia de la IA
                resultado, confianza = procesar_y_clasificar(img_pil)
                
                with contenedor_metricas.container():
                    st.metric(
                        label="🐾 Identificación de la IA:", 
                        value=resultado, 
                        delta=f"Confianza: {confianza * 100:.1f}%"
                    )
                    st.progress(confianza)
                
                # Filtro lógico de estabilidad contra falsos positivos
                michi_detectado_ahora = resultado if confianza > 0.70 else "Nadie"
                
                if michi_detectado_ahora == st.session_state.michi_candidato:
                    st.session_state.contador_estabilidad += 1
                else:
                    st.session_state.michi_candidato = michi_detectado_ahora
                    st.session_state.contador_estabilidad = 0
                    
                if st.session_state.contador_estabilidad >= 2:
                    if michi_detectado_ahora != st.session_state.ultimo_michi_visto:
                        st.session_state.ultimo_michi_visto = michi_detectado_ahora
                        enviar_estado_sistema()
                
                with contenedor_alertas.container():
                    if st.session_state.ultimo_michi_visto != "Nadie":
                        st.info(f"🚨 Servomotores en Wokwi ordenados para abrir el plato de **{st.session_state.ultimo_michi_visto}**.")
                    else:
                        st.success("✨ Zona despejada. Todos los platos permanecen resguardados.")
                        
            except Exception as e:
                st.error(f"Error procesando el cuadro: {e}")
        else:
            with contenedor_metricas.container():
                st.info("Esperando flujo continuo desde el navegador... (Concede permisos de cámara)")

    # Definimos el input oculto que actúa como receptor síncrono del video
    # Al ponerle 'on_change', obligamos a Streamlit a refrescar la app cada vez que llega un cuadro
    st.text_input(
        "transfer_frame_bridge", 
        key="hub_comunicacion", 
        on_change=procesar_cuadro_camara, 
        label_visibility="collapsed"
    )

    # INYECCIÓN DE COMPONENTE WEB JAVASCRIPT: Captura frames de la webcam nativa del navegador
    st.components.v1.html(
        """
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: sans-serif;">
            <video id="webcam" autoplay playsinline width="380" height="285" style="border-radius: 10px; background-color: #222; transform: scaleX(-1);"></video>
            <canvas id="canvas_oculto" width="224" height="224" style="display:none;"></canvas>
            <p style="color: #4CAF50; font-size: 13px; margin-top: 5px;">● Transmisión de Video en Vivo Activa🟢</p>
        </div>
        
        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas_oculto');
            const ctx = canvas.getContext('2d');
            
            navigator.mediaDevices.getUserMedia({ video: { width: 380, height: 285 } })
                .then((stream) => { 
                    video.srcObject = stream; 
                })
                .catch((err) => { 
                    console.error("Error webcam: ", err); 
                });
                
            setInterval(() => {
                if(video.videoWidth > 0) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    // Captura comprimida al 50% para no ahogar la red
                    const dataURL = canvas.toDataURL('image/jpeg', 0.5); 
                    
                    // Buscamos el input del puente de Streamlit
                    const doc = window.parent.document;
                    const inputs = doc.querySelectorAll('input');
                    let streamLitInput = null;
                    
                    for (let input of inputs) {
                        if (input.getAttribute('aria-label') === 'transfer_frame_bridge') {
                            streamLitInput = input;
                            break;
                        }
                    }
                    
                    // Si encontramos el input, inyectamos el valor y simulamos un cambio humano para despertar a Python
                    if (streamLitInput) {
                        streamLitInput.value = dataURL;
                        streamLitInput.dispatchEvent(new Event('input', { bubbles: true }));
                        streamLitInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            }, 1000); // 1 segundo entre capturas
        </script>
        """,
        height=330
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
                datos_audio = reconceptual = reconocedor.record(origen)
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
