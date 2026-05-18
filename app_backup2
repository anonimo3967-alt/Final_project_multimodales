import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
import json

st.set_page_config(page_title="Control de Acceso Michi IA", layout="centered")
st.title("🐱 Sistema de Acceso Inteligente para Gatos")
st.write("Procesando video localmente en el navegador y ejecutando IA en Streamlit Cloud.")

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN MQTT Y CONFIGURACIÓN DE PARÁMETROS TRADICIONALES
# -------------------------------------------------------------------------
BROKER_IP = "157.230.214.127"
PORT = 1883
TOPIC_DIGITAL = "cmqtt_sdesi"
CLIENT_ID = "stream_client_michi_992"

# Inicializar modelo y cliente MQTT usando cache para que no ralentice la app
@st.cache_resource
def inicializar_recursos():
    # Carga tu modelo exportado de Teachable Machine
    try:
        modelo_keras = tf.keras.models.load_model('keras_model.h5', compile=False)
    except Exception as e:
        modelo_keras = None
        st.error(f"Error al cargar 'keras_model.h5'. Asegúrate de subirlo a tu GitHub: {e}")
        
    # Inicializar cliente MQTT
    cliente_mqtt = mqtt.Client(CLIENT_ID)
    try:
        cliente_mqtt.connect(BROKER_IP, PORT, 60)
        cliente_mqtt.loop_start() # Inicia el loop en segundo plano para asegurar envíos estables
    except Exception as e:
        st.error(f"No se pudo conectar al Broker MQTT ({BROKER_IP}): {e}")
        
    return modelo_keras, cliente_mqtt

model, client1 = inicializar_recursos()

# Lee las etiquetas de las clases (Normalmente Teachable Machine exporta: 0 Gato_Permitido, 1 Gato_Intruso, 2 Vacio)
# ¡MODIFICA ESTE ORDEN según cómo hayan quedado tus clases en Teachable Machine!
ETIQUETAS = ["Gato Permitido", "Gato Intruso", "Nadie"]

# Estado para controlar qué se envió por última vez y no saturar el Broker con datos idénticos
if "ultimo_comando" not in st.session_state:
    st.session_state.ultimo_comando = None

# -------------------------------------------------------------------------
# 2. PROCESAMIENTO ESTRICTO DE TEACHABLE MACHINE
# -------------------------------------------------------------------------
def procesar_y_clasificar(imagen_pil):
    if model == None:
        return "Nadie", 0.0
        
    # 1. Ajustar el tamaño exacto que espera el modelo (224, 224)
    size = (224, 224)
    image = ImageOps.fit(imagen_pil, size, Image.Resampling.LANCZOS)
    
    # 2. Convertir la imagen a un array de numpy
    image_array = np.asarray(image)
    
    # 3. Normalización matemática exacta de Teachable Machine (-1 a 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
    
    # 4. Crear el lote (Batch) para meterlo a la predicción
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 5. Predicción
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction[0])
    score = prediction[0][index]
    
    return ETIQUETAS[index], score

# -------------------------------------------------------------------------
# 3. CAPTURA EN VIVO Y CONTROL DE LOGICA IOT
# -------------------------------------------------------------------------
@st.fragment
def pipeline_camara():
    imagen_feed = camera_input_live(width=420, height=315, debounce=750)
    
    if imagen_feed:
        st.image(imagen_feed, caption="Cámara del Comedero", use_container_width=True)
        
        img_pil = Image.open(imagen_feed).convert("RGB")
        resultado, confianza = procesar_y_clasificar(img_pil)
        
        st.subheader(f"Resultado: **{resultado}**")
        st.progress(float(confianza), text=f"Confianza: {confianza*100:.2f}%")
        
        # --- NUEVA LÓGICA DE DECISIONES PARA DOS COMEDEROS ---
        # Filtramos por confianza (mínimo 75% para accionar)
        if confianza > 0.75:
            if resultado == "Gato Permitido":   # Tu Gato A
                comando_actual = "GATO_A"
            elif resultado == "Gato Intruso":   # Tu Gato B (que ahora tiene su propio plato)
                comando_actual = "GATO_B"
            else:
                comando_actual = "NADIE"
        else:
            comando_actual = "NADIE"
            
        # Enviar por MQTT solo si el estado cambió
        if comando_actual != st.session_state.ultimo_comando:
            st.session_state.ultimo_comando = comando_actual
            
            # Seguimos usando la estructura JSON que ya conoce tu ESP32
            payload = json.dumps({"Act1": comando_actual})
            
            try:
                client1.publish(TOPIC_DIGITAL, payload)
                st.toast(f"Publicado: {payload}", icon="📡")
            except Exception as e:
                st.error(f"Error MQTT: {e}")

# Ejecutar componente aislado de la cámara
pipeline_camara()

# -------------------------------------------------------------------------
# 4. CONTROLES MANUALES EXTRAS (Mantenidos de tu código viejo)
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("Controles Manuales de Respaldo")

col1, col2 = st.columns(2)
with col1:
    if st.button('Forzar Apertura (ON)', use_container_width=True):
        st.session_state.ultimo_comando = "ON"
        client1.publish(TOPIC_DIGITAL, json.dumps({"Act1": "ON"}))
        st.success("Comando manual ON enviado.")
with col2:
    if st.button('Forzar Cierre (OFF)', use_container_width=True):
        st.session_state.ultimo_comando = "OFF"
        client1.publish(TOPIC_DIGITAL, json.dumps({"Act1": "OFF"}))
        st.error("Comando manual OFF enviado.")

# Slider analógico para el servo
valores_servo = st.slider('Control manual del ángulo del Servo', 0.0, 100.0, 50.0)
if st.button('Enviar ángulo analógico'):
    payload_analog = json.dumps({"Analog": float(valores_servo)})
    client1.publish("cmqtt_adeanalogo", payload_analog)
    st.info(f"Ángulo enviado: {valores_servo}")
