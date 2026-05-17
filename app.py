import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt

st.title("🐱 Clasificador de Gatos IA + IoT (Wokwi)")

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE MQTT & MODELO IA (Se ejecutan una sola vez para optimizar)
# -------------------------------------------------------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "mi_casa/gatos/sistema_alerta" # Asegúrate de que sea el mismo en Wokwi

@st.cache_resource
def inicializar_modelo_y_mqtt():
    # Cargar el modelo de Keras (Sube tu archivo 'modelo_gatos.h5' a tu repositorio de GitHub)
    try:
        modelo = tf.keras.models.load_model("modelo_gatos.h5")
    except Exception as e:
        modelo = None
        st.error(f"No se pudo cargar el modelo 'modelo_gatos.h5'. Verifica que esté en tu GitHub. Error: {e}")
    
    # Inicializar cliente MQTT
    cliente_mqtt = mqtt.Client()
    cliente_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    return modelo, cliente_mqtt

modelo, cliente_mqtt = inicializar_modelo_y_mqtt()

# Clases de tu modelo (Ajusta el orden según cómo entrenaste tu red)
CLASES = ["Gato 1 (Michi A)", "Gato 2 (Michi B)", "Nadie"]

# Estado para evitar enviar MQTT repetidos innecesariamente
if "ultimo_estado" not in st.session_state:
    st.session_state.ultimo_estado = "Nadie"

# -------------------------------------------------------------------------
# 2. FUNCIÓN DE PREDICCIÓN DE LA IA
# -------------------------------------------------------------------------
def clasificar_imagen(imagen_pil, modelo_keras):
    if modelo_keras is None:
        return "Nadie", 1.0
        
    # Redimensionar la imagen al tamaño que pide tu modelo (comúnmente 224x224 o 150x150)
    # AJUSTA ESTE TAMAÑO al que usaste para entrenar tu modelo
    TAMAÑO_IA = (224, 224) 
    imagen_redimensionada = ImageOps.fit(imagen_pil, TAMAÑO_IA, Image.Resampling.LANCZOS)
    
    # Convertir a array de numpy y normalizar (si tu modelo fue entrenado con valores de 0 a 1)
    img_array = np.asarray(imagen_redimensionada, dtype=np.float32) / 255.0
    
    # Añadir la dimensión del "Batch" (Keras espera recibir un lote de imágenes: [1, 224, 224, 3])
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Hacer la predicción
    prediccion = modelo_keras.predict(img_batch)
    indice_resultado = np.argmax(prediccion[0])
    confianza = prediccion[0][indice_resultado]
    
    # Si la confianza es muy baja, asumimos que no hay nadie con certeza
    if confianza < 0.70: 
        return "Nadie", confianza
        
    return CLASES[indice_resultado], confianza

# -------------------------------------------------------------------------
# 3. INTERFAZ DE LA CÁMARA Y PROCESAMIENTO CONTINUO
# -------------------------------------------------------------------------
@st.fragment
def contenedor_camara():
    # Captura ligera cada 700ms para no saturar la CPU de la nube procesando Keras
    imagen = camera_input_live(width=400, height=300, debounce=700)
    
    if imagen:
        st.image(imagen, caption="Monitoreo en vivo", use_container_width=True)
        
        img_pil = Image.open(imagen).convert("RGB")
        
        # Ejecutar la IA
        resultado, confianza = clasificar_imagen(img_pil, modelo)
        
        # Mostrar el resultado en pantalla en tiempo real
        st.metric(label="Detección actual:", value=resultado, delta=f"{confianza*100:.1f}% confianza")
        
        # Enviar señal por MQTT solo si cambió el estado detectado
        if resultado != st.session_state.ultimo_estado:
            st.session_state.ultimo_estado = resultado
            
            # Enviar el mensaje a Wokwi (Convertimos el texto a comandos simples, ej: "GATO_1", "GATO_2", "VACIO")
            comando = "VACIO"
            if resultado == "Gato 1 (Michi A)": comando = "GATO_1"
            elif resultado == "Gato 2 (Michi B)": comando = "GATO_2"
            
            cliente_mqtt.publish(MQTT_TOPIC, comando)
            st.toast(f"Señal MQTT enviada a Wokwi: {comando}", icon="📡")

contenedor_camara()
