import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image
import numpy as np

st.title("🐱 Capturador IA con Vista Previa")

if "foto_gato" not in st.session_state:
    st.session_state.foto_gato = None

# Simulador de tu modelo de IA
def detectar_gato(img):
    # Aquí irá tu lógica (YOLO, OpenCV, etc.)
    return False

@st.fragment
def contenedor_camara():
    # 1. El componente invisible captura el frame de la webcam
    imagen = camera_input_live(width=400, height=300, debounce=600)
    
    if imagen:
        # 2. ¡ESTA LÍNEA ES LA SOLUCIÓN! Muestra el feed en vivo en la pantalla
        st.image(imagen, caption="Tu cámara en vivo", use_container_width=True)
        
        # 3. Procesamiento de la IA en segundo plano
        if st.session_state.foto_gato is None:
            img_pil = Image.open(imagen)
            img_array = np.array(img_pil)
            
            if detectar_gato(img_array):
                st.session_state.foto_gato = img_pil
                st.rerun()

# Ejecutamos la cámara con su vista previa
contenedor_camara()

# Si la IA detecta al gato, congela la foto aquí abajo
if st.session_state.foto_gato is not None:
    st.subheader("📸 ¡Gato Detectado y Guardado!")
    st.image(st.session_state.foto_gato, caption="Captura congelada por la IA")
    if st.button("Buscar otro gato"):
        st.session_state.foto_gato = None
        st.rerun()
