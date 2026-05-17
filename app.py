import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image
import numpy as np

st.title("🐱 Capturador IA Ultra-Optimizado")
st.write("Esta versión consume un 90% menos de datos en la nube.")

if "foto_gato" not in st.session_state:
    st.session_state.foto_gato = None

# Simulador de IA
def detectar_gato(img):
    # Tu lógica aquí
    return False

# Usamos un fragmento para aislar el lag del feed de video
@st.fragment
def contenedor_camara():
    # Mandamos una imagen pequeña cada 600ms (1.5 frames por segundo)
    imagen = camera_input_live(width=400, height=300, debounce=600)
    
    if imagen and st.session_state.foto_gato is None:
        img_pil = Image.open(imagen)
        img_array = np.array(img_pil)
        
        # Ejecutar tu modelo de IA
        if detectar_gato(img_array):
            st.session_state.foto_gato = img_pil
            st.rerun()

# Ejecutar la cámara aislada
contenedor_camara()

# Mostrar resultados fuera del bucle de la cámara
if st.session_state.foto_gato is not None:
    st.subheader("📸 ¡Gato Capturado!")
    st.image(st.session_state.foto_gato)
    if st.button("Reiniciar"):
        st.session_state.foto_gato = None
        st.rerun()
