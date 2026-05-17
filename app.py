import streamlit as st
from camera_input_live import camera_input_live
from PIL import Image
import numpy as np
import time

st.title("🐱 Capturador Automático de Gatos")
st.write("Apunta la cámara. En cuanto aparezca un gato, se guardará una foto automáticamente.")

# Inicializar una variable en la sesión para guardar la foto del gato detectado
if "foto_gato" not in st.session_state:
    st.session_state.foto_gato = None

# Función simulada de tu IA (Reemplázala con tu modelo real: YOLO, OpenCV, etc.)
def detectar_gato(imagen_np):
    # AQUÍ CORRES TU MODELO: resultado = modelo(imagen_np)
    # Por ahora simulemos que si la app lleva 5 segundos abierta, "detecta" un gato.
    # Devuelve True si hay un gato, False si no.
    return False 

# 1. El feed de video corre continuamente en segundo plano
imagen_en_vivo = camera_input_live(debounce=200) # analiza un frame cada 200ms

if imagen_en_vivo and st.session_state.foto_gato is None:
    # Convertir la imagen del feed a formato procesable por la IA
    img = Image.open(imagen_en_vivo)
    img_array = np.array(img)
    
    # 2. La IA analiza el frame continuamente
    gato_detectado = detectar_gato(img_array)
    
    if gato_detectado:
        # 3. ¡ACCION AUTOMÁTICA! Si hay un gato, guardamos el frame inmediatamente
        st.session_state.foto_gato = img
        st.toast("¡Gato detectado! Guardando captura...", icon="🐱")
        st.rerun() # Reinicia la app para mostrar la foto guardada

# --- SECCIÓN DE RESULTADOS ---
# Si ya se guardó la foto del gato, la mostramos abajo
if st.session_state.foto_gato is not None:
    st.subheader("📸 ¡Captura guardada automáticamente!")
    st.image(st.session_state.foto_gato, caption="Gato capturado por la IA")
    
    # Botón manual por si quieren reiniciar el sistema y buscar otro gato
    if st.button("Buscar otro gato"):
        st.session_state.foto_gato = None
        st.rerun()
