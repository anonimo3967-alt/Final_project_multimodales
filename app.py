import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import cv2
import av

st.title("IA de Reconocimiento de Imágenes en Vivo 🚀")

# Esta función procesará cada frame del video en tiempo real
def video_frame_callback(frame):
    # Convertir el frame de WebRTC a un array de NumPy (formato BGR para OpenCV)
    img = frame.to_ndarray(format="bgr24")

    # ---------------------------------------------------------
    # ¡AQUÍ VA TU MODELO DE IA!
    # Ejemplo con OpenCV: Convertir a escala de grises (reemplázalo por tu IA)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # ---------------------------------------------------------

    # Retornar el frame procesado para que el usuario lo vea en su pantalla
    return av.VideoFrame.from_ndarray(output_img, format="bgr24")

# Componente de Streamlit para el Live Feed
webrtc_streamer(
    key="reconocimiento-ia",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        # Servidor STUN gratuito de Google para establecer la conexión WebRTC
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}, # Desactivar audio evita eco
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, playsInline=True)
)
