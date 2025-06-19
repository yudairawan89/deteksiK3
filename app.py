import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import torch
from ultralytics import YOLO
import os

# =============================
# Konfigurasi Tampilan Streamlit
# =============================
st.set_page_config(
    page_title="Deteksi Kepatuhan K3 Ruang Arsip",
    layout="wide",
    page_icon="🚀"
)

st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .title {
            font-size: 32px;
            color: #2c3e50;
            font-weight: bold;
        }
        .subtitle {
            font-size: 18px;
            color: #7f8c8d;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown("<div class='title'>🌟 Sistem Deteksi Kepatuhan K3 Ruang Arsip</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Berbasis YOLOv12 + Streamlit | Deteksi Masker, Sepatu, Sarung Tangan & Evaluasi Kepatuhan APD</div><br>", unsafe_allow_html=True)

# =============================
# Load Model YOLOv12
# =============================
@st.cache_resource

def load_model():
    model_path = os.path.join("model", "best.pt")
    model = YOLO(model_path)
    return model

model = load_model()

# =============================
# Fitur Pilihan: Upload atau Kamera
# =============================
option = st.radio("Pilih Metode Deteksi:", ("📷 Upload Gambar", "📹 Kamera Langsung"))

# =============================
# Deteksi dari Gambar Upload
# =============================
if option == "📷 Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar ruang arsip:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # Jalankan deteksi YOLO
        results = model.predict(source=img_np, conf=0.4, imgsz=640, device=0 if torch.cuda.is_available() else "cpu")

        # Ambil hasil deteksi dan render gambar
        rendered = results[0].plot()

        st.image(rendered, caption="Hasil Deteksi K3", use_column_width=True)

        # TODO: Tambahkan analisis skor dan status "APD Lengkap / Tidak Lengkap"

# =============================
# Deteksi dari Kamera Langsung
# =============================
elif option == "📹 Kamera Langsung":
    stframe = st.empty()
    run = st.checkbox("Mulai Kamera")
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Kamera tidak tersedia.")
            break

        results = model.predict(source=frame, conf=0.4, imgsz=640, device=0 if torch.cuda.is_available() else "cpu")
        rendered = results[0].plot()

        stframe.image(rendered, channels="BGR", use_column_width=True)

    cap.release()
    st.success("Deteksi kamera dihentikan.")
