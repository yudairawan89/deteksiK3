# app.py - Streamlit App untuk Deteksi Kepatuhan K3 Ruang Arsip (Snapshot Akumulasi)

import streamlit as st
import cv2
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- Load Model YOLO (TorchScript) ---
model = YOLO("best.torchscript")  # pastikan file ini sudah ada di root repo

# --- Setup Page ---
st.set_page_config(page_title="Deteksi K3 Ruang Arsip", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #004080;'>🚨 Sistem Deteksi Kepatuhan K3 Ruang Arsip</h1>
    <h4 style='text-align: center; color: gray;'>Deteksi otomatis APD (Masker, Sarung Tangan, Sepatu) dan Objek K3 seperti APAR & Rambu Evakuasi</h4><br>
""", unsafe_allow_html=True)

# --- Inisialisasi Session State ---
if "detected_labels" not in st.session_state:
    st.session_state.detected_labels = []

# --- Pilihan Input ---
input_type = st.radio("Pilih metode input:", ["Upload Gambar", "Snapshot Kamera"], horizontal=True)

# --- Fungsi Deteksi ---
def deteksi_dan_visualisasi(img, show_output=True):
    results = model(img)[0]
    im_array = results.plot()
    im_pil = Image.fromarray(im_array)

    # Deteksi dan Simpan Label
    label_k3 = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        if label in ["APAR", "Jendela", "Rambu Evakuasi", "Sarung Tangan", "Masker", "Sepatu"]:
            label_k3.append(label)
            st.session_state.detected_labels.append(label)

    # Evaluasi APD (hanya snapshot ini)
    apd = {"Masker": False, "Sarung Tangan": False, "Sepatu": False}
    for label in label_k3:
        if label in apd:
            apd[label] = True
    status_apd = "✅ APD Lengkap" if all(apd.values()) else "❌ APD Tidak Lengkap"

    if show_output:
        st.image(im_pil, caption="📷 Hasil Deteksi Snapshot", use_column_width=True)
        st.markdown(f"**Status APD (Snapshot):** {status_apd}")
        st.markdown(f"**Objek K3 Terdeteksi:** {', '.join(label_k3) if label_k3 else 'Tidak ada'}")

# --- Upload Image ---
if input_type == "Upload Gambar":
    uploaded = st.file_uploader("Unggah gambar ruang arsip", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        deteksi_dan_visualisasi(img)

# --- Snapshot Kamera ---
elif input_type == "Snapshot Kamera":
    cam = st.camera_input("Ambil Gambar dari Kamera")
    if cam is not None:
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        deteksi_dan_visualisasi(img)

# --- Hitung Skor Akhir ---
st.markdown("---")
if st.button("🔍 Hitung Skor Akhir dari Semua Snapshot"):
    semua_label = st.session_state.detected_labels
    unik = set(semua_label)

    # Hitung hanya objek K3 non-APD
    objek_k3_non_apd = {"APAR", "Jendela", "Rambu Evakuasi"}
    jumlah_k3 = sum(1 for obj in unik if obj in objek_k3_non_apd)

    st.success(f"📊 Tingkat Kepatuhan Ruangan: **{jumlah_k3} poin** dari objek K3 yang terdeteksi.")
    st.info(f"🧾 Objek Unik Terdeteksi: {', '.join(sorted(unik)) if unik else 'Tidak ada'}")

# --- Reset Data ---
if st.button("🔄 Reset Semua Snapshot"):
    st.session_state.detected_labels.clear()
    st.rerun()

# --- Footer ---
st.markdown("<hr><center><small>Developed with ❤️ by Universitas Hang Tuah Pekanbaru</small></center>", unsafe_allow_html=True)
