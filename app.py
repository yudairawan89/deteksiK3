# app.py - Streamlit App untuk Deteksi Kepatuhan K3 Ruang Arsip

import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np

# --- Inisialisasi Model ---
model = YOLO("best.pt")  # pastikan file best.pt ada di repo GitHub

# --- Judul Aplikasi ---
st.set_page_config(page_title="Deteksi K3 Ruang Arsip", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #004080;'>
        🚨 Sistem Deteksi Kepatuhan K3 Ruang Arsip
    </h1>
    <h4 style='text-align: center; color: gray;'>
        Deteksi otomatis APD (Masker, Sarung Tangan, Sepatu) dan Objek K3 seperti APAR & Rambu
    </h4>
    <br>
""", unsafe_allow_html=True)

# --- Pilihan Input ---
input_type = st.radio("Pilih metode input:", ["Upload Gambar", "Live Kamera"], horizontal=True)

# --- Fungsi Deteksi ---
def deteksi_dan_visualisasi(img):
    results = model(img)[0]
    im_array = results.plot()
    im_pil = Image.fromarray(im_array)

    # --- Hitung skor objek non-APD ---
    skor = 0
    label_k3 = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        if label in ["APAR", "Jendela", "Rambu Evakuasi", "Sarung Tangan", "Masker", "Sepatu"]:
            skor += 1
            label_k3.append(label)

    # --- Deteksi APD pada orang ---
    apd = {"Masker": False, "Sarung Tangan": False, "Sepatu": False}
    for label in label_k3:
        if label in apd:
            apd[label] = True

    if all(apd.values()):
        status_apd = "✅ APD Lengkap"
    else:
        status_apd = "❌ APD Tidak Lengkap"

    st.image(im_pil, caption="Hasil Deteksi", use_column_width=True)
    st.markdown(f"**Status APD:** {status_apd}")
    st.markdown(f"**Tingkat Kepatuhan Ruangan:** {skor} poin dari deteksi objek")

# --- Upload Image ---
if input_type == "Upload Gambar":
    uploaded = st.file_uploader("Unggah gambar ruang arsip", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        deteksi_dan_visualisasi(img)

# --- Live Camera ---
elif input_type == "Live Kamera":
    st.warning("Fitur kamera hanya bekerja di perangkat lokal atau mobile yang didukung.")
    cam = st.camera_input("Ambil gambar dari kamera langsung")
    if cam is not None:
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        deteksi_dan_visualisasi(img)

# --- Footer ---
st.markdown("""
    <hr>
    <center><small>Developed with ❤️ by [Universitas Hang Tuah Pekanbaru]</small></center>
""", unsafe_allow_html=True)
