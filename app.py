# app.py - Streamlit App untuk Deteksi Kepatuhan K3 Ruang Arsip (Full Snapshot & Laporan APD)

import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- Load Model YOLO ---
model = YOLO("best.torchscript")

# --- Setup Page ---
st.set_page_config(page_title="Deteksi K3 Ruang Arsip", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #004080;'>🚨 Sistem Deteksi Kepatuhan K3 Ruang Arsip</h1>
    <h4 style='text-align: center; color: gray;'>Deteksi otomatis APD (Masker, Sarung Tangan, Sepatu) dan Objek K3 seperti APAR & Rambu Evakuasi</h4><br>
""", unsafe_allow_html=True)

# --- Inisialisasi Session State ---
if "detected_labels" not in st.session_state:
    st.session_state.detected_labels = []
if "detected_images" not in st.session_state:
    st.session_state.detected_images = []
if "last_apd_status" not in st.session_state:
    st.session_state.last_apd_status = []

# --- Fungsi Deteksi ---
def deteksi_dan_visualisasi(img):
    results = model(img)[0]
    im_array = results.plot()
    im_pil = Image.fromarray(im_array)

    # Simpan hasil gambar ke list snapshot
    st.session_state.detected_images.append(im_pil)

    # Label & Kategori
    label_k3 = []
    apd_labels = {"Masker": False, "Sarung Tangan": False, "Sepatu": False}
    
    for box in results.boxes:
        label = model.names[int(box.cls)]
        if label in ["APAR", "Jendela", "Rambu Evakuasi", "Sarung Tangan", "Masker", "Sepatu"]:
            label_k3.append(label)
            st.session_state.detected_labels.append(label)
            if label in apd_labels:
                apd_labels[label] = True

    # Simpan status APD terbaru
    st.session_state.last_apd_status = apd_labels

# --- Input ---
input_type = st.radio("Pilih metode input:", ["Upload Gambar", "Snapshot Kamera"], horizontal=True)

if input_type == "Upload Gambar":
    uploaded = st.file_uploader("Unggah gambar ruang arsip", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        deteksi_dan_visualisasi(img)

elif input_type == "Snapshot Kamera":
    cam = st.camera_input("Ambil Gambar dari Kamera")
    if cam is not None:
        file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        deteksi_dan_visualisasi(img)

# --- Tampilkan Semua Snapshot ---
if st.session_state.detected_images:
    st.markdown("## 📸 Snapshot Deteksi Sebelumnya:")
    for i, gambar in enumerate(st.session_state.detected_images):
        st.image(gambar, caption=f"Hasil Deteksi #{i+1}", use_column_width=True)

# --- Hitung Skor Akhir ---
st.markdown("---")
if st.button("🔍 Hitung Skor Akhir dari Semua Snapshot"):
    semua_label = st.session_state.detected_labels
    unik = set(semua_label)

    # Penilaian Kepatuhan
    objek_k3_non_apd = {"APAR", "Jendela", "Rambu Evakuasi"}
    jumlah_k3 = sum(1 for obj in unik if obj in objek_k3_non_apd)

    # Evaluasi Patuh atau Tidak
    if jumlah_k3 >= 2:
        status_kepatuhan = "✅ Tingkat Kepatuhan Ruangan: **Patuh**"
    else:
        status_kepatuhan = "❌ Tingkat Kepatuhan Ruangan: **Tidak Patuh**"

    st.success(status_kepatuhan)

    # Laporan APD Orang dari Snapshot Terakhir
    apd = st.session_state.last_apd_status
    st.markdown("### 👤 Status APD Individu (Snapshot Terakhir):")
    for key, val in apd.items():
        icon = "✅" if val else "❌"
        st.write(f"{icon} {key}")

    # Laporan APD yang berhasil terdeteksi
    apd_terdeteksi = [l for l in unik if l in ["Masker", "Sarung Tangan", "Sepatu"]]
    if apd_terdeteksi:
        st.info(f"🧰 Perlengkapan APD Terdeteksi: {', '.join(sorted(apd_terdeteksi))}")
    else:
        st.info("🧰 Perlengkapan APD Terdeteksi: Tidak ada")

# --- Reset ---
if st.button("🔄 Reset Semua Snapshot"):
    st.session_state.detected_labels.clear()
    st.session_state.detected_images.clear()
    st.session_state.last_apd_status.clear()
    st.rerun()

# --- Footer ---
st.markdown("<hr><center><small>Developed with ❤️ by Universitas Hang Tuah Pekanbaru</small></center>", unsafe_allow_html=True)
