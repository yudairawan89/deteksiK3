# app.py - Deteksi K3 Ruangan dengan Snapshot + Hapus Per Gambar

import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- Load Model ---
model = YOLO("best.torchscript")

# --- Setup UI ---
st.set_page_config(page_title="Deteksi K3 Ruang Arsip", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #004080;'>🚨 Sistem Deteksi Kepatuhan K3 Ruang Arsip</h1>
    <h4 style='text-align: center; color: gray;'>Deteksi otomatis APD (Masker, Sarung Tangan, Sepatu) dan Objek K3 seperti APAR & Rambu Evakuasi</h4><br>
""", unsafe_allow_html=True)

# --- Session States ---
if "detected_labels" not in st.session_state:
    st.session_state.detected_labels = []
if "detected_images" not in st.session_state:
    st.session_state.detected_images = []
if "last_apd_status" not in st.session_state:
    st.session_state.last_apd_status = {}
if "delete_index" not in st.session_state:
    st.session_state.delete_index = None

# --- Hapus Gambar jika diminta ---
if st.session_state.delete_index is not None:
    del st.session_state.detected_images[st.session_state.delete_index]
    st.session_state.delete_index = None
    st.rerun()

# --- Fungsi Deteksi ---
def deteksi_dan_visualisasi(img):
    results = model(img)[0]
    im_array = results.plot()
    im_pil = Image.fromarray(im_array)
    st.session_state.detected_images.append(im_pil)

    label_k3 = []
    apd_labels = {"Masker": False, "Sarung Tangan": False, "Sepatu": False}
    for box in results.boxes:
        label = model.names[int(box.cls)]
        if label in ["APAR", "Jendela", "Rambu Evakuasi", "Sarung Tangan", "Masker", "Sepatu"]:
            label_k3.append(label)
            st.session_state.detected_labels.append(label)
            if label in apd_labels:
                apd_labels[label] = True
    st.session_state.last_apd_status = apd_labels

# --- Input Gambar ---
input_type = st.radio("Pilih metode input:", ["Upload Gambar", "Snapshot Kamera"], horizontal=True)
with st.form("input_form"):
    if input_type == "Upload Gambar":
        uploaded = st.file_uploader("Unggah gambar ruang arsip", type=["jpg", "png", "jpeg"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
    else:
        cam = st.camera_input("Ambil Gambar dari Kamera")
        if cam:
            file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
        else:
            img = None
    submitted = st.form_submit_button("🔍 Deteksi Gambar Sekarang")
    if submitted and img is not None:
        deteksi_dan_visualisasi(img)

# --- Tampilkan Snapshot + Tombol Hapus ---
if st.session_state.detected_images:
    st.markdown("## 📸 Snapshot Deteksi Sebelumnya:")
    for i, gambar in enumerate(st.session_state.detected_images):
        col1, col2 = st.columns([6, 1])
        with col1:
            st.image(gambar, caption=f"Hasil Deteksi #{i+1}", use_column_width=True)
        with col2:
            if st.button(f"❌ Hapus #{i+1}", key=f"hapus_{i}"):
                st.session_state.delete_index = i
                st.rerun()

# --- Hitung Skor Akhir ---
st.markdown("---")
if st.button("📊 Hitung Tingkat Kepatuhan"):
    unik = set(st.session_state.detected_labels)
    objek_k3_non_apd = {"APAR", "Jendela", "Rambu Evakuasi"}
    jumlah_k3 = sum(1 for obj in unik if obj in objek_k3_non_apd)
    status_kepatuhan = "✅ Patuh" if jumlah_k3 >= 2 else "❌ Tidak Patuh"
    st.success(f"Tingkat Kepatuhan Ruangan: {status_kepatuhan}")

    st.markdown("### 👤 Status APD Individu (Snapshot Terakhir):")
    for key, val in st.session_state.last_apd_status.items():
        icon = "✅" if val else "❌"
        st.write(f"{icon} {key}")

    apd_terdeteksi = [l for l in unik if l in ["Masker", "Sarung Tangan", "Sepatu"]]
    st.info(f"🧰 Perlengkapan APD Terdeteksi: {', '.join(sorted(apd_terdeteksi)) if apd_terdeteksi else 'Tidak ada'}")

# --- Reset Semua ---
if st.button("🔄 Reset Semua Snapshot"):
    st.session_state.detected_labels.clear()
    st.session_state.detected_images.clear()
    st.session_state.last_apd_status.clear()
    st.session_state.delete_index = None
    st.rerun()

# --- Footer ---
st.markdown("<hr><center><small>Developed with ❤️ by Universitas Hang Tuah Pekanbaru</small></center>", unsafe_allow_html=True)
