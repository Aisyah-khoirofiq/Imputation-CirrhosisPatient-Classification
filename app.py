import streamlit as st
import pandas as pd
import joblib
import altair as alt # Import Altair

# =====================================================
# 🩺 KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Status Pasien Sirosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KUSTOM UNTUK MEMPERBESAR FONT (LEBIH AGRESIF) ---
st.markdown("""
<style>
/* 1. Memperbesar Judul Utama */
h1 {
    font-size: 3.5em !important;
}

/* 2. Memperbesar Sub-Header */
h2, h3 {
    font-size: 2.2em !important;
}

/* 3. Memperbesar Teks Label Input (Paling penting untuk Number Input dan Selectbox) */
/* Menargetkan class utama untuk label widget */
[data-testid="stForm"] label, 
.st-emotion-cache-1wb9f5u, /* Label for Number/Text Input */
.st-emotion-cache-1erzcv, /* Label for Selectbox/Multiselect */
label.st-emotion-cache-1wb9f5u,
label.st-emotion-cache-1erzcv {
    font-size: 1.35rem; /* Ukuran font label */
    font-weight: bold;
}

/* 4. Memperbesar Teks Nilai Input (Angka/Pilihan yang dipilih) */
[data-testid="stForm"] input,
[data-testid="stForm"] .st-emotion-cache-1n76ch6 { 
    font-size: 1.25rem !important;
}

/* 5. Memperbesar Teks di Tombol Prediksi */
.stButton>button {
    font-size: 1.6rem;
    height: auto;
    padding: 12px 25px;
    font-weight: bold;
}

/* 6. Memperbesar Teks pada Tabel (Ringkasan Data & Probabilitas) */
.dataframe {
    font-size: 1.3rem !important;
    line-height: 1.8rem;
}

/* 7. Memperbesar Teks Output Prediksi (Success/Error/Warning Banners) */
[data-testid="stAlert"] {
    font-size: 1.4rem;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------

# =====================================================
# 🧠 MEMUAT MODEL YANG TELAH DILATIH
# =====================================================
try:
    model = joblib.load('cirrhosis_model.pkl')
except FileNotFoundError:
    st.error("❌ File model tidak ditemukan. Pastikan file 'cirrhosis_model.pkl' berada di direktori yang sama dengan aplikasi ini.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# =====================================================
# 🧾 JUDUL DAN DESKRIPSI APLIKASI
# =====================================================
st.title("🩺 Aplikasi Prediksi Status Pasien Sirosis")
st.markdown("""
Aplikasi ini menggunakan model **Support Vector Machine** yang telah dilatih untuk memprediksi status pasien sirosis 
berdasarkan data klinis dan laboratorium.  
Silakan isi data pasien pada tabel input di bawah ini untuk mendapatkan hasil prediksi.
""")

# =====================================================
# 📋 INPUT DATA TABULAR DI HALAMAN UTAMA
# =====================================================
st.subheader("🧍‍♂️ Formulir Input Data Pasien")

# Pemetaan fitur kategorikal
drug_map = {'D-penicillamine': 0, 'Placebo': 1}
sex_map = {'Perempuan (F)': 0, 'Laki-laki (M)': 1}
ascites_map = {'Tidak (N)': 0, 'Ya (Y)': 1}
hepatomegaly_map = {'Tidak (N)': 0, 'Ya (Y)': 1}
spiders_map = {'Tidak (N)': 0, 'Ya (Y)': 1}
edema_map = {'Tidak (N)': 0, 'Sedikit (S)': 1, 'Ya (Y)': 2}

# --- PENTING: MAPPING STATUS DIPERBAIKI UNTUK LABEL PENDEK DAN URUTAN ---
# Urutan: 0, 1, 2
status_map_reverse = {
    0: 'Meninggal', 
    1: 'Disensor',
    2: 'Transplan', # Label disingkat agar tidak terpotong
}
# --------------------------------------------------------------------------

# Tata letak kolom input
col1, col2, col3 = st.columns(3)

with col1:
    n_days = st.number_input('Jumlah Hari (N_Days)', min_value=1, max_value=5000, value=1920)
    bilirubin = st.number_input('Bilirubin (mg/dl)', min_value=0.3, max_value=30.0, value=1.4, step=0.1)
    albumin = st.number_input('Albumin (gm/dl)', min_value=1.9, max_value=5.0, value=3.5, step=0.1)
    alk_phos = st.number_input('Alkali Fosfatase (U/liter)', min_value=200, max_value=14000, value=1980)
    tryglicerides = st.number_input('Trigliserida (mg/dl)', min_value=30, max_value=600, value=124)
    ascites = st.selectbox('Asites (Penumpukan Cairan Perut)', list(ascites_map.keys()))

with col2:
    age = st.number_input('Usia (hari)', min_value=9000, max_value=30000, value=18850)
    cholesterol = st.number_input('Kolesterol (mg/dl)', min_value=100, max_value=1800, value=315)
    copper = st.number_input('Tembaga (ug/day)', min_value=4, max_value=600, value=96)
    sgot = st.number_input('SGOT (U/ml)', min_value=10, max_value=620, value=122)
    platelets = st.number_input('Trombosit (/1000)', min_value=60, max_value=730, value=251)
    hepatomegaly = st.selectbox('Hepatomegali (Pembesaran Hati)', list(hepatomegaly_map.keys()))

with col3:
    stage = st.selectbox('Stadium Penyakit (Stage)', [1.0, 2.0, 3.0, 4.0], index=3)
    drug = st.selectbox('Jenis Obat yang Diberikan', list(drug_map.keys()), index=1)
    sex = st.selectbox('Jenis Kelamin', list(sex_map.keys()), index=0)
    prothrombin = st.number_input('Protrombin (s)', min_value=9.0, max_value=18.0, value=10.7, step=0.1)
    spiders = st.selectbox('Adakah Spiders?', list(spiders_map.keys()))
    edema = st.selectbox('Kondisi Edema (Pembengkakan)', list(edema_map.keys()))

# Membentuk DataFrame input pengguna
data = {
    'N_Days': n_days,
    'Age': age,
    'Drug': drug_map[drug],
    'Sex': sex_map[sex],
    'Ascites': ascites_map[ascites],
    'Hepatomegaly': hepatomegaly_map[hepatomegaly],
    'Spiders': spiders_map[spiders],
    'Edema': edema_map[edema],
    'Bilirubin': bilirubin,
    'Cholesterol': cholesterol,
    'Albumin': albumin,
    'Copper': copper,
    'Alk_Phos': alk_phos,
    'SGOT': sgot,
    'Tryglicerides': tryglicerides,
    'Platelets': platelets,
    'Prothrombin': prothrombin,
    'Stage': stage
}

input_df = pd.DataFrame(data, index=[0])

# =====================================================
# 🧩 MENYESUAIKAN KOLOM DENGAN MODEL (FIX ORDER/NAMA)
# =====================================================
if hasattr(model, 'feature_names_in_'):
    expected_features = model.feature_names_in_.tolist()
    
    # Periksa apakah ada kolom yang hilang atau berlebih
    missing = [col for col in expected_features if col not in input_df.columns]
    
    if missing:
        st.error(f"❌ Kolom yang diharapkan model hilang: {missing}")
        st.stop()
    
    # Paksakan urutan DataFrame input agar sama dengan urutan model
    try:
        input_df = input_df[expected_features]
    except KeyError as e:
        st.error(f"⚠️ Kesalahan saat mengurutkan kolom. Pastikan nama-nama kolom input sudah benar: {e}")
        st.stop()
else:
    st.warning("Model tidak menyediakan daftar fitur yang diharapkan. Melanjutkan dengan urutan input saat ini.")


# =====================================================
# 📊 MENAMPILKAN DATA INPUT
# =====================================================
st.markdown("### 📋 Tabel Ringkasan Data Pasien")
st.dataframe(input_df, use_container_width=True)

# =====================================================
# 🔮 PROSES PREDIKSI
# =====================================================
if st.button('🔮 Jalankan Prediksi'):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        predicted_status = status_map_reverse[prediction[0]]
        st.subheader("📈 Hasil Prediksi")

        # Warna hasil sesuai kondisi
        if prediction[0] == 0:
            st.error(f"**Status Prediksi:** {predicted_status}")
        elif prediction[0] == 2:
            st.warning(f"**Status Prediksi:** {predicted_status}")
        else:
            st.success(f"**Status Prediksi:** {predicted_status}")

        # Menampilkan probabilitas
        st.subheader("📊 Tingkat Keyakinan Model (Probabilitas)")

        status_cols = [status_map_reverse[i] for i in range(len(status_map_reverse))]
        
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=status_cols,
            index=['Probabilitas']
        )
        st.dataframe(proba_df.style.format("{:.6f}"), use_container_width=True) 

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat melakukan prediksi: {e}")

# =====================================================
# ⚠️ PENAFIAN
# =====================================================
st.markdown("---")
st.warning("""
**(Disclaimer):** 
Hasil prediksi dari model ini hanya digunakan untuk **tujuan informasi dan edukasi**.  
Aplikasi ini **bukan alat diagnosis medis** dan tidak dapat menggantikan keputusan dokter atau tenaga kesehatan profesional.
""")
