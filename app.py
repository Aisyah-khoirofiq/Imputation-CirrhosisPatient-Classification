import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Prediksi Status Pasien Sirosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model Pipeline ---
try:
    model = joblib.load('cirrhosis_model.pkl')
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan 'cirrhosis_model.pkl' ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()


# --- Application Header ---
st.title("🩺 Aplikasi Klasifikasi Hasil Pasien Sirosis")
st.markdown("""
Aplikasi ini menggunakan model *Random Forest* yang telah dilatih untuk memprediksi status pasien sirosis berdasarkan data klinis. 
Masukkan data pasien di sidebar untuk melihat hasil prediksi.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Input Data Pasien")
st.sidebar.markdown("Silakan masukkan nilai untuk setiap fitur di bawah ini.")

# Define mapping for categorical features for user-friendly input
drug_map = {'D-penicillamine': 0, 'Placebo': 1}
sex_map = {'F': 0, 'M': 1}
ascites_map = {'N': 0, 'Y': 1}
hepatomegaly_map = {'N': 0, 'Y': 1}
spiders_map = {'N': 0, 'Y': 1}
edema_map = {'N': 0, 'S': 1, 'Y': 2}
status_map_reverse = {0: 'D (Meninggal)', 1: 'C (Disensor)', 2: 'CL (Disensor karena Transplantasi Hati)'}


def user_input_features():
    """Creates sidebar widgets and returns a DataFrame of user inputs."""
    
    # Use columns for a better layout in the sidebar
    col1, col2 = st.sidebar.columns(2)

    n_days = col1.slider('Jumlah Hari (N_Days)', 1, 5000, 1920)
    age = col2.slider('Usia (dalam hari)', 9000, 30000, 18850)
    
    stage = st.sidebar.selectbox('Stadium Penyakit (Stage)', [1.0, 2.0, 3.0, 4.0], index=3)
    
    drug = col1.selectbox('Obat (Drug)', list(drug_map.keys()), index=1)
    sex = col2.selectbox('Jenis Kelamin (Sex)', list(sex_map.keys()), index=0)

    st.sidebar.markdown("---")

    bilirubin = col1.slider('Bilirubin (mg/dl)', 0.3, 30.0, 1.4, 0.1)
    cholesterol = col2.slider('Kolesterol (mg/dl)', 100, 1800, 315)
    albumin = col1.slider('Albumin (gm/dl)', 1.9, 5.0, 3.5, 0.1)
    copper = col2.slider('Tembaga (ug/day)', 4, 600, 96)
    alk_phos = col1.slider('Alk_Phos (U/liter)', 200, 14000, 1980)
    sgot = col2.slider('SGOT (U/ml)', 10, 620, 122)
    tryglicerides = col1.slider('Trigliserida (mg/dl)', 30, 600, 124)
    platelets = col2.slider('Trombosit (per cubic ml/1000)', 60, 730, 251)
    prothrombin = col1.slider('Protrombin (s)', 9.0, 18.0, 10.7, 0.1)

    st.sidebar.markdown("---")
    
    ascites = col1.selectbox('Asites (Ascites)', list(ascites_map.keys()))
    hepatomegaly = col2.selectbox('Hepatomegali (Hepatomegaly)', list(hepatomegaly_map.keys()))
    spiders = col1.selectbox('Spiders', list(spiders_map.keys()))
    edema = col2.selectbox('Edema', list(edema_map.keys()))

    # Map categorical inputs to their numeric equivalents
    data = {
        'N_Days': n_days,
        'Age': age,
        'Bilirubin': bilirubin,
        'Cholesterol': cholesterol,
        'Albumin': albumin,
        'Copper': copper,
        'Alk_Phos': alk_phos,
        'SGOT': sgot,
        'Tryglicerides': tryglicerides,
        'Platelets': platelets,
        'Prothrombin': prothrombin,
        'Stage': stage,
        'Drug': drug_map[drug],
        'Sex': sex_map[sex],
        'Ascites': ascites_map[ascites],
        'Hepatomegaly': hepatomegaly_map[hepatomegaly],
        'Spiders': spiders_map[spiders],
        'Edema': edema_map[edema],
    }
    
    # The model was trained on these columns in this specific order
    feature_order = [
        'N_Days', 'Age', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders',
        'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
        'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage'
    ]
    
    features = pd.DataFrame(data, index=[0])
    # Ensure the order of columns matches the training data
    features = features[feature_order] 
    return features


# Get user input
input_df = user_input_features()

# --- Main Panel for Displaying Results ---
st.subheader("Ringkasan Data Input Pasien")
st.write(input_df)

# Prediction button
if st.button('🔮 Prediksi Status Pasien'):
    try:
        # Get prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Get the predicted status label
        predicted_status = status_map_reverse[prediction[0]]

        st.subheader("Hasil Prediksi")
        
        # Display the result with a colored box
        if prediction[0] == 0: # Meninggal
            st.error(f"**Status Prediksi:** {predicted_status}")
        elif prediction[0] == 2: # Transplantasi
            st.warning(f"**Status Prediksi:** {predicted_status}")
        else: # Disensor
            st.success(f"**Status Prediksi:** {predicted_status}")

        # Display probabilities
        st.subheader("Tingkat Keyakinan Model (Probabilitas)")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=[status_map_reverse[i] for i in range(len(status_map_reverse))],
            index=['Probabilitas']
        )
        st.write(proba_df)

        # Visualize probabilities with a bar chart
        st.bar_chart(proba_df.T)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")


# --- Disclaimer ---
st.markdown("---")
st.warning("""
**Penafian (Disclaimer):** Hasil prediksi dari model ini adalah untuk tujuan informasi dan demonstrasi saja. 
Model ini tidak boleh digunakan sebagai pengganti diagnosis, nasihat, atau perawatan medis profesional. 
Selalu konsultasikan dengan dokter atau penyedia layanan kesehatan yang berkualifikasi untuk masalah medis apa pun.
""")
