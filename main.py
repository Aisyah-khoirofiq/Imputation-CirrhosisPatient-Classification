import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import joblib
import io

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Patient Outcome Classifier",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fungsi Inti ---

@st.cache_data
def load_data(uploaded_file):
    """Memuat dan membersihkan file CSV yang diunggah."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    cols_to_drop = [col for col in df.columns if col.lower().strip() in ['id', 'unnamed: 0']]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.session_state.info_message = f"Info: Kolom identifikasi seperti **{', '.join(cols_to_drop)}** telah dihapus secara otomatis."
    return df

@st.cache_data(show_spinner=False)
def train_and_evaluate(_df, target_column, model_name):
    """Fungsi utama untuk melatih dan mengevaluasi model."""
    with st.spinner(f'Melatih model {model_name} dengan parameter optimal...'):
        X = _df.drop(target_column, axis=1)
        y = _df[target_column]

        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(exclude=np.number).columns

        numeric_transformer = Pipeline(steps=[('imputer', KNNImputer()), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
        
        # Hyperparameter yang telah ditentukan
        params = {
            'Support Vector Machine': {
                'preprocessor__numeric__imputer__n_neighbors': 13, 'classifier__C': 1, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf'
            },
            'Random Forest': {
                'preprocessor__numeric__imputer__n_neighbors': 3, 'classifier__n_estimators': 100, 'classifier__max_depth': None
            }
        }
        
        classifier = SVC(probability=True, random_state=42) if model_name == 'Support Vector Machine' else RandomForestClassifier(random_state=42)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])

        pipeline.set_params(**params.get(model_name, {}))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        return pipeline, accuracy, report, cm, params.get(model_name, {})

# --- Inisialisasi Session State ---
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.trained_models = {}
    st.session_state.feature_info = None
    st.session_state.df = None
    st.session_state.target_column = 'Status'

# --- Tampilan Utama Streamlit ---
st.title("⚕️ Klasifikasi Hasil Pasien Interaktif")
st.markdown("""
Aplikasi ini memungkinkan Anda untuk melatih model klasifikasi untuk memprediksi hasil pasien dan kemudian menggunakan model yang telah dilatih untuk membuat prediksi baru.
- **Tab Pelatihan & Evaluasi Model**: Unggah dataset Anda, latih model, bandingkan performa, dan unduh model yang telah dilatih.
- **Tab Buat Prediksi Baru**: Unggah model yang telah Anda simpan dan masukkan data pasien baru untuk mendapatkan prediksi.
""")

# --- Membuat Dua Tab Utama ---
train_tab, predict_tab = st.tabs(["**Pelatihan & Evaluasi Model**", "**Buat Prediksi Baru**"])


# =====================================================================================
# --- TAB 1: PELATIHAN & EVALUASI MODEL ---
# =====================================================================================
with train_tab:
    with st.sidebar:
        st.header("⚙️ Konfigurasi Pelatihan")
        uploaded_file = st.file_uploader("Pilih file CSV (harus ada kolom 'Status')", type="csv", key="training_uploader")
        
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            if st.session_state.target_column not in st.session_state.df.columns:
                st.error(f"Error: File CSV Anda harus memiliki kolom bernama '{st.session_state.target_column}'.")
                st.stop()
            st.success(f"Dataset dimuat. Variabel target: **{st.session_state.target_column}**")

        if st.sidebar.button("Latih Semua Model", type="primary", disabled=st.session_state.df is None):
            st.session_state.models_trained = True
            st.session_state.trained_models = {} # Reset models
            st.session_state.feature_info = None # Reset feature info

    if not st.session_state.models_trained:
        st.info("Silakan unggah dataset dan klik 'Latih Semua Model' di sidebar untuk memulai.")
        if st.session_state.df is not None:
            st.subheader("Pratinjau Data")
            st.dataframe(st.session_state.df.head())
            if 'info_message' in st.session_state:
                st.info(st.session_state.info_message)
                del st.session_state.info_message

    if st.session_state.models_trained and st.session_state.df is not None:
        df = st.session_state.df
        target_column = st.session_state.target_column
        df_for_training = df.dropna(subset=[target_column]).copy()
        
        st.subheader("📊 Perbandingan & Hasil Model")
        model_names = ['Support Vector Machine', 'Random Forest']
        results = {}

        # Simpan informasi fitur untuk digunakan di tab prediksi
        X = df_for_training.drop(target_column, axis=1)
        st.session_state.feature_info = {
            'columns': X.columns.tolist(),
            'numeric_cols': X.select_dtypes(include=np.number).columns.tolist(),
            'categorical_cols': X.select_dtypes(exclude=np.number).columns.tolist(),
            'categorical_options': {col: df[col].dropna().unique().tolist() for col in X.select_dtypes(exclude=np.number).columns}
        }
        
        model_tabs = st.tabs(model_names)

        for i, model_name in enumerate(model_names):
            with model_tabs[i]:
                best_model, accuracy, report, cm, best_params = train_and_evaluate(df_for_training, target_column, model_name)
                st.session_state.trained_models[model_name] = best_model

                st.header(f"Hasil {model_name}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Akurasi Test", f"{accuracy:.2%}")
                    
                    # --- Tombol Download Model ---
                    st.markdown("---")
                    st.write("**Simpan Model**")
                    model_buffer = io.BytesIO()
                    joblib.dump(best_model, model_buffer)
                    model_buffer.seek(0)
                    
                    st.download_button(
                        label=f"Unduh Model {model_name}",
                        data=model_buffer,
                        file_name=f"{model_name.replace(' ', '_').lower()}_model.joblib",
                        mime='application/octet-stream'
                    )
                    st.caption("Simpan model ini untuk digunakan di tab prediksi.")

                with col2:
                    st.text("Laporan Klasifikasi:")
                    st.dataframe(pd.DataFrame(report).transpose())

                st.text("Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(6, 4))
                class_labels = sorted(df_for_training[target_column].dropna().unique())
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels)
                plt.ylabel('Aktual')
                plt.xlabel('Prediksi')
                st.pyplot(fig)

# =====================================================================================
# --- TAB 2: BUAT PREDIKSI BARU ---
# =====================================================================================
with predict_tab:
    st.header("🔮 Buat Prediksi untuk Data Baru")

    if not st.session_state.feature_info:
        st.warning("Harap latih model terlebih dahulu di tab 'Pelatihan & Evaluasi Model' untuk menyiapkan form prediksi.", icon="⚠️")
    else:
        uploaded_model_file = st.file_uploader(
            "Unggah file model yang telah Anda simpan (.joblib)", 
            type="joblib",
            key="prediction_uploader"
        )
        
        if uploaded_model_file:
            try:
                model = joblib.load(uploaded_model_file)
                st.success("Model berhasil dimuat!")

                with st.form("new_prediction_form"):
                    st.write("Masukkan data pasien di bawah ini:")
                    input_data = {}
                    
                    # Buat form input secara dinamis
                    feature_info = st.session_state.feature_info
                    cols_in_row = st.columns(3)
                    
                    for i, col_name in enumerate(feature_info['columns']):
                        with cols_in_row[i % 3]:
                            if col_name in feature_info['numeric_cols']:
                                input_data[col_name] = st.number_input(
                                label=f"{col_name}",
                                value=float(df[col_name].mean()), # Ini FLOAT
                                step=0.01,                        # Ini FLOAT juga ✅
                                format="%.2f"
                                )
                            elif col_name in feature_info['categorical_cols']:
                                input_data[col_name] = st.selectbox(
                                    label=f"{col_name}",
                                    options=feature_info['categorical_options'][col_name]
                                )
                    
                    predict_button = st.form_submit_button("Prediksi Hasil Pasien", type="primary")

                if predict_button:
                    input_df = pd.DataFrame([input_data])
                    
                    with st.spinner("Membuat prediksi..."):
                        prediction = model.predict(input_df)
                        prediction_proba = model.predict_proba(input_df)

                    st.markdown("---")
                    st.subheader("✔️ Hasil Prediksi")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Status Pasien yang Diprediksi", prediction[0])
                    
                    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=['Probabilitas'])
                    with col2:
                        st.write("Probabilitas Prediksi:")
                        st.dataframe(proba_df.style.format("{:.2%}"))

            except Exception as e:
                st.error(f"Gagal memuat model. Pastikan file yang diunggah benar. Error: {e}")
        else:
            st.info("Silakan unggah model yang telah dilatih untuk memulai prediksi.")
