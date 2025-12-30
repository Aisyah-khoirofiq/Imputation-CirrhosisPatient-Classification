import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings

# Supress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_and_save_model(data_path='cirrhosis.csv', model_output_path='cirrhosis_model.pkl'):
    """
    Melatih model klasifikasi dengan urutan preprocessing:
    1. Imputer (KNNImputer)
    2. SMOTE (menyeimbangkan kelas)
    3. Scaler (StandardScaler)
    4. Klasifikasi (ModelClasification)
    """
    print("🚀 Memulai proses pelatihan model...")

    # --- 1. Muat Data ---
    try:
        df = pd.read_csv(data_path)
        print("✅ Dataset berhasil dimuat.")
    except FileNotFoundError:
        print(f"❌ File '{data_path}' tidak ditemukan. Pastikan berada di direktori yang sama.")
        return

    # Hapus kolom ID jika ada
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # --- 2. Penyiapan Target ---
    TARGET = 'Status'
    status_map = {'D': 0, 'C': 1, 'CL': 2}
    df[TARGET] = df[TARGET].map(status_map)
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)
    print(f"🎯 Variabel target '{TARGET}' siap digunakan.")

    # --- 3. Encoding Fitur Kategorikal ---
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"   - Kolom kategorikal '{col}' telah di-encode.")

    # --- 4. Pisahkan Fitur dan Target ---
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # --- 5. Pipeline dengan Urutan yang Benar ---
    # KNNImputer → SMOTE → StandardScaler → Klasifikasi Model
    pipeline = ImbPipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=7)),
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42
        ))
    ])

    # --- 6. Latih Model ---
    pipeline.fit(X, y)
    print("✅ Proses pelatihan selesai.")

    # --- 7. Simpan Model ---
    joblib.dump(pipeline, model_output_path)
    print(f"💾 Model disimpan ke '{model_output_path}'")

    # --- 8. Tampilkan Urutan Fitur ---
    if hasattr(pipeline, 'feature_names_in_'):
        print("📋 Urutan fitur dalam model:")
        print(pipeline.feature_names_in_)
    else:
        print("⚠️ Model tidak memiliki atribut feature_names_in_ (gunakan sklearn >= 1.0).")

    print("🎉 Proses selesai dengan sukses.")


if __name__ == '__main__':
    train_and_save_model()
