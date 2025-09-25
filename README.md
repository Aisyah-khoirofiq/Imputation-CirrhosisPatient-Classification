# Pengklasifikasi Hasil Akhir Pasien yang Dioptimalkan

Ini adalah aplikasi web Streamlit yang dirancang untuk melatih dan membandingkan dua model klasifikasi yang telah dioptimalkan (Support Vector Machine dan Random Forest) untuk memprediksi hasil akhir pasien berdasarkan data klinis. Aplikasi ini dirancang khusus untuk menangani tantangan umum dalam data medis, seperti nilai yang hilang (*missing values*) dan kelas yang tidak seimbang (*imbalanced classes*).

Fitur utama dari aplikasi ini adalah penggunaan pipeline `scikit-learn` dan `imblearn` yang terintegrasi untuk menerapkan alur kerja prapemrosesan dan pemodelan yang metodologis dan kuat, menggunakan parameter-parameter terbaik yang telah ditemukan dari analisis sebelumnya.

## Fitur Utama ✨

-   **Antarmuka Pengguna Interaktif**: Dibangun dengan Streamlit untuk kemudahan penggunaan.
-   **Model yang Telah Dioptimalkan**: Menggunakan konfigurasi hyperparameter terbaik yang telah ditentukan sebelumnya untuk **Support Vector Machine (SVM)** dan **Random Forest**, sehingga proses pelatihan menjadi sangat cepat.
-   **Penanganan *Missing Value* Otomatis**:
    * Menggunakan **KNN Imputer** untuk mengisi nilai yang hilang pada fitur numerik.
    * Menggunakan **Simple Imputer** untuk fitur kategorikal.
-   **Penanganan Kelas Tidak Seimbang**: Mengintegrasikan **SMOTE (Synthetic Minority Over-sampling Technique)** ke dalam pipeline pelatihan untuk menyeimbangkan set data latih dan meningkatkan kemampuan model dalam memprediksi kelas minoritas.
-   **Pembersihan Data Otomatis**: Secara otomatis mendeteksi dan menghapus kolom identifier umum (seperti 'ID', 'id', 'Unnamed: 0') saat data diunggah.
-   **Visualisasi Komparatif**: Menampilkan hasil dari kedua model dalam *tab* terpisah untuk perbandingan yang mudah, termasuk:
    * Akurasi pada data uji.
    * Laporan Klasifikasi (Presisi, Recall, F1-Score).
    * Visualisasi *Confusion Matrix*.
    * Grafik distribusi kelas sebelum dan sesudah SMOTE.
-   **Prediksi Tunggal**: Sebuah formulir memungkinkan pengguna untuk memasukkan data pasien baru dan mendapatkan prediksi hasil akhir dari model yang dipilih.

## Cara Menjalankan Aplikasi

### 1. Prasyarat

-   Python 3.7+
-   `pip` (Manajer paket Python)

### 2. Pengaturan

1.  **Unduh file proyek.**
    Letakkan `main.py`, `requirements.txt`, dan `packages.txt` di dalam direktori yang sama.

2.  **Buat lingkungan virtual (direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Pada Windows, gunakan `venv\Scripts\activate`
    ```

3.  **Instal pustaka yang diperlukan:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Menjalankan Aplikasi

1.  **Buka terminal** dan navigasikan ke direktori tempat Anda menyimpan file-file aplikasi.

2.  **Jalankan perintah Streamlit berikut:**
    ```bash
    streamlit run app.py
    ```

3.  Browser web Anda akan otomatis terbuka dengan aplikasi yang berjalan. Jika tidak, buka browser Anda dan kunjungi `http://localhost:8501`.

## Cara Kerja Aplikasi

1.  **Pemuatan Data**: Pengguna mengunggah file CSV. Aplikasi akan memuat data, membersihkan nama kolom, dan secara otomatis menghapus kolom identifier yang tidak relevan.
2.  **Pra-pemrosesan**: Saat tombol "Train Optimized Models" ditekan, data yang targetnya (`Status`) tidak hilang akan dibagi menjadi set latih dan uji.
3.  **Pipeline Pelatihan**: Untuk setiap model (SVM dan RF), sebuah pipeline `imblearn` yang cerdas akan dijalankan **hanya pada data latih**. Pipeline ini melakukan langkah-langkah berikut secara berurutan:
    * **Prapemrosesan Fitur**: Mengisi nilai yang hilang, melakukan penskalaan pada data numerik, dan *one-hot encoding* pada data kategorikal.
    * **SMOTE**: Menyeimbangkan data latih yang telah diproses.
    * **Pelatihan Classifier**: Melatih model SVM atau Random Forest menggunakan hyperparameter optimal yang telah ditentukan sebelumnya.
4.  **Evaluasi**: Model yang telah dilatih kemudian digunakan untuk membuat prediksi pada **data uji yang asli (tidak seimbang)**. Metrik kinerja dihitung berdasarkan prediksi ini.
5.  **Tampilan**: Hasil dari setiap model ditampilkan dalam *tab* terpisah, dan formulir prediksi tunggal menjadi tersedia untuk digunakan.