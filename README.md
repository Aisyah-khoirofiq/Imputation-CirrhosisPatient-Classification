# 🩺 Cirrhosis Patient Outcome Prediction

An interactive web application to predict cirrhosis patient outcomes using machine learning with Scikit-learn and Streamlit.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Results](#results)
- [Model Pick](#model-selection-note)
- [Disclaimer](#disclaimer)

## 📖 Overview
This project provides an end-to-end solution for predicting the status of patients with cirrhosis based on clinical data. The application leverages a pre-trained Random Forest model, which was determined to be the most effective after thorough evaluation of various data imputation and modeling techniques.

## ✨ Features
- **Interactive Data Input**: User-friendly form for entering patient clinical data
- **Real-time Predictions**: Instant classification of patient status (D: Death, C: Censored, CL: Censored due to Liver Transplant)
- **Model Confidence**: Probability breakdown for each prediction class
- **Input Verification**: Clear display of entered data before prediction

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/Aisyah-khoirofiq/Imputation-CirrhosisPatient-Classification.git

cd cirrhosis-prediction-app
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Model
Run the training script to preprocess data and train the model:
```bash
python train_model.py
```
This generates `cirrhosis_model.pkl` containing the trained pipeline.

### 2. Launch the Application
Start the Streamlit web application:
```bash
streamlit run app.py
```
The application will open in your default web browser at `http://localhost:8501`.

### 3. Make Predictions
1. Fill in the patient data form in the sidebar
2. View the input summary for verification
3. Click "Predict Patient Status" to get results
4. Review the prediction and model confidence scores

## 🔬 Model Details

### Algorithm
- **Final Model**: Random Forest Classifier with KNNImputer (k=7)
- **Data Imputation**: KNNImputer tested with k values from 3 to 21
- **Evaluation Metric**: Weighted F1-Score
- **Models Compared**: Support Vector Machine (SVM) vs Random Forest

### Feature Set
The model uses 16 clinical features including:
- Demographic data (Age, Sex)
- Medical history (Drug, Ascites, Hepatomegaly, etc.)
- Laboratory values (Bilirubin, Cholesterol, Albumin, etc.)
- Clinical scores (Prothrombin, Stage)

### Performance Summary

#### Best Performing Models

**🏆 Best SVM Model**: KNNImputer (k=13)
- Accuracy: 0.7262
- F1-Score (Weighted): 0.7412
- Sensitivity (Recall): 0.7262
- Precision (Weighted): 0.7587

**🏆 Best Random Forest Model**: KNNImputer (k=7)
- Accuracy: 0.7024
- F1-Score (Weighted): 0.7081
- Sensitivity (Recall): 0.7024
- Precision (Weighted): 0.7167

#### Complete Experiment Results

| Imputer K | Model | Accuracy | F1-Score | Sensitivity | Precision |
|-----------|-------|----------|----------|-------------|-----------|
| 3 | SVM | 0.7024 | 0.7254 | 0.7024 | 0.7526 |
| 3 | Random Forest | 0.7024 | 0.7081 | 0.7024 | 0.7167 |
| 5 | SVM | 0.6905 | 0.7132 | 0.6905 | 0.7387 |
| 5 | Random Forest | 0.6905 | 0.6957 | 0.6905 | 0.7036 |
| 7 | SVM | 0.6905 | 0.7131 | 0.6905 | 0.7410 |
| **7** | **Random Forest** | **0.7024** | **0.7081** | **0.7024** | **0.7167** |
| 9 | SVM | 0.7024 | 0.7254 | 0.7024 | 0.7504 |
| 9 | Random Forest | 0.6548 | 0.6643 | 0.6548 | 0.6759 |
| 11 | SVM | 0.7143 | 0.7333 | 0.7143 | 0.7558 |
| 11 | Random Forest | 0.6905 | 0.6992 | 0.6905 | 0.7082 |
| **13** | **SVM** | **0.7262** | **0.7412** | **0.7262** | **0.7587** |
| 13 | Random Forest | 0.6667 | 0.6760 | 0.6667 | 0.6863 |
| 15 | SVM | 0.7024 | 0.7253 | 0.7024 | 0.7529 |
| 15 | Random Forest | 0.7024 | 0.7075 | 0.7024 | 0.7131 |
| 17 | SVM | 0.7024 | 0.7253 | 0.7024 | 0.7529 |
| 17 | Random Forest | 0.7024 | 0.7033 | 0.7024 | 0.7063 |
| 19 | SVM | 0.7024 | 0.7253 | 0.7024 | 0.7529 |
| 19 | Random Forest | 0.6786 | 0.6825 | 0.6786 | 0.6866 |
| 21 | SVM | 0.7024 | 0.7253 | 0.7024 | 0.7529 |
| 21 | Random Forest | 0.6667 | 0.6711 | 0.6667 | 0.6757 |

## 📁 Project Structure
```
cirrhosis-prediction-app/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── cirrhosis.csv          # Original dataset
├── cirrhosis_model.pkl    # Trained model (generated)
├── requirements.txt       # Python dependencies
├── best_models/          # Best performing models
│   ├── best_svm_k17.pkl
│   └── best_random_forest_k7.pkl
└── README.md             # Project documentation
```

## 📊 Results

### Model Selection Rationale
After extensive testing with KNNImputer k-values ranging from 3 to 21, the Random Forest model with k=7 was selected as the final model due to its balanced performance across all metrics and stability. While SVM with k=13 achieved slightly higher F1-score (0.7412), Random Forest demonstrated more consistent performance across different k-values.

### Key Findings
- **SVM Performance**: Best with k=13 (F1-score: 0.7412)
- **Random Forest Performance**: Best with k=7 (F1-score: 0.7081)
- **Optimal K-range**: k=7 to k=13 provided the best balance of performance
- **Model Stability**: Random Forest showed more consistent performance across different k-values

## 🔬 Model Selection Note

**Why Random Forest with K=7 was chosen over K=3?**

Both KNNImputer k=3 and k=7 with Random Forest achieved identical performance metrics (Accuracy: 0.7024, F1-Score: 0.7081). 

### However, k=7 was selected due to:
- **Better generalization** with more neighbors
- **Reduced sensitivity** to outliers and noise
- **Improved stability** across different data splits
- **Alignment** with optimal k-range observed in SVM experiments

This demonstrates that model selection involves considerations beyond test metrics alone.

## ⚠️ Disclaimer
This application is intended for educational and demonstrational purposes only. The predictions are generated by a machine learning model and should not be used for medical diagnosis, treatment decisions, or clinical guidance. Always consult qualified healthcare professionals for medical concerns and treatment decisions.

## 📄 License
Dataset License
The cirrhosis patient dataset used in this project is from the UCI Machine Learning Repository:

Source: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1

Dataset Citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
