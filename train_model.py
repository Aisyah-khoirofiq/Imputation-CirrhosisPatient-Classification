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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_and_save_model(data_path='cirrhosis.csv', model_output_path='cirrhosis_model.pkl'):
    """
    Loads data, trains the best model pipeline, and saves it to a file.
    """
    print("🚀 Starting model training process...")

    # --- 1. Data Loading and Initial Setup ---
    try:
        df = pd.read_csv(data_path)
        print("✅ Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: '{data_path}' not found. Please ensure the file is in the correct directory.")
        return

    # Drop unnecessary ID column and handle target variable
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    TARGET = 'Status'
    status_map = {'D': 0, 'C': 1, 'CL': 2}
    df[TARGET] = df[TARGET].map(status_map)
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)

    print(f"🎯 Target variable '{TARGET}' is ready.")

    # --- 2. Feature Preprocessing ---
    # Convert categorical columns to numeric using LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        # Fill missing categorical values with the mode before encoding
        df[col].fillna(df[col].mode()[0], inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        print(f"   - Encoded categorical column: {col}")


    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # --- 3. Define the Best Model Pipeline ---
    # Based on your notebook, Random Forest with KNNImputer(k=7) was a top performer.
    # We will build a full pipeline to handle all steps: scaling, imputation, SMOTE, and classification.

    # We need to separate numerical columns for scaling and imputation
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Create separate pipelines for preprocessing numerical and categorical data
    # In this case, all categorical data was already label encoded and filled,
    # but a full pipeline would handle it more robustly. We will focus on the numerical pipeline.

    # We will use ImbPipeline from imblearn to correctly handle SMOTE
    # SMOTE should only be applied to the training data.
    # The pipeline ensures this by applying it after the train-test split internally during cross-validation,
    # but for final model saving, we'll apply it before fitting the classifier.

    # First, let's resample the data with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("⚖️ Applied SMOTE for handling class imbalance.")


    # Define the preprocessing and model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=7)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42))
    ])

    # --- 4. Train the Pipeline on the Full Resampled Dataset ---
    print(f"💪 Training RandomForestClassifier with KNNImputer (k=7) on the full dataset...")
    pipeline.fit(X_resampled, y_resampled)
    print("✅ Model training complete.")

    # --- 5. Save the Pipeline ---
    joblib.dump(pipeline, model_output_path)
    print(f"💾 Model pipeline saved to '{model_output_path}'")
    print("🎉 Process finished successfully.")


if __name__ == '__main__':
    # This block will run when the script is executed directly
    train_and_save_model()
