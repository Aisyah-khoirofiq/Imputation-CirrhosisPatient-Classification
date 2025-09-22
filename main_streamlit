import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import hashlib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Chronic Disease Classifier",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Efficiency ---

@st.cache_data
def get_dataframe_hash(df):
    """Generates a hash for a pandas DataFrame to use as a cache key."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def save_model_and_params(model, params, model_name, df_hash):
    """Saves the trained model and its best parameters."""
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, f'models/{model_name}_{df_hash}.joblib')
    joblib.dump(params, f'models/{model_name}_{df_hash}_params.joblib')

def load_model_and_params(model_name, df_hash):
    """Loads a cached model and its parameters if they exist."""
    model_path = f'models/{model_name}_{df_hash}.joblib'
    params_path = f'models/{model_name}_{df_hash}_params.joblib'
    if os.path.exists(model_path) and os.path.exists(params_path):
        model = joblib.load(model_path)
        params = joblib.load(params_path)
        return model, params
    return None, None

# --- Core Machine Learning Functions ---

def get_model_pipeline(model_name):
    """Creates a scikit-learn pipeline for the selected model."""
    
    # Preprocessing for numerical data: impute then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)), # Placeholder, will be tuned
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data: impute then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Model selection
    if model_name == 'K-Nearest Neighbors (KNN)':
        classifier = KNeighborsClassifier()
    elif model_name == 'Support Vector Machine (SVM)':
        classifier = SVC(probability=True) # probability=True for predict_proba
    elif model_name == 'Random Forest':
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Unknown model selected")
        
    return classifier, numeric_transformer, categorical_transformer

def get_param_grid(model_name):
    """Returns the hyperparameter grid for GridSearchCV for a given model."""
    
    # Common imputer parameters to search
    imputer_params = {
        'preprocessor__numeric__imputer__n_neighbors': [3, 5, 7, 9]
    }

    # Model-specific parameters
    if model_name == 'K-Nearest Neighbors (KNN)':
        classifier_params = {
            'classifier__n_neighbors': [3, 5, 9, 11],
            'classifier__weights': ['uniform', 'distance']
        }
    elif model_name == 'Support Vector Machine (SVM)':
        classifier_params = {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': [0.1, 0.01, 'scale'],
            'classifier__kernel': ['rbf']
        }
    elif model_name == 'Random Forest':
        classifier_params = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
    
    # Combine imputer and classifier parameters
    return {**imputer_params, **classifier_params}

def train_and_evaluate(df, target_column, model_name):
    """Main function to train a model, find best params, and evaluate it."""
    
    # Check if a cached model exists for the current data and model type
    df_hash = get_dataframe_hash(df)
    cached_model, cached_params = load_model_and_params(model_name, df_hash)

    if cached_model and cached_params:
        st.info(f"Loading cached model for {model_name}.")
        st.session_state.trained_pipeline = cached_model
        st.session_state.best_params = cached_params
        
        # We still need to evaluate it on a new train/test split to show metrics
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        y_pred = cached_model.predict(X_test)
        
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred), cached_params

    # If no cached model, train from scratch
    with st.spinner(f'Training {model_name} and performing GridSearchCV... This may take a moment.'):
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Identify column types
        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(exclude=np.number).columns

        # Get the pipeline components
        classifier, numeric_transformer, categorical_transformer = get_model_pipeline(model_name)
        
        # Create the full preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline with preprocessor and classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        # Get parameter grid and run GridSearchCV
        param_grid = get_param_grid(model_name)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred = best_model.predict(X_test)
        
        # Save the newly trained model and params to cache
        save_model_and_params(best_model, best_params, model_name, df_hash)
        
        # Store in session state for single prediction use
        st.session_state.trained_pipeline = best_model
        st.session_state.best_params = best_params

        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred), best_params

# --- Streamlit UI Layout ---

st.title("⚕️ Interactive Chronic Disease Classifier")
st.markdown("""
This application allows you to train and evaluate three different classification models (KNN, SVM, Random Forest) on a dataset for predicting chronic disease status. 
The app handles missing values by finding the optimal imputation parameters simultaneously with the model's hyperparameters using `GridSearchCV`.

**How to use:**
1.  **Upload your data** in CSV format using the sidebar. A sample dataset is available for download.
2.  Select a **classification model** from the dropdown.
3.  The model will be trained, and the **results will be displayed**.
4.  Use the **"Single Prediction"** form in the sidebar to predict the outcome for a new data point.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("### 1. Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.markdown("[Download Sample Cirrhosis Dataset](https://raw.githubusercontent.com/fedesoriano/cirrhosis-prediction-dataset/main/cirrhosis.csv)")
    
    # Model and target selection appears only after file upload
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Clean up column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        st.session_state.df = df
        st.session_state.target_options = df.columns.tolist()
        
        st.markdown("### 2. Select Target & Model")
        target_column = st.selectbox(
            "Choose the target variable (what you want to predict):",
            st.session_state.target_options,
            index=len(st.session_state.target_options) - 1 # Default to the last column
        )
        st.session_state.target_column = target_column
        
        model_name = st.selectbox(
            "Choose the classification model:",
            ('K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)', 'Random Forest')
        )
        st.session_state.model_name = model_name
    
    # Single prediction form
    if 'df' in st.session_state:
        st.markdown("---")
        st.header("🔮 Single Prediction")
        
        input_data = {}
        # Dynamically create input fields based on dataframe columns
        for col in st.session_state.df.drop(st.session_state.target_column, axis=1).columns:
            if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                min_val = float(st.session_state.df[col].min())
                max_val = float(st.session_state.df[col].max())
                mean_val = float(st.session_state.df[col].mean())
                input_data[col] = st.number_input(f"Enter value for {col}", min_value=min_val, max_value=max_val, value=mean_val, step=0.1)
            else:
                options = st.session_state.df[col].unique().tolist()
                # Handle potential NaN values in options
                clean_options = [opt for opt in options if pd.notna(opt)]
                input_data[col] = st.selectbox(f"Select value for {col}", options=clean_options)

        if st.button("Predict"):
            st.session_state.single_prediction_data = pd.DataFrame([input_data])
            st.session_state.run_single_prediction = True

# --- Main Panel for Displaying Results ---
if uploaded_file is None:
    st.info("Please upload a CSV file to get started.")
else:
    # Display data preview
    st.subheader("📄 Data Preview")
    st.dataframe(st.session_state.df.head())
    
    # Display missing value info
    st.subheader("❓ Missing Values Overview")
    missing_counts = st.session_state.df.isnull().sum()
    missing_df = missing_counts[missing_counts > 0].reset_index()
    missing_df.columns = ['Feature', 'Missing Count']
    if not missing_df.empty:
        st.write("The following features have missing values:")
        st.table(missing_df)
    else:
        st.success("No missing values detected in the dataset.")
        
    st.subheader(f"📊 Model Results: {st.session_state.model_name}")

    # Train and evaluate the model
    accuracy, report, cm, best_params = train_and_evaluate(st.session_state.df, st.session_state.target_column, st.session_state.model_name)

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{accuracy:.2%}")
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.text("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=st.session_state.df[st.session_state.target_column].unique(), 
                    yticklabels=st.session_state.df[st.session_state.target_column].unique())
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    st.subheader("🛠️ Optimal Hyperparameters Found")
    st.write("The best parameters found by GridSearchCV (including for the KNN imputer) are:")
    st.json(best_params)

    # --- Handle Single Prediction Display ---
    if st.session_state.get('run_single_prediction', False):
        if 'trained_pipeline' in st.session_state:
            pipeline = st.session_state.trained_pipeline
            prediction_data = st.session_state.single_prediction_data
            
            # Predict
            prediction = pipeline.predict(prediction_data)[0]
            prediction_proba = pipeline.predict_proba(prediction_data)[0]
            
            st.subheader("🔮 Single Prediction Result")
            st.success(f"**Predicted Class:** {prediction}")
            
            # Display probabilities
            proba_df = pd.DataFrame({
                'Class': pipeline.classes_,
                'Probability': prediction_proba
            })
            st.write("Prediction Probabilities:")
            st.dataframe(proba_df)

        else:
            st.error("Model is not trained yet. Please wait for training to complete.")
        
        # Reset the flag
        st.session_state.run_single_prediction = False
