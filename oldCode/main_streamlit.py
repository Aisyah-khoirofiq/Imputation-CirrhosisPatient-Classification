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

# --- Page Configuration ---
st.set_page_config(
    page_title="Patient Outcome Classifier",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching for Data Loading ---
@st.cache_data
def load_data(uploaded_file):
    """Loads and caches the uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    return df

# --- Core Machine Learning Functions ---
def get_model_and_transformers():
    """Creates scikit-learn transformers."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return numeric_transformer, categorical_transformer

def get_param_grid(model_name):
    """Returns the hyperparameter grid for GridSearchCV for a given model."""
    imputer_params = {
        'preprocessor__numeric__imputer__n_neighbors': [3, 5, 7, 9]
    }
    if model_name == 'K-Nearest Neighbors (KNN)':
        classifier_params = {
            'classifier__n_neighbors': [3, 5, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
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
    return {**imputer_params, **classifier_params}

@st.cache_data(show_spinner=False)
def train_and_evaluate(_df, target_column, model_name):
    """Main function to train a single model, find best params, and evaluate it."""
    with st.spinner(f'Training {model_name} and performing GridSearchCV...'):
        X = _df.drop(target_column, axis=1)
        y = _df[target_column]

        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(exclude=np.number).columns

        numeric_transformer, categorical_transformer = get_model_and_transformers()

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ])
            
        if model_name == 'K-Nearest Neighbors (KNN)':
            classifier = KNeighborsClassifier()
        elif model_name == 'Support Vector Machine (SVM)':
            classifier = SVC(probability=True)
        else: # Random Forest
            classifier = RandomForestClassifier(random_state=42)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        param_grid = get_param_grid(model_name)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        return best_model, accuracy, report, cm, best_params

def get_imputed_dataframe(fitted_pipeline, original_df, target_column):
    """Applies the fitted preprocessor to the original data and returns a viewable DataFrame."""
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    features_df = original_df.drop(columns=[target_column])
    
    transformed_data = preprocessor.transform(features_df)
    
    numeric_features = features_df.select_dtypes(include=np.number).columns
    categorical_features = features_df.select_dtypes(exclude=np.number).columns
    
    try:
        ohe_feature_names = preprocessor.named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out(categorical_features)
    except Exception:
        ohe_feature_names = [f"cat_{i}" for i in range(transformed_data.shape[1] - len(numeric_features))]

    all_feature_names = list(numeric_features) + list(ohe_feature_names)
    
    imputed_df = pd.DataFrame(transformed_data, columns=all_feature_names, index=original_df.index)
    imputed_df[target_column] = original_df[target_column].values
    
    return imputed_df

# --- Streamlit UI Layout ---
st.title("⚕️ Interactive Patient Outcome Classifier")
st.markdown("""
This application trains and compares three classification models (KNN, SVM, Random Forest) to predict patient outcomes based on clinical data. The target variable is fixed to the **'Status'** column.

**How to use:**
1.  **Upload your data** in CSV format. Ensure it contains a `Status` column.
2.  Click the **"Train All Models"** button in the sidebar.
3.  View the comparative results in the tabs below.
4.  Use the **"Make a New Prediction"** form to predict the outcome for new data.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("### 1. Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file (must contain a 'Status' column)", type="csv")
    st.markdown("[Download Sample Cirrhosis Dataset](https://raw.githubusercontent.com/fedesoriano/cirrhosis-prediction-dataset/main/cirrhosis.csv)")
    
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.models_trained = False
    
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.models_trained = False

    if st.session_state.df is not None:
        df = st.session_state.df
        target_column = 'Status'
        if target_column not in df.columns:
            st.error(f"Error: Your uploaded CSV file must contain a '{target_column}' column.")
            st.stop()
        
        st.success(f"Target variable automatically set to: **{target_column}**")
        st.session_state.target_column = target_column

        if st.button("Train All Models", type="primary"):
            st.session_state.models_trained = True

# --- Main Panel for Displaying Results ---
if st.session_state.df is None:
    st.info("Please upload a CSV file and click 'Train All Models' to get started.")
elif not st.session_state.models_trained:
    st.info("Click the 'Train All Models' button in the sidebar to begin analysis.")
else:
    df = st.session_state.df
    target_column = st.session_state.target_column
    
    initial_rows = len(df)
    df_for_training = df.dropna(subset=[target_column])
    cleaned_rows = len(df_for_training)
    
    if initial_rows > cleaned_rows:
        st.warning(f"Your target column **'{target_column}'** contains {initial_rows - cleaned_rows} missing value(s). These rows have been excluded, leaving **{cleaned_rows}** rows for the analysis.")
    else:
        st.success(f"The target column **'{target_column}'** has no missing values. All {initial_rows} rows will be used.")

    st.subheader("📊 Model Comparison")
    model_names = ['K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)', 'Random Forest']
    results = {}

    for model_name in model_names:
        best_model, accuracy, report, cm, best_params = train_and_evaluate(df_for_training, target_column, model_name)
        results[model_name] = {'model': best_model, 'accuracy': accuracy, 'report': report, 'cm': cm, 'params': best_params}

    tab1, tab2, tab3 = st.tabs(model_names)
    
    for i, model_name in enumerate(model_names):
        with [tab1, tab2, tab3][i]:
            st.header(f"{model_name} Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Accuracy", f"{results[model_name]['accuracy']:.2%}")
                st.text("Classification Report:")
                st.dataframe(pd.DataFrame(results[model_name]['report']).transpose())
            with col2:
                st.text("Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(6, 5))
                class_labels = sorted(df_for_training[target_column].dropna().unique())
                sns.heatmap(results[model_name]['cm'], annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
            
            with st.expander("View Optimal Hyperparameters"):
                st.json(results[model_name]['params'])
            st.subheader("View Data After Imputation")
            st.info(f"Displaying the full dataset after being processed by the optimal {model_name} pipeline.")
            imputed_df = get_imputed_dataframe(results[model_name]['model'], df_for_training, target_column)
            st.dataframe(imputed_df)

    # --- Single Prediction Form (in main panel) ---
    st.markdown("---")
    st.subheader("🔮 Make a New Prediction")
    
    with st.form("prediction_form"):
        st.write("Enter the details for a single patient and choose a model for prediction.")
        
        # Add a dropdown to select the model for prediction
        selected_model_for_prediction = st.selectbox("Choose a model for prediction", options=model_names)

        input_data = {}
        feature_columns = df.drop(target_column, axis=1).columns
        
        pred_col1, pred_col2 = st.columns(2)
        
        for idx, col in enumerate(feature_columns):
            target_col = pred_col1 if idx < len(feature_columns) / 2 else pred_col2
            with target_col:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    if pd.isna(mean_val): mean_val = 0.0
                    input_data[col] = st.number_input(f"{col}", value=float(mean_val), key=f"pred_{col}")
                else:
                    options = df[col].dropna().unique().tolist()
                    if options:
                        input_data[col] = st.selectbox(f"{col}", options=options, index=0, key=f"pred_{col}")
                    else:
                        input_data[col] = st.text_input(f"{col} (no options)", "", key=f"pred_{col}")

        submitted = st.form_submit_button("Predict Patient Outcome")

        if submitted:
            prediction_data = pd.DataFrame([input_data])
            
            # Use the selected model to make a prediction
            pipeline = results[selected_model_for_prediction]['model']
            prediction = pipeline.predict(prediction_data)[0]
            prediction_proba = pipeline.predict_proba(prediction_data)[0]
            max_proba = prediction_proba.max()

            st.write("### Prediction Result:")
            result_df = pd.DataFrame([{
                'Model Used': selected_model_for_prediction,
                'Predicted Outcome': prediction,
                'Confidence': f"{max_proba:.2%}"
            }])
            st.table(result_df)

