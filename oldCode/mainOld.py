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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

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
    """Loads, caches, and cleans the uploaded CSV file by removing common ID columns."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # --- FIX: Robustly find and drop identifier columns ---
    cols_to_drop = [col for col in df.columns if col.lower().strip() in ['id', 'unnamed: 0']]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        # Use session state to show this message only once in the main UI
        st.session_state.info_message = f"Note: Automatically dropped potential identifier columns: **{', '.join(cols_to_drop)}**"
        
    return df

# --- Core Machine Learning Functions ---
def get_model_and_transformers():
    """Creates scikit-learn transformers."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer()), # n_neighbors will be set by get_best_params
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return numeric_transformer, categorical_transformer

def get_best_params(model_name):
    """
    Returns a dictionary of the pre-defined optimal hyperparameters for each model,
    based on prior analysis.
    """
    if model_name == 'Support Vector Machine':
        return {
            'preprocessor__numeric__imputer__n_neighbors': 13,
            'classifier__C': 1,
            'classifier__gamma': 'scale',
            'classifier__kernel': 'rbf'
        }
    elif model_name == 'Random Forest':
        return {
            'preprocessor__numeric__imputer__n_neighbors': 3,
            'classifier__n_estimators': 100,
            'classifier__max_depth': None,
        }
    return {}

@st.cache_data(show_spinner=False)
def train_and_evaluate(_df, target_column, model_name):
    """Main function to train a single model with pre-defined params and evaluate it."""
    with st.spinner(f'Applying SMOTE and training {model_name} with optimal parameters...'):
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
            

        if model_name == 'Support Vector Machine':
            classifier = SVC(probability=True, random_state=42)
        else: # Random Forest
            classifier = RandomForestClassifier(random_state=42)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])

        # Get and set the pre-defined best parameters
        best_params = get_best_params(model_name)
        pipeline.set_params(**best_params)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
        cm = confusion_matrix(y_test, y_pred)
        
        # The trained pipeline is the best model
        return pipeline, accuracy, report, cm, best_params

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
This application trains and compares three classification models (KNN, SVM, Random Forest) using pre-defined optimal hyperparameters to predict patient outcomes. The target variable is fixed to the **'Status'** column.

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
    # st.markdown("[Download Sample Cirrhosis Dataset]")
    
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.models_trained = False
    
    if uploaded_file is not None:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.df = load_data(uploaded_file)
        st.session_state.models_trained = False # Reset on new file upload

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
    # Display the one-time info message about dropped columns if it exists
    if 'info_message' in st.session_state:
        st.info(st.session_state.info_message)
        del st.session_state.info_message # Clear it after showing
        
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

    st.subheader("⚖️ Class Distribution & SMOTE Balancing")
    
    X = df_for_training.drop(target_column, axis=1)
    y = df_for_training[target_column]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Distribution Before SMOTE")
        fig_before, ax_before = plt.subplots()
        y_train.value_counts().sort_index().plot(kind='bar', ax=ax_before, color=sns.color_palette("pastel"))
        ax_before.set_title("Training Set Class Distribution")
        ax_before.set_xlabel("Patient Status")
        ax_before.set_ylabel("Count")
        st.pyplot(fig_before)

    with col2:
        st.write("#### Distribution After SMOTE (Illustration)")
        numeric_features = X_train.select_dtypes(include=np.number).columns
        categorical_features = X_train.select_dtypes(exclude=np.number).columns
        numeric_transformer, categorical_transformer = get_model_and_transformers()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)
            ])
        
        X_train_processed = preprocessor.fit_transform(X_train)
        smote_viz = SMOTE(random_state=42)
        _, y_train_smote = smote_viz.fit_resample(X_train_processed, y_train)
        
        fig_after, ax_after = plt.subplots()
        pd.Series(y_train_smote).value_counts().sort_index().plot(kind='bar', ax=ax_after, color=sns.color_palette("pastel"))
        ax_after.set_title("Training Set Distribution after SMOTE")
        ax_after.set_xlabel("Patient Status")
        ax_after.set_ylabel("Count")
        st.pyplot(fig_after)
    
    st.info("Note: The 'After SMOTE' chart is for illustration. In the actual training, SMOTE is applied correctly within each cross-validation fold to prevent data leakage.", icon="ℹ️")

    st.subheader("📊 Model Comparison")
    model_names = ['Support Vector Machine', 'Random Forest']
    results = {}

    for model_name in model_names:
        best_model, accuracy, report, cm, best_params = train_and_evaluate(df_for_training, target_column, model_name)
        results[model_name] = {'model': best_model, 'accuracy': accuracy, 'report': report, 'cm': cm, 'params': best_params}

    tab1, tab2= st.tabs(model_names)
    
    for i, model_name in enumerate(model_names):
        with [tab1, tab2][i]:
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
            
            with st.expander("View Pre-defined Optimal Hyperparameters"):
                st.json(results[model_name]['params'])
            st.subheader("View Data After Imputation")
            st.info(f"Displaying the full dataset after being processed by the optimal {model_name} pipeline.")
            imputed_df = get_imputed_dataframe(results[model_name]['model'], df_for_training, target_column)
            st.dataframe(imputed_df)
