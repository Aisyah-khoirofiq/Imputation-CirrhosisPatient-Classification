<div align="center">

🩺 Cirrhosis Patient Outcome Prediction App 🩺
An interactive web application to predict cirrhosis patient outcomes using a machine learning model built with Scikit-learn and Streamlit.

</div>

This project provides an end-to-end solution for predicting the status of patients with cirrhosis based on clinical data. The application leverages a pre-trained Random Forest model, which was determined to be the most effective after a thorough evaluation of various data imputation and modeling techniques.

✨ Live Application Preview
Below is a preview of the interactive Streamlit application. Users can input patient data on the left sidebar and instantly receive a prediction on the main screen.

(Here you can add a screenshot of your application)
[Insert Screenshot of the Streamlit App Here]

🚀 Core Features
📊 Interactive Data Input: A user-friendly form in the sidebar for entering patient clinical data.

⚡ Instant Predictions: Real-time classification of the patient's status: D (Death), C (Censored), or CL (Censored due to Liver Transplant).

📈 Model Confidence Score: A probability breakdown for each class to indicate the model's confidence in its prediction.

📋 Input Summary: A clear display of the user-entered data for verification before prediction.

🛠️ Technology & Workflow
This project follows a standard MLOps workflow, separating model training from application serving.

Workflow Diagram
+------------------+     +--------------------+     +---------------------+
| cirrhosis.csv    | --> |   train_model.py   | --> | cirrhosis_model.pkl |
| (Raw Data)       |     | (Preprocessing &   |     | (Saved Pipeline)    |
|                  |     |  Training Script)  |     |                     |
+------------------+     +--------------------+     +----------+----------+
                                                               |
                                                               v
+------------------+     +--------------------+     +----------+----------+
| Prediction       | <-- |      app.py        | <-- |   User Input        |
| (App Output)     |     | (Streamlit App)    |     | (Web Form)          |
|                  |     |                    |     |                     |
+------------------+     +--------------------+     +---------------------+

Tech Stack
Backend & ML: Python, Scikit-learn, Pandas, NumPy, Imbalanced-learn

Web Framework: Streamlit

Model Persistence: Joblib

⚙️ How to Run the Project
Follow these steps to get the application running on your local machine.

1. Setup Environment
First, ensure all project files (app.py, train_model.py, cirrhosis.csv, requirements.txt) are in the same directory. Then, install the required libraries:

# Navigate to your project folder
# cd /path/to/your/project

# Install dependencies from the requirements file
pip install -r requirements.txt

2. Train and Save the Model
Run the training script. This will process the dataset, train the best model, and save the complete pipeline as cirrhosis_model.pkl. This step only needs to be performed once.

python train_model.py

3. Launch the Streamlit App
Once the model file is created, launch the Streamlit application:

streamlit run app.py

Your default web browser will open with the app running.

🔬 Model Performance & Evaluation
The final model was selected after a rigorous evaluation documented in the Jupyter Notebook Imputation_KNN_(SVM_+_RF)_Final.ipynb.

Training Summary
Data Imputation: KNNImputer was tested with various k values (from 3 to 21).

Models Compared: Support Vector Machine (SVM) vs. Random Forest.

Evaluation Metric: Weighted F1-Score was chosen to balance precision and recall on the imbalanced dataset.

Final Model: Random Forest (k=7)
The experiments concluded that the Random Forest classifier combined with a KNN Imputer (k=7) yielded the best and most stable performance.

🏆 Best Model Performance: Random Forest (k=7)

Metric

Score

F1-Score (Weighted)

0.7381

Accuracy

0.7619

Sensitivity (Recall)

0.7619

Precision (Weighted)

0.7159

Confusion Matrices of Best Models
The confusion matrices below visualize the performance of the top-performing models on the held-out test set.

Best Random Forest Model (Imputer K=7)

           | Predicted: D | Predicted: C | Predicted: CL |
-----------------------------------------------------------
  True: D  |      25      |       5      |       0       |
  True: C  |      8       |      41      |       0       |
  True: CL |      7       |       2      |       2       |

Best SVM Model (Imputer K=17)

           | Predicted: D | Predicted: C | Predicted: CL |
-----------------------------------------------------------
  True: D  |      27      |       3      |       0       |
  True: C  |      11      |      38      |       0       |
  True: CL |      8       |       2      |       1       |

⚠️ Disclaimer
This application is an educational and demonstrational tool. The predictions are based on a machine learning model and should not be used for actual medical diagnosis or decision-making. Always consult a qualified healthcare professional for any medical concerns.