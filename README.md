Banking Transaction Fraud Detection Using Deep Learning

Project Overview

This project develops a machine learning and deep learning-based banking transaction fraud detection system.

The solution uses an Artificial Neural Network (ANN) to predict potentially fraudulent transactions and a Streamlit dashboard to visualize transaction risk, fraud probability, and model performance.

Business Objective

To support banking fraud monitoring by identifying suspicious transactions, prioritizing investigation, and providing model explainability.

Technologies Used

Python

Pandas and NumPy

Matplotlib and Seaborn

Scikit-learn

TensorFlow / Keras

SHAP

Streamlit

Plotly

Project Workflow

Dataset collection and exploratory data analysis

Data cleaning and preprocessing

Feature engineering

Logistic Regression baseline model

Artificial Neural Network model

Model performance evaluation

Fraud prediction threshold tuning

Feature importance and SHAP explainability

Interactive Streamlit dashboard

Transaction-level fraud investigation

Key Features

Fraud probability prediction

Legitimate vs potentially fraudulent transaction classification

Transaction risk categorization

Confusion matrix and performance visualizations

Precision, Recall, F1 Score, ROC-AUC and PR-AUC

Global feature importance

SHAP-based individual transaction explanations

CSV upload and prediction report download

Dataset

Dataset: 1M Fraud-Labeled Synthetic Bank Transactions

Source: Hugging Face Dataset

A reproducible sample of 5,000 transactions was used for this project.

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER

Install dependencies:

python -m pip install -r requirements.txt

Run model training:

python train.py

Run the dashboard:

python -m streamlit run app.py

Dashboard

The Streamlit dashboard provides:

Transaction analytics

Fraud probability distribution

Transaction-level investigation

Risk categorization

SHAP feature explanations

Prediction report downloads

Model Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

PR-AUC

Confusion Matrix

Actual evaluation results should be documented after final validation on the untouched test dataset.

Project Limitations

The dataset contains synthetic banking transactions.

Predictions are model-generated risk indicators, not confirmed fraud decisions.

The project is an educational prototype and is not approved for production banking decisions.

Further validation, leakage checks, monitoring, and security controls are required before real-world deployment.

Future Enhancements

Power BI integration

Real-time fraud monitoring

FastAPI model deployment

Model drift monitoring

Advanced fraud detection algorithms

Improved transaction-level explainability
