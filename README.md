# 1. Project Overview

Financial institutions process thousands of transactions every day across domestic and international payment channels. Identifying suspicious transactions quickly while minimizing false alerts is a major challenge for banking fraud monitoring teams.
This project develops an AI-powered Banking Transaction Fraud Detection and Risk Monitoring solution using Machine Learning and Deep Learning.
The solution combines transaction-level data analysis, engineered banking risk indicators, an Artificial Neural Network (ANN), and SHAP explainability to identify potentially fraudulent transactions.
An interactive Streamlit dashboard provides transaction monitoring, fraud probability, risk categorization, and investigation support.

**Business focus: Transform transaction data into actionable fraud risk insights to support faster and more informed investigation.**


# 2. Business Objectives & Study Areas

1.Identify potentially fraudulent banking transactions.

2.Analyze transaction behavior and suspicious activity patterns.

3.Improve fraud detection through machine learning and deep learning.

4.Balance fraud detection recall with unnecessary false-positive alerts.

5.Explain model predictions to support analyst investigation.

6.Build an interactive monitoring dashboard for operational visibility.

Banking and analytical study areas

Business area                            Analytical approach

Transaction Monitoring -------->>> Transaction behaviour and velocity analysis

Payment Risk -------->>> Domestic and international transaction indicators

Customer Behavior -------->>> Customer segment and account characteristics

Device Risk -------->>>  New device and trusted device indicators

Financial Risk -------->>> Transaction amount and balance movement

Fraud Investigation -------->>> Prediction probability and SHAP explanations.


# 3. Dataset & Technology Stack

1M Fraud-Labeled Synthetic Bank Transactions

Source: Hugging Face Dataset

Project sample: 5,000 transactions.

Target variable: fraud_label.

0 – Legitimate transaction.
1 – Fraudulent transaction.

The dataset includes transaction amounts, account balances, transaction timing, payment channels, customer segments, device information, and risk-related indicators.

**Technology stack**

Technology                               Purpose

Python                 ------->>   Core programming

Pandas & NumPy         ------->>   Data manipulation

Matplotlib & Seaborn   ------->>   Exploratory visualization

Scikit-learn           ------->>   Preprocessing and baseline ML

TensorFlow / Keras     ------->>   Deep Learning ANN

SHAP                   ------->>   Model Explainability

Streamlit              ------->>   Interactive dashboard

Plotly                 ------->>   Interactive charts

Joblib                 ------->>   Model artifact persistence


# 4. Methodology

**Phase 1 – Exploratory Data Analysis (EDA)**

1.Dataset structure and data quality analysis.

2.Missing values and duplicate identification.

3.Fraud versus legitimate transaction distribution.

4.Transaction amount distribution.

5.Comparison of transaction behavior across fraud labels.

**Phase 2 – Feature Engineering**

Developed additional features to capture banking transaction risk:

**Engineered feature                        Business interpretation**

balance_change            --------->>  Change in account balance after transaction

amount_balance_ratio      --------->>  Transaction amount relative to previous balance

high_value_transaction    --------->>  Identifies transactions above a selected amount threshold

risk_indicator            --------->>  Aggregated count of selected transaction risk flags

**Phase 3 – Data Preprocessing**

1.Numerical feature imputation.

2.Standard scaling.

3.Categorical feature imputation.

4.One-hot encoding.

5.Stratified training, validation and test split.

6.Class imbalance handling through class weights during ANN training.

**Phase 4 – Model Development**

Two classification approaches were implemented:

**Model 1: Logistic Regression**
> Baseline classification model.

> Class-weight balancing.

> Probability-based prediction.

> Performance comparison.

**Model 2: Artificial Neural Network (ANN)**

Architecture:

> Input layer.

> Dense layer – 128 neurons, ReLU.

> Dropout – 30%.

> Dense layer – 64 neurons, ReLU.

> Dropout – 20%.

> Dense layer – 32 neurons, ReLU.

> Output layer – 1 neuron, Sigmoid.

> Training configuration:

> Optimizer: Adam.

> Loss: Binary Cross-Entropy.

> Early stopping.

> Class weights for imbalanced classification.

> Validation monitoring using PR-AUC.


# 5. Model Evaluation & Prediction Analysis

**Metric                     Business interpretation**

Accuracy     ------>>    Overall classification correctness

Precision    ------>>    Reliability of fraud alerts

Recall       ------>>    Ability to identify actual fraud

F1 Score     ------>>    Balance between precision and recall

ROC-AUC      ------>>    Overall discrimination capability

PR-AUC       ------>>    Fraud detection performance under class imbalance

**Fraud threshold optimization**
Instead of relying only on the default 0.50 probability threshold, a validation-based threshold selection process was implemented using the F1 score.
This helps balance fraud detection and false-positive alerts.

**Visual performance analysis**

1.Fraud distribution analysis.

2.Transaction amount comparison.

3.ANN training accuracy and loss.

4.Validation precision and recall.

5.ROC-AUC and PR-AUC trends.

6.Confusion matrix.

7.Precision-recall threshold analysis.

Model results: Actual test-set metric values should be added here after final validation. No performance values are assumed or fabricated.


# 6. Explainable AI – SHAP Analysis

Model explainability is important in banking because analysts need to understand which transaction characteristics influence a fraud prediction.

SHAP was implemented to provide:

> Global explainability
> Overall feature importance.
> Feature contribution analysis.
> Identification of influential transaction characteristics.
> Individual transaction explainability
> Transaction-specific feature contributions.
> Positive SHAP values that push predictions towards fraud.
> Negative SHAP values that push predictions away from fraud.

This provides additional analytical context for fraud investigation.


# 7. Interactive Fraud Monitoring Dashboard

The Streamlit dashboard transforms model outputs into an operational monitoring interface.

Dashboard capabilities

> Transaction analytics and KPI monitoring.

> Fraud versus legitimate transaction distribution.

> Fraud probability visualization.

> Transaction amount analysis.

> Low, Medium and High risk categorization.

> Transaction-level investigation panel.

> Individual transaction SHAP explanation.

> CSV upload and prediction report download.

> Dashboard : http://localhost:8501/

<img width="1917" height="911" alt="image" src="https://github.com/user-attachments/assets/7a845dbc-4ff5-4a83-8503-56b7f6019128" />

<img width="1621" height="873" alt="image" src="https://github.com/user-attachments/assets/f2ce9ec5-3b76-47be-b305-d4ea9b6b408a" />

<img width="1602" height="603" alt="image" src="https://github.com/user-attachments/assets/9e43679d-e1a8-4107-8fe8-dcf582ca65e8" />






# 8. Business Solution & Operational Value

The proposed solution demonstrates how banking transaction data can be transformed into a fraud risk monitoring workflow.

Business challenge                                 Proposed solution

Large transaction volumes        ---------->    Automated transaction scoring

Suspicious transaction patterns  ---------->    Engineered risk indicators

Fraud detection prioritization   ---------->    Probability-based risk categorization

Excessive false alerts           ---------->    Validation-based threshold tuning

Limited prediction transparency  ---------->    SHAP Explainability

Manual reporting                  ---------->   Interactive dashboard and CSV exports.


# 9. Future Business Enhancements

**Enhancement                                      Expected business value**

Real-time transaction scoring   ---------> Faster suspicious transaction identification

FastAPI deployment              ---------> Integration with banking applications

Power BI integration            ---------> Management-level fraud reporting

Advanced ensemble models        ---------> Model performance comparison

Graph-based fraud analytics     ---------> Identification of connected suspicious accounts

Model drift monitoring          ---------> Detection of changing fraud patterns

Feedback-based retraining      ---------> Continuous model improvement

Case management integration     ---------> Structured fraud investigation workflow.

