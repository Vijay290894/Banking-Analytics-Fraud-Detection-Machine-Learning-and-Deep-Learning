import os
from pathlib import Path
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# TASK 2: CREATE PROJECT FOLDERS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ==========================================
# TASK 3: CHECK PYTHON AND LIBRARIES
# ==========================================

print("Banking Fraud Detection Project")
print("--------------------------------")

print("Pandas Version:", pd.__version__)
print("NumPy Version:", np.__version__)
print("TensorFlow Version:", tf.__version__)

print("\nProject folders created successfully!")

print("Data folder:", DATA_DIR)
print("Artifacts folder:", ARTIFACT_DIR)
print("Reports folder:", REPORT_DIR)

print("\nTASK 1 COMPLETED SUCCESSFULLY!")


from datasets import load_dataset

print("\nDownloading Banking Fraud Dataset...")

# Download dataset from Hugging Face
dataset = load_dataset(
    "Abdulmajeedyahya/1M-Fraud-Labeled-Synthetic-Bank-Transactions"
)

# Select the training dataset split
if "train" in dataset:
    hf_data = dataset["train"]
else:
    hf_data = dataset[list(dataset.keys())[0]]

print("\nDataset downloaded successfully!")
print("Total available records:", len(hf_data))
print("Dataset columns:", hf_data.column_names)

# Select 5,000 random transactions
sample_size = min(5000, len(hf_data))

sample_data = hf_data.shuffle(seed=42).select(
    range(sample_size)
)

# Convert into Pandas DataFrame
df = sample_data.to_pandas()

# Save dataset in the data folder
csv_path = DATA_DIR / "banking_fraud_data.csv"

df.to_csv(csv_path, index=False)

print("\nDataset saved successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five transactions:")
print(df.head())

print("\nTASK 2 COMPLETED SUCCESSFULLY!")


# TASK 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

print("\n========== TASK 3: EDA ==========")

# 1. Check dataset information
print("\nDataset Information:")
print("Total transactions:", len(df))
print("Total columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 3. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 4. Fraud and legitimate transaction count
print("\nFraud Distribution:")

fraud_counts = df["fraud_label"].value_counts().sort_index()

print(fraud_counts)

# 5. Calculate fraud percentage
fraud_percentage = (
    df["fraud_label"].mean() * 100
)

print("\nFraud Percentage:",
      round(fraud_percentage, 2), "%")

legitimate_percentage = 100 - fraud_percentage

print("Legitimate Percentage:",
      round(legitimate_percentage, 2), "%")

# ==========================================
# VISUALIZATION 1: FRAUD DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="fraud_label"
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_distribution.png"
)

plt.show()

# ==========================================
# VISUALIZATION 2: FRAUD PERCENTAGE
# ==========================================

plt.figure(figsize=(6, 6))

plt.pie(
    [legitimate_percentage, fraud_percentage],
    labels=["Legitimate", "Fraud"],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Transaction Fraud Percentage")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_percentage.png"
)

plt.show()

# ==========================================
# VISUALIZATION 3: TRANSACTION AMOUNT
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="amount",
    bins=40,
    kde=True
)

plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "transaction_amount.png"
)

plt.show()

# ==========================================
# VISUALIZATION 4: FRAUD BY AMOUNT
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_label",
    y="amount"
)

plt.title("Transaction Amount: Fraud vs Legitimate")
plt.xlabel("0 = Legitimate, 1 = Fraud")
plt.ylabel("Transaction Amount")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_amount_comparison.png"
)

plt.show()

# ==========================================
# TASK 3 SUMMARY
# ==========================================

print("\n========== EDA SUMMARY ==========")

print("Total Transactions:", len(df))
print("Fraud Transactions:",
      int((df["fraud_label"] == 1).sum()))

print("Legitimate Transactions:",
      int((df["fraud_label"] == 0).sum()))

print("Fraud Percentage:",
      round(fraud_percentage, 2), "%")

print("\nEDA charts saved in reports folder.")

print("\nTASK 3 COMPLETED SUCCESSFULLY!")


# TASK 4: FEATURE ENGINEERING
# ==========================================

print("\n========== TASK 4: FEATURE ENGINEERING ==========")

# Create a copy of the original dataset
df = df.copy()

# 1. Balance Change
df["balance_change"] = (
    df["new_balance"] - df["old_balance"]
)

# 2. Transaction Amount to Balance Ratio
df["amount_balance_ratio"] = (
    df["amount"] / (df["old_balance"].abs() + 1)
)

# 3. High Value Transaction
# Use 95th percentile as the threshold
amount_threshold = df["amount"].quantile(0.95)

df["high_value_transaction"] = (
    df["amount"] >= amount_threshold
).astype(int)

# 4. Risk Indicator
risk_columns = [
    "is_international",
    "new_device_flag",
    "is_weekend",
    "is_holiday",
    "is_round_amount"
]

# Check whether all required columns exist
missing_columns = [
    col for col in risk_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns for risk indicator: {missing_columns}"
    )

df["risk_indicator"] = df[risk_columns].fillna(0).sum(axis=1)

# ==========================================
# FEATURE ENGINEERING RESULTS
# ==========================================

print("\nNew Features Created:")

new_features = [
    "balance_change",
    "amount_balance_ratio",
    "high_value_transaction",
    "risk_indicator"
]

print(df[new_features].head())

print("\nHigh Value Transaction Count:")
print(df["high_value_transaction"].value_counts())

print("\nRisk Indicator Distribution:")
print(df["risk_indicator"].value_counts().sort_index())

# ==========================================
# VISUALIZATION 1: RISK INDICATOR
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="risk_indicator",
    hue="fraud_label"
)

plt.title("Risk Indicators vs Fraud Transactions")
plt.xlabel("Number of Risk Indicators")
plt.ylabel("Transaction Count")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "risk_indicator_analysis.png"
)

plt.show()

# ==========================================
# VISUALIZATION 2: HIGH VALUE TRANSACTIONS
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="high_value_transaction",
    hue="fraud_label"
)

plt.title("High Value Transactions vs Fraud")
plt.xlabel("High Value (0 = No, 1 = Yes)")
plt.ylabel("Transaction Count")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "high_value_fraud_analysis.png"
)

plt.show()

print("\nTASK 4 COMPLETED SUCCESSFULLY!")

# TASK 5: DATA PREPROCESSING
# ==========================================

print("\n========== TASK 5: DATA PREPROCESSING ==========")

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Select numerical and categorical features
numeric_features = [
    "transaction_hour",
    "transaction_dow",
    "is_weekend",
    "is_holiday",
    "month",
    "quarter",
    "amount",
    "old_balance",
    "new_balance",
    "balance_change",
    "amount_balance_ratio",
    "is_international",
    "is_round_amount",
    "trusted_device_count",
    "new_device_flag",
    "distance_from_home_km",
    "credit_limit",
    "credit_utilization",
    "credit_score",
    "account_age_days",
    "transaction_velocity_1h",
    "high_value_transaction",
    "risk_indicator"
]

categorical_features = [
    "customer_segment",
    "account_type",
    "currency",
    "mcc_category",
    "merchant_sector",
    "payment_rail",
    "transaction_status",
    "device_type",
    "country"
]

# Check required columns
all_features = numeric_features + categorical_features

missing_features = [
    col for col in all_features if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

# 2. Separate input and target
X = df[all_features].copy()
y = df["fraud_label"].astype(int)

# 3. Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split:")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# 4. Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# 5. Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

# 6. Combine both preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# 7. Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)

# 8. Display processed data
print("\nProcessed Dataset Shapes:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Testing:", X_test_processed.shape)

print("\nTASK 5 COMPLETED SUCCESSFULLY!")


# TASK 6: LOGISTIC REGRESSION BASELINE
# ==========================================

print("\n========== TASK 6: LOGISTIC REGRESSION ==========")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Create Logistic Regression model
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# 2. Train model
print("\nTraining Logistic Regression model...")

lr_model.fit(X_train_processed, y_train)

# 3. Predict validation data
lr_predictions = lr_model.predict(X_val_processed)

lr_probabilities = lr_model.predict_proba(
    X_val_processed
)[:, 1]

# 4. Evaluate model
lr_accuracy = accuracy_score(
    y_val, lr_predictions
)

lr_precision = precision_score(
    y_val, lr_predictions, zero_division=0
)

lr_recall = recall_score(
    y_val, lr_predictions, zero_division=0
)

lr_f1 = f1_score(
    y_val, lr_predictions, zero_division=0
)

lr_roc_auc = roc_auc_score(
    y_val, lr_probabilities
)

print("\nLOGISTIC REGRESSION RESULTS")

print("Accuracy:", round(lr_accuracy, 4))
print("Precision:", round(lr_precision, 4))
print("Recall:", round(lr_recall, 4))
print("F1 Score:", round(lr_f1, 4))
print("ROC-AUC:", round(lr_roc_auc, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        lr_predictions,
        zero_division=0
    )
)

# 5. Confusion Matrix
cm = confusion_matrix(
    y_val,
    lr_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "logistic_regression_confusion_matrix.png"
)

plt.show()

print("\nTASK 6 COMPLETED SUCCESSFULLY!")


# TASK 7: DEEP LEARNING MODEL
# ==========================================

print("\n========== TASK 7: DEEP LEARNING ==========")

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# 1. Convert data into TensorFlow-compatible format
X_train_dl = np.asarray(X_train_processed, dtype=np.float32)
X_val_dl = np.asarray(X_val_processed, dtype=np.float32)

y_train_dl = np.asarray(y_train, dtype=np.float32)
y_val_dl = np.asarray(y_val, dtype=np.float32)

# 2. Handle imbalanced fraud data
classes = np.unique(y_train_dl)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_dl
)

class_weights = dict(zip(classes.astype(int), weights))

print("\nClass Weights:", class_weights)

# 3. Build Artificial Neural Network
model = Sequential([
    Input(shape=(X_train_dl.shape[1],)),

    Dense(128, activation="relu"),
    Dropout(0.30),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(
            curve="PR",
            name="pr_auc"
        )
    ]
)

# 5. Display model architecture
print("\nDeep Learning Model Summary:")
model.summary()

# 6. Early stopping
early_stopping = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=5,
    restore_best_weights=True
)

# 7. Train the model
print("\nStarting Deep Learning Model Training...")

history = model.fit(
    X_train_dl,
    y_train_dl,
    validation_data=(X_val_dl, y_val_dl),
    epochs=30,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# 8. Save trained model
model_path = ARTIFACT_DIR / "fraud_detection_model.keras"

model.save(model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)

print("\nTASK 7 COMPLETED SUCCESSFULLY!")


print("\n========== TASK 8: MODEL PERFORMANCE VISUALIZATION ==========")

# Create a 2x2 visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Accuracy
axes[0, 0].plot(history.history["accuracy"], label="Training Accuracy")
axes[0, 0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0, 0].set_title("Model Accuracy")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Loss
axes[0, 1].plot(history.history["loss"], label="Training Loss")
axes[0, 1].plot(history.history["val_loss"], label="Validation Loss")
axes[0, 1].set_title("Model Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. Precision and Recall
axes[1, 0].plot(history.history["val_precision"], label="Validation Precision")
axes[1, 0].plot(history.history["val_recall"], label="Validation Recall")
axes[1, 0].set_title("Validation Precision vs Recall")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Score")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. ROC-AUC and PR-AUC
axes[1, 1].plot(history.history["val_roc_auc"], label="Validation ROC-AUC")
axes[1, 1].plot(history.history["val_pr_auc"], label="Validation PR-AUC")
axes[1, 1].set_title("Validation AUC Scores")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Score")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()

# Save visualization
chart_path = REPORT_DIR / "deep_learning_training_analysis.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print("Training visualization saved:", chart_path)
print("\nTASK 8 COMPLETED SUCCESSFULLY!")


print("\n========== TASK 9: FRAUD THRESHOLD TUNING ==========")

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Predict fraud probabilities using validation data
val_probabilities = model.predict(X_val_dl, verbose=0).ravel()

# 2. Calculate precision, recall and possible thresholds
precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

# 3. Calculate F1 score for each threshold
f1_values = (
    2 * precision_values[:-1] * recall_values[:-1] /
    (precision_values[:-1] + recall_values[:-1] + 1e-10)
)

# 4. Select threshold with highest F1 score
best_index = np.argmax(f1_values)
best_threshold = float(thresholds[best_index])

print("\nBest Threshold:", round(best_threshold, 4))
print("Validation Precision:", round(precision_values[best_index], 4))
print("Validation Recall:", round(recall_values[best_index], 4))
print("Validation F1 Score:", round(f1_values[best_index], 4))

# 5. Apply selected threshold
val_predictions = (val_probabilities >= best_threshold).astype(int)

# 6. Confusion Matrix
cm = confusion_matrix(y_val, val_predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Deep Learning - Validation Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

cm_path = REPORT_DIR / "deep_learning_confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.show()

# 7. Classification Report
print("\nClassification Report:")
print(classification_report(
    y_val,
    val_predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# 8. Precision-Recall Threshold Visualization
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precision_values[:-1], label="Precision")
plt.plot(thresholds, recall_values[:-1], label="Recall")
plt.plot(thresholds, f1_values, label="F1 Score")

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    label=f"Best Threshold: {best_threshold:.3f}"
)

plt.title("Precision, Recall and F1 Score at Different Thresholds")
plt.xlabel("Prediction Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

threshold_path = REPORT_DIR / "threshold_analysis.png"
plt.savefig(threshold_path, dpi=300)
plt.show()

print("\n========== TASK 10: FINAL MODEL EVALUATION ==========")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# 1. Prepare test data
X_test_dl = np.asarray(X_test_processed, dtype=np.float32)
y_test_dl = np.asarray(y_test, dtype=np.int32)

# 2. Predict fraud probabilities
test_probabilities = model.predict(X_test_dl, verbose=0).ravel()

# 3. Apply threshold selected using validation data
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

# 4. Calculate performance metrics
test_accuracy = accuracy_score(y_test_dl, test_predictions)
test_precision = precision_score(y_test_dl, test_predictions, zero_division=0)
test_recall = recall_score(y_test_dl, test_predictions, zero_division=0)
test_f1 = f1_score(y_test_dl, test_predictions, zero_division=0)
test_roc_auc = roc_auc_score(y_test_dl, test_probabilities)
test_pr_auc = average_precision_score(y_test_dl, test_probabilities)

print("\nFINAL TEST PERFORMANCE")
print("------------------------------")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1 Score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_roc_auc:.4f}")
print(f"PR-AUC:    {test_pr_auc:.4f}")

# 5. Classification report
print("\nClassification Report:")
print(classification_report(
    y_test_dl,
    test_predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# 6. Final confusion matrix
final_cm = confusion_matrix(y_test_dl, test_predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Final Test Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

final_cm_path = REPORT_DIR / "final_test_confusion_matrix.png"
plt.savefig(final_cm_path, dpi=300)
plt.show()

# 7. Save final performance metrics
metrics = {
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1_score": test_f1,
    "roc_auc": test_roc_auc,
    "pr_auc": test_pr_auc,
    "threshold": best_threshold
}

metrics_path = REPORT_DIR / "final_model_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

print("\nFinal confusion matrix:", final_cm_path)
print("Final metrics saved:", metrics_path)

print("\nTASK 10 COMPLETED SUCCESSFULLY!")


print("\n========== TASK 11: FEATURE IMPORTANCE ==========")

from sklearn.inspection import permutation_importance

# Use validation data to measure feature importance
def ann_predict(X):
    return model.predict(
        np.asarray(X, dtype=np.float32),
        verbose=0
    ).ravel()

# Permutation importance for fraud probability
result = permutation_importance(
    lr_model,
    X_val_processed,
    y_val,
    scoring="average_precision",
    n_repeats=5,
    random_state=42
)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

# Visualize top 15 features
top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Fraud Detection Features - Logistic Regression")
plt.xlabel("Decrease in Average Precision")
plt.ylabel("Features")
plt.tight_layout()

importance_path = REPORT_DIR / "feature_importance.png"
plt.savefig(importance_path, dpi=300)
plt.show()

# Save feature importance
importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

print("\nFeature importance saved:", importance_path)
print("\nTASK 11 COMPLETED SUCCESSFULLY!")

print("\n========== TASK 12: SHAP EXPLAINABILITY ==========")

import shap

# 1. Select a small background sample from training data
background = X_train_dl[:100]

# 2. Select validation transactions for explanation
explain_data = X_val_dl[:50]

# 3. Create SHAP Deep Explainer
explainer = shap.DeepExplainer(model, background)

# 4. Calculate SHAP values
shap_values = explainer.shap_values(explain_data)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

# Remove final output dimension if present
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

# 5. Get feature names
feature_names = preprocessor.get_feature_names_out()

# 6. SHAP summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    explain_data,
    feature_names=feature_names,
    max_display=15,
    show=False
)

shap_summary_path = REPORT_DIR / "shap_summary.png"
plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
plt.show()

# 7. SHAP feature importance
mean_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_SHAP_Value": mean_shap
}).sort_values(
    by="Mean_SHAP_Value",
    ascending=False
)

print("\nTop 15 SHAP Features:")
print(shap_importance.head(15).to_string(index=False))

# 8. Save SHAP importance
shap_importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nSHAP summary saved:", shap_summary_path)
print("\nTASK 12 COMPLETED SUCCESSFULLY!")

print("\n========== SAVING DASHBOARD COMPONENTS ==========")

joblib.dump(
    preprocessor,
    ARTIFACT_DIR / "fraud_preprocessor.pkl"
)

joblib.dump(
    best_threshold,
    ARTIFACT_DIR / "fraud_threshold.pkl"
)

joblib.dump(
    amount_threshold,
    ARTIFACT_DIR / "amount_threshold.pkl"
)

joblib.dump(
    numeric_features + categorical_features,
    ARTIFACT_DIR / "model_features.pkl"
)

print("All dashboard components saved successfully!")

