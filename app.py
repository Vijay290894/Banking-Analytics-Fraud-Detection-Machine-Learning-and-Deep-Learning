import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
from pathlib import Path
import shap

# --------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------

@st.cache_resource
def load_components():
    model = tf.keras.models.load_model(
        ARTIFACT_DIR / "fraud_detection_model.keras"
    )

    preprocessor = joblib.load(
        ARTIFACT_DIR / "fraud_preprocessor.pkl"
    )

    threshold = joblib.load(
        ARTIFACT_DIR / "fraud_threshold.pkl"
    )

    amount_threshold = joblib.load(
        ARTIFACT_DIR / "amount_threshold.pkl"
    )

    features = joblib.load(
        ARTIFACT_DIR / "model_features.pkl"
    )

    return model, preprocessor, threshold, amount_threshold, features


try:
    model, preprocessor, threshold, amount_threshold, features = (
        load_components()
    )
except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()

# --------------------------------
# 3. DASHBOARD HEADER
# --------------------------------

st.title("🏦 Banking Transaction Fraud Detection")
st.write(
    "An AI-powered dashboard to identify suspicious "
    "banking transactions using Deep Learning."
)

st.info(f"Fraud prediction threshold: {threshold:.4f}")

# --------------------------------
# 4. LOAD DATASET
# --------------------------------

st.sidebar.header("Transaction Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    default_path = DATA_DIR / "banking_fraud_data.csv"

    if default_path.exists():
        df = pd.read_csv(default_path)
        st.sidebar.success("Using project sample dataset")
    else:
        st.warning("Please upload a transaction CSV file.")
        st.stop()

# --------------------------------
# 5. DATA PREPARATION
# --------------------------------

required_raw = [
    "amount",
    "old_balance",
    "new_balance",
    "is_international",
    "new_device_flag",
    "is_weekend",
    "is_holiday",
    "is_round_amount"
]

missing_raw = [
    col for col in required_raw if col not in df.columns
]

if missing_raw:
    st.error(
        "Missing required raw columns: "
        + ", ".join(missing_raw)
    )
    st.stop()

df = df.copy()

# Create engineered features
df["balance_change"] = (
    df["new_balance"] - df["old_balance"]
)

df["amount_balance_ratio"] = (
    df["amount"] / (df["old_balance"].abs() + 1)
)

df["high_value_transaction"] = (
    df["amount"] >= amount_threshold
).astype(int)

risk_columns = [
    "is_international",
    "new_device_flag",
    "is_weekend",
    "is_holiday",
    "is_round_amount"
]

df["risk_indicator"] = (
    df[risk_columns].fillna(0).sum(axis=1)
)

missing_features = [
    col for col in features if col not in df.columns
]

if missing_features:
    st.error(
        "Missing model features: "
        + ", ".join(missing_features)
    )
    st.stop()

# --------------------------------
# 6. PREDICT FRAUD
# --------------------------------

X = df[features].copy()

X_processed = preprocessor.transform(X)

probabilities = model.predict(
    np.asarray(X_processed, dtype=np.float32),
    verbose=0
).ravel()

df["fraud_probability"] = probabilities

df["fraud_prediction"] = (
    probabilities >= threshold
).astype(int)

df["prediction_status"] = df["fraud_prediction"].map({
    0: "Legitimate",
    1: "Potential Fraud"
})

# --------------------------------
# 7. DASHBOARD KPI
# --------------------------------

total_transactions = len(df)

fraud_count = int(df["fraud_prediction"].sum())

legitimate_count = total_transactions - fraud_count

fraud_percentage = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Potential Fraud",
    f"{fraud_count:,}"
)

col3.metric(
    "Legitimate",
    f"{legitimate_count:,}"
)

col4.metric(
    "Potential Fraud Rate",
    f"{fraud_percentage:.2f}%"
)

st.divider()

# --------------------------------
# 8. FRAUD VISUALIZATION
# --------------------------------

st.subheader("Transaction Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    status_counts = df["prediction_status"].value_counts()

    fig1 = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        title="Fraud vs Legitimate Transactions",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    fig2 = px.histogram(
        df,
        x="fraud_probability",
        color="prediction_status",
        nbins=30,
        title="Fraud Probability Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------
# 9. FRAUD AMOUNT ANALYSIS
# --------------------------------

st.subheader("Transaction Amount Analysis")

fig3 = px.box(
    df,
    x="prediction_status",
    y="amount",
    color="prediction_status",
    title="Transaction Amount by Prediction"
)

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------
# 10. FEATURE IMPORTANCE
# --------------------------------

st.subheader("Important Fraud Detection Features")

importance_path = REPORT_DIR / "shap_feature_importance.csv"

if importance_path.exists():

    importance_df = pd.read_csv(importance_path)

    top_features = importance_df.head(10).sort_values(
        "Mean_SHAP_Value"
    )

    fig4 = px.bar(
        top_features,
        x="Mean_SHAP_Value",
        y="Feature",
        orientation="h",
        title="Top 10 SHAP Features"
    )

    st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("SHAP feature importance report not found.")

# --------------------------------
# 11. TRANSACTION PREDICTION TABLE
# --------------------------------

st.subheader("Transaction Prediction Results")

display_columns = [
    col for col in [
        "transaction_id",
        "amount",
        "fraud_probability",
        "prediction_status"
    ]
    if col in df.columns
]

st.dataframe(
    df[display_columns],
    use_container_width=True
)

# --------------------------------
# 12. DOWNLOAD RESULTS
# --------------------------------

csv_output = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Fraud Prediction Report",
    data=csv_output,
    file_name="banking_fraud_predictions.csv",
    mime="text/csv"
)

st.caption(
    "Educational prototype using a synthetic banking transaction dataset. "
    "Predictions require human review and are not automatic fraud decisions."
)

df["prediction_status"] = df["fraud_prediction"].map({
    0: "Legitimate",
    1: "Potential Fraud"
})

# Risk categorization
def categorize_risk(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"

df["risk_category"] = df["fraud_probability"].apply(
    categorize_risk
)

# --------------------------------
# 13. FRAUD INVESTIGATION PANEL
# --------------------------------

st.divider()

st.subheader("🔍 Transaction Investigation")

# Select transaction
if "transaction_id" in df.columns:
    selected_transaction = st.selectbox(
        "Select Transaction ID",
        df["transaction_id"].astype(str).tolist()
    )

    selected_row = df[
        df["transaction_id"].astype(str)
        == selected_transaction
    ].iloc[0]

else:
    selected_index = st.selectbox(
        "Select Transaction",
        df.index.tolist()
    )

    selected_row = df.loc[selected_index]

# Display transaction information
col1, col2, col3 = st.columns(3)

col1.metric(
    "Transaction Amount",
    f"{selected_row['amount']:,.2f}"
)

col2.metric(
    "Fraud Probability",
    f"{selected_row['fraud_probability']:.2%}"
)

col3.metric(
    "Risk Category",
    selected_row["risk_category"]
)

# Display fraud status
if selected_row["fraud_prediction"] == 1:
    st.error("⚠️ Potential Fraud Detected - Review Required")
else:
    st.success("Transaction classified as Legitimate")

# Display investigation details
st.markdown("### Transaction Details")

details = {
    "Transaction Amount": selected_row["amount"],
    "Fraud Probability": f"{selected_row['fraud_probability']:.2%}",
    "Risk Category": selected_row["risk_category"],
    "Prediction": selected_row["prediction_status"],
    "Balance Change": selected_row["balance_change"],
    "Risk Indicator": selected_row["risk_indicator"]
}

st.table(
    pd.DataFrame(
        details.items(),
        columns=["Parameter", "Value"]
    )
)

# 14. INDIVIDUAL TRANSACTION SHAP
# --------------------------------

st.divider()

st.subheader("🧠 Why Did the Model Flag This Transaction?")

st.write(
    "SHAP explains how individual transaction features "
    "influence the model's fraud probability."
)

if st.button("Generate SHAP Explanation"):

    with st.spinner("Analyzing transaction features..."):

        try:
            # Background sample from current dataset
            background_size = min(20, len(X_processed))

            background_data = X_processed[:background_size]

            # Selected transaction
            selected_index = selected_row.name

            selected_data = np.asarray(
                X_processed[selected_index:selected_index + 1],
                dtype=np.float32
            )

            background_data = np.asarray(
                background_data,
                dtype=np.float32
            )

            # Prediction function
            def predict_probability(input_data):
                return model.predict(
                    np.asarray(input_data, dtype=np.float32),
                    verbose=0
                ).ravel()

            # Create SHAP explainer
            explainer = shap.KernelExplainer(
                predict_probability,
                background_data
            )

            # Calculate SHAP values
            shap_values = explainer.shap_values(
                selected_data,
                nsamples=100
            )

            shap_values = np.asarray(shap_values)

            # Handle SHAP output dimensions
            if shap_values.ndim == 3:
                shap_values = shap_values[0, :, 0]
            elif shap_values.ndim == 2:
                shap_values = shap_values[0]

            # Feature names
            feature_names = preprocessor.get_feature_names_out()

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_values
            })

            explanation_df["Impact"] = np.where(
                explanation_df["SHAP Value"] > 0,
                "Increases Fraud Probability",
                "Decreases Fraud Probability"
            )

            # Sort by absolute impact
            explanation_df["Absolute Impact"] = (
                explanation_df["SHAP Value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(15)

            # Display explanation
            st.success("SHAP explanation generated!")

            fig = px.bar(
                explanation_df.sort_values("SHAP Value"),
                x="SHAP Value",
                y="Feature",
                color="Impact",
                orientation="h",
                title="Individual Transaction SHAP Explanation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.write("### Top Influential Features")

            st.dataframe(
                explanation_df[
                    ["Feature", "SHAP Value", "Impact"]
                ],
                use_container_width=True
            )

        except Exception as error:
            st.error(f"SHAP explanation failed: {error}")

            st.info(
                "Please share the complete error message "
                "so we can troubleshoot it."
            )