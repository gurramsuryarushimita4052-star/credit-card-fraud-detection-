
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Enter transaction details to check the model's prediction."
)

# Load the trained model
model_path = Path(__file__).parent / "fraud_model.joblib"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Trained model not found. Please run python train.py first.")
    st.stop()

# Load dataset to get the exact feature names
data_path = Path(__file__).parent / "dataset" / "creditcard.csv"

try:
    df = pd.read_csv(data_path, nrows=5)
except FileNotFoundError:
    st.error("Dataset not found at dataset/creditcard.csv")
    st.stop()

feature_names = list(df.drop(columns=["Class"]).columns)

st.info(
    "Demo only: this app uses a trained educational model, "
    "not a real bank fraud detection service."
)

st.subheader("Transaction Features")

with st.form("transaction_form"):
    inputs = {}

    for feature in feature_names:
        default_value = 0.0

        if feature == "Amount":
            inputs[feature] = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=0.0,
                step=1.0
            )
        else:
            inputs[feature] = st.number_input(
                feature,
                value=default_value,
                format="%.6f"
            )

    submitted = st.form_submit_button("Check Transaction")

if submitted:
    input_df = pd.DataFrame([inputs], columns=feature_names)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Model prediction: Potential Fraud")
    else:
        st.success("Model prediction: Normal Transaction")

    st.write(f"Model-estimated fraud probability: {probability:.2%}")

    st.caption(
        "This is a model estimate, not a confirmed fraud determination. "
        "The default inputs may not represent a real transaction."
    )