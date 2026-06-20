import streamlit as st
import pandas as pd
import joblib

st.title("📊 Prediction Results Visualization")

# Load metrics
metrics = joblib.load("model_metrics.pkl")
# Convert to DataFrame
df = pd.DataFrame(metrics).T

st.subheader("Model Comparison Table")
st.dataframe(df.style.highlight_max(axis=0))

st.subheader("Accuracy Comparison")

st.bar_chart(df["accuracy"])

st.subheader("Precision Comparison")

st.bar_chart(df["precision"])

st.subheader("Recall Comparison")

st.bar_chart(df["recall"])

st.subheader("F1 Score Comparison")

st.bar_chart(df["f1"])

# Best model highlight
best_model = df["accuracy"].idxmax()

st.success(f"🏆 Best Performing Model: {best_model}")