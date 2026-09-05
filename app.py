import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Muhammad Shamsudeen Yakubu Crop Recommender", page_icon="🌱", layout="centered")

MODEL_PATH = Path(__file__).parent / "artifacts" / "crop_recommender.joblib"
model = joblib.load(MODEL_PATH)

st.title("🌱 Muhammad Shamsudeen Yakubu")
st.subheader("AI Crop Recommendation System")
st.write("Enter soil and environmental conditions to receive a crop recommendation.")

with st.form("crop_form"):
    c1, c2 = st.columns(2)
    with c1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=40.0)
        K = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, value=40.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.01)
    with c2:
        soil = st.selectbox("Soil Texture", ["sandy loam", "loamy", "clay", "loamy clay"])
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0)

    submitted = st.form_submit_button("Recommend Crop")

if submitted:
    X_new = pd.DataFrame([{
        "N": N, "P": P, "K": K, "temperature": temperature,
        "humidity": humidity, "ph": ph, "rainfall": rainfall, "soil": soil
    }])
    prediction = model.predict(X_new)[0]
    st.success(f"Primary Recommended Crop: **{prediction.replace('_', ' ').title()}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)[0]
        classes = model.named_steps["model"].classes_
        top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
        st.write("### Top 3 Recommendations")
        for crop, prob in top:
            st.write(f"**{crop.replace('_', ' ').title()}** — {prob:.1%}")
            st.progress(float(prob))

st.caption("Model pipeline: preprocessing + classification. This tool is intended as a decision-support aid.")
