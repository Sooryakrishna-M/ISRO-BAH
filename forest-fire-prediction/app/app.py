# app/app.py
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("src/forest_fire_model_all_years.pkl")

st.title("🔥 Forest Fire Risk Predictor")

brightness = st.slider("Brightness", 300, 400, 340)
frp = st.slider("Fire Radiative Power (FRP)", 0, 100, 15)
bright_t31 = st.slider("Brightness Temp (Band 31)", 270, 330, 300)

if st.button("Predict"):
    input_data = np.array([[brightness, frp, bright_t31]])
    result = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0][1]

    st.markdown(f"### Prediction: {'🔥 High Risk' if result == 1 else '✅ Low Risk'}")
    st.write(f"Confidence: {confidence*100:.2f}%")
