import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.keras")
scaler = joblib.load("scaler.pk1")

st.title("🧠 pregnancy detection using ANN")

st.write("Enter patient details")

case = st.number_input("Case ID",0,100000)
bwt = st.number_input("bwt", 0, 300)
gestation = st.number_input("gestation", 0, 200)
parity = st.number_input("parity",0,2)
age = st.number_input("age", 18, 100)
height = st.number_input("height", 0.0, 200.0)
weight = st.number_input("weight", 0.0, 200.0)

if st.button("Predict"):
    
    input_data = np.array([[case, bwt, gestation, parity, age, height, weight]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0][0] > 0.5:
        st.error("⚠️ pregnancy risk is  detected")
    else:
        st.success("✅ No pregnancy risk detected")
