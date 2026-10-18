import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_encoders():
    enc = {}
    enc["ChestPain"] = LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
    enc["Sex"] = LabelEncoder().fit(["Male", "Female"])
    enc["FBS"] = LabelEncoder().fit(["Normal", "High"])
    enc["ExAng"] = LabelEncoder().fit(["No", "Yes"])
    return enc

model = load_model()
encoders = build_encoders()

st.title("❤️ Heart Disease Predictor")
st.markdown("Trained SVC on UCI Cleveland heart disease dataset (4 features).")

col1, col2 = st.columns(2)
with col1:
    chestpain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"], index=0)
    sex = st.selectbox("Sex", ["Male", "Female"])
with col2:
    fbs = st.selectbox("Fasting Blood Sugar", ["Normal", "High"])
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({"ChestPain": [chestpain], "Sex": [sex], "FBS": [fbs], "ExAng": [exang]})
    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "FBS": encoders["FBS"].transform(user_input["FBS"]),
        "ExAng": encoders["ExAng"].transform(user_input["ExAng"]),
    })
    pred = model.predict(encoded)[0]
    if pred == 1:
        st.error("### HEART DISEASE (Yes)")
    else:
        st.success("### NO HEART DISEASE (No)")
    with st.expander("Encoded feature vector"):
        st.dataframe(encoded, hide_index=True)
PYEOF