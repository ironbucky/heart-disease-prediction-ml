import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

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

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
    :root {
        --card-bg: #ffffff;
        --muted: #64748b;
        --primary: #dc2626;
        --accent: #2563eb;
        --success: #16a34a;
        --radius: 16px;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .hero { text-align: center; padding: 2rem 1rem 1rem; }
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #dc2626, #2563eb);
        color: white; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        padding: 0.35rem 0.9rem; border-radius: 999px; margin-bottom: 0.75rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    .title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #dc2626, #2563eb);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; margin: 0.25rem 0;
    }
    .subtitle { color: var(--muted); font-size: 1rem; max-width: 520px; margin: 0 auto 1.5rem; }
    .card {
        background: var(--card-bg); border: 1px solid #e2e8f0;
        border-radius: var(--radius); padding: 1.5rem; box-shadow: 0 6px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .card-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 0.75rem; }
    .result-card { border-radius: var(--radius); padding: 1.5rem; text-align: center; margin-top: 1rem; box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08); }
    .result-disease { background: linear-gradient(135deg, #fee2e2, #fecaca); border: 1px solid #fca5a5; color: #991b1b; }
    .result-healthy { background: linear-gradient(135deg, #dcfce7, #bbf7d0); border: 1px solid #86efac; color: #166534; }
    .result-icon { font-size: 2.6rem; margin-bottom: 0.25rem; }
    .result-label { font-size: 1.5rem; font-weight: 800; margin: 0.25rem 0; }
    .result-note { font-size: 0.9rem; opacity: 0.85; }
    .footer-note { text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 2rem; }
    label[data-testid="stWidgetLabel"] p { font-weight: 600; color: #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <span class="badge">Medical ML Demo</span>
        <div class="title">Heart Disease Predictor</div>
        <p class="subtitle">
            Estimate heart disease risk from 4 clinical features using a trained
            Support Vector Classifier on the UCI Cleveland dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
st.markdown('<div class="card"><div class="card-title">🩺 Patient Features</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    chestpain = st.selectbox(
        "Chest Pain Type",
        ["Typical", "Atypical", "NonAnginal", "Asymptomatic"],
        index=0,
        help="Typical angina, atypical angina, non-anginal pain, or asymptomatic.",
    )
    sex = st.selectbox(
        "Sex",
        ["Male", "Female"],
        help="Biological sex of the patient.",
    )
with col2:
    fbs = st.selectbox(
        "Fasting Blood Sugar",
        ["Normal", "High"],
        help="Normal (< 120 mg/dl) or High (>= 120 mg/dl).",
    )
    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"],
        help="Angina triggered by exercise.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Predict ----------
predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

if predict_clicked:
    user_input = pd.DataFrame({"ChestPain": [chestpain], "Sex": [sex], "FBS": [fbs], "ExAng": [exang]})
    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "FBS": encoders["FBS"].transform(user_input["FBS"]),
        "ExAng": encoders["ExAng"].transform(user_input["ExAng"]),
    })
    pred = model.predict(encoded)[0]

    if pred == 1:
        st.markdown(
            """
            <div class="result-card result-disease">
                <div class="result-icon">⚠️</div>
                <div class="result-label">HEART DISEASE (Yes)</div>
                <div class="result-note">Model indicates elevated risk. Consult a clinician for proper evaluation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-card result-healthy">
                <div class="result-icon">✅</div>
                <div class="result-label">NO HEART DISEASE (No)</div>
                <div class="result-note">Model indicates low risk based on the provided features.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Encoded feature vector"):
        st.dataframe(encoded, hide_index=True, use_container_width=True)

st.markdown(
    '<p class="footer-note">⚠️ Educational demo only — not a medical diagnosis. Always consult a qualified healthcare professional.</p>',
    unsafe_allow_html=True,
)
