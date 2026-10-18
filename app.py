import pickle
from pathlib import Path
import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).parent
MODEL_PATH = BASE / "svc_trained_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def build_encoders():
    enc = {}
    enc["ChestPain"] = LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
    enc["Sex"] = LabelEncoder().fit(["Male", "Female"])
    enc["FBS"] = LabelEncoder().fit(["Normal", "High"])
    enc["ExAng"] = LabelEncoder().fit(["No", "Yes"])
    return enc

model = load_model()
encoders = build_encoders()

def predict(chestpain, sex, fbs, exang):
    row = pd.DataFrame({"ChestPain": [chestpain], "Sex": [sex], "FBS": [fbs], "ExAng": [exang]})
    encoded = pd.DataFrame({
        "ChestPain": encoders["ChestPain"].transform(row["ChestPain"]),
        "Sex": encoders["Sex"].transform(row["Sex"]),
        "FBS": encoders["FBS"].transform(row["FBS"]),
        "ExAng": encoders["ExAng"].transform(row["ExAng"]),
    })
    pred = model.predict(encoded)[0]
    return "HEART DISEASE (Yes)" if pred == 1 else "NO HEART DISEASE (No)"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Typical", "Atypical", "NonAnginal", "Asymptomatic"], value="Typical", label="Chest Pain Type"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Sex"),
        gr.Dropdown(["Normal", "High"], value="Normal", label="Fasting Blood Sugar"),
        gr.Dropdown(["No", "Yes"], value="No", label="Exercise Induced Angina"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Predictor",
    description="SVC model trained on UCI Cleveland heart disease dataset (4 features).",
)

if __name__ == "__main__":
    demo.launch(share=True)
PYEOF