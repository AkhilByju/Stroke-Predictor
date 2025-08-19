# app.py
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from joblib import load

MODEL_PATH = Path("model.joblib")
META_PATH  = Path("metadata.json")

model = load("model.joblib")
meta = json.loads(Path("metadata.json").read_text())
THRESHOLD = meta.get("chosen_threshold", 0.5)


COLUMNS = [
    "gender",           # "Male","Female","Other"
    "age",              # float
    "hypertension",     # 0/1
    "heart_disease",    # 0/1
    "ever_married",     # "Yes"/"No"
    "work_type",        # "children","Govt_job","Never_worked","Private","Self-employed"
    "Residence_type",   # "Urban"/"Rural"
    "avg_glucose_level",# float
    "bmi",              # float
    "smoking_status",   # "formerly smoked","never smoked","smokes","Unknown"
]

def predict_risk(
    gender, age, hypertension, heart_disease, ever_married, work_type,
    residence_type, avg_glucose_level, bmi, smoking_status, threshold
):
    row = {
        "gender": gender,
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi),
        "smoking_status": smoking_status,
    }
    X = pd.DataFrame([row])
    proba = model.predict_proba(X)[:, 1][0]
    label = int(proba >= float(threshold))
    return float(proba), int(label)

with gr.Blocks(title="Stroke Risk Demo (Educational)") as demo:
    gr.Markdown(
        "### Stroke Risk (Educational Demo)\n"
        "- This tool is **for education only**, not medical advice.\n"
        "- Model: Logistic Regression in an sklearn **Pipeline** with calibration.\n"
    )
    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(choices=["Male","Female","Other"], value="Female", label="Gender")
            age = gr.Number(value=45, label="Age")
            hypertension = gr.Radio(choices=[0,1], value=0, label="Hypertension (0/1)")
            heart_disease = gr.Radio(choices=[0,1], value=0, label="Heart Disease (0/1)")
            ever_married = gr.Dropdown(choices=["Yes","No"], value="Yes", label="Ever Married")
            work_type = gr.Dropdown(
                choices=["children","Govt_job","Never_worked","Private","Self-employed"],
                value="Private", label="Work Type"
            )
            residence_type = gr.Dropdown(choices=["Urban","Rural"], value="Urban", label="Residence Type")
            avg_glucose_level = gr.Number(value=95.0, label="Avg Glucose Level")
            bmi = gr.Number(value=26.0, label="BMI")
            smoking_status = gr.Dropdown(
                choices=["formerly smoked","never smoked","smokes","Unknown"],
                value="never smoked", label="Smoking Status"
            )
            threshold = gr.Slider(minimum=0.05, maximum=0.95, step=0.01,
                                  value=THRESHOLD, label="Decision Threshold")
            btn = gr.Button("Predict")

        with gr.Column():
            proba_out = gr.Number(label="Calibrated Risk Probability", precision=6)
            label_out = gr.Number(label="Predicted Label (1=risk, 0=no risk)")

    btn.click(
        predict_risk,
        inputs=[gender, age, hypertension, heart_disease, ever_married, work_type,
                residence_type, avg_glucose_level, bmi, smoking_status, threshold],
        outputs=[proba_out, label_out]
    )

    gr.Markdown(
        "> **Disclaimer:** This demo is not a diagnostic tool. Do not use for medical decisions."
    )

if __name__ == "__main__":
    demo.launch()
