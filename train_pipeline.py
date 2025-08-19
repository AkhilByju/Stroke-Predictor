# train_pipeline_simple.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from kagglehub import KaggleDatasetAdapter, load_dataset

# ---- Load
df = load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/stroke-prediction-dataset",
    "healthcare-dataset-stroke-data.csv",
)

# ---- Config
MODEL_PATH = Path("model.joblib")
META_PATH  = Path("metadata.json")
RANDOM_STATE = 42
TEST_SIZE = 0.2
RECALL_TARGET = 0.60      # tune as you like
MIN_PRECISION = 0.25

# ---- Clean
assert "stroke" in df.columns
df = df.drop(columns=["id"], errors="ignore")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df["stroke"] = df["stroke"].astype(int)

y = df["stroke"]
X = df.drop(columns=["stroke"])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

clf = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    penalty="l2",
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# ---- Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ---- Train
pipe.fit(X_tr, y_tr)

# ---- Evaluate + choose threshold
proba = pipe.predict_proba(X_te)[:, 1]
roc = roc_auc_score(y_te, proba)
ap  = average_precision_score(y_te, proba)

prec, rec, thr = precision_recall_curve(y_te, proba)

chosen = None
for p, r, t in zip(prec[1:], rec[1:], thr):
    if (r >= RECALL_TARGET) and (p >= MIN_PRECISION):
        chosen = float(t); break
if chosen is None:
    # fallback: maximize F1
    f1s = [(f1_score(y_te, (proba>=t).astype(int), zero_division=0), t) for t in thr]
    chosen = float(max(f1s, key=lambda x: x[0])[1])

y_def = (proba >= 0.5).astype(int)
y_opt = (proba >= chosen).astype(int)

print(f"Positives in test: {int(y_te.sum())}/{len(y_te)}")
print(f"ROC-AUC: {roc:.3f} | PR-AUC: {ap:.3f}")
print("\n=== @0.50 threshold ===")
print(classification_report(y_te, y_def, digits=3))
print(f"\nChosen threshold: {chosen:.3f}")
print("\n=== @chosen threshold ===")
print(classification_report(y_te, y_opt, digits=3))

# ---- Save
dump(pipe, MODEL_PATH)
meta = {
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
    "roc_auc": float(roc),
    "pr_auc": float(ap),
    "chosen_threshold": float(chosen),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
}
Path(META_PATH).write_text(json.dumps(meta, indent=2))
print(f"\nSaved: {MODEL_PATH} and {META_PATH}")
