import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

st.set_page_config(page_title="Adult Income Classifier", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_info():
    with open(os.path.join(MODEL_DIR, "label_info.json"), encoding="utf-8") as file:
        return json.load(file)


def positive_class_scores(model, X, positive_class):
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index(positive_class)
        return model.predict_proba(X)[:, positive_index]
    scores = model.decision_function(X)
    if model.classes_[1] != positive_class:
        scores = -scores
    return scores


def compute_metrics(model, X, y, positive_class):
    y_pred = model.predict(X)
    try:
        auc = roc_auc_score(
            y == positive_class,
            positive_class_scores(model, X, positive_class),
        )
    except ValueError:
        auc = np.nan
    return y_pred, {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


st.title("Adult income classification")
st.write("Compare five pre-trained models predicting whether annual income exceeds $50K.")

info = load_info()
features = info["features"]
target = info["target"]
classes = info["classes"]
positive_class = info["positive_class"]

with st.sidebar:
    st.header("Evaluation data")
    uploaded = st.file_uploader("Upload a labeled Adult CSV", type="csv")
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))

if uploaded is not None:
    data = pd.read_csv(uploaded, na_values="?")
elif use_sample and os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv", na_values="?")
else:
    st.info("Upload a labeled Adult CSV or select the bundled test data.")
    st.stop()

missing = [column for column in features + [target] if column not in data.columns]
if missing:
    st.error(f"CSV is missing required Adult columns: {missing}")
    st.stop()

data = data.dropna(subset=[target]).copy()
if data.empty:
    st.error("The CSV contains no rows with an income label.")
    st.stop()

y = data[target].astype(str).str.strip().str.rstrip(".")
unknown_classes = sorted(set(y) - set(classes))
if unknown_classes:
    st.error(f"Income values must be one of {classes}; found {unknown_classes}.")
    st.stop()

X = data[features].copy()
for feature in info["numeric_features"]:
    X[feature] = pd.to_numeric(X[feature], errors="coerce")

st.subheader("Data preview")
st.write(f"{len(data):,} records | {len(features)} features")
st.dataframe(data.head(), alt="Preview of Adult income evaluation records")

model = load_model(model_name)
y_pred, metrics = compute_metrics(model, X, y, positive_class)

st.subheader(f"Evaluation metrics - {model_name}")
with st.container(horizontal=True):
    for metric, value in metrics.items():
        st.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y, y_pred, labels=classes)
    matrix_frame = pd.DataFrame(
        matrix,
        index=pd.Index(classes, name="Actual"),
        columns=pd.Index(classes, name="Predicted"),
    )
    st.dataframe(matrix_frame, alt="Actual versus predicted income counts")
with right:
    st.subheader("Classification report")
    report = classification_report(
        y, y_pred, labels=classes, zero_division=0, output_dict=True
    )
    st.dataframe(
        pd.DataFrame(report).T.round(3),
        alt="Precision, recall, and F1 by income class",
    )

st.subheader("Model comparison")
rows = []
for name in MODEL_FILES:
    _, model_metrics = compute_metrics(load_model(name), X, y, positive_class)
    rows.append({"Model": name, **model_metrics})
comparison = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    comparison.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
    alt="Evaluation metrics compared across all five models",
)
st.bar_chart(
    comparison[["Accuracy", "F1", "MCC"]],
    alt="Accuracy, macro F1, and Matthews correlation by model",
)
