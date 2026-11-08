"""Train and evaluate income classifiers on the UCI Adult dataset."""
import json
import os
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SAMPLE_SIZE = 5000
DATASET_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
DATA_DIR = Path("adult")
MODEL_DIR = Path("model")
TARGET = "income"
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    TARGET,
]
FEATURES = ADULT_COLUMNS[:-1]
CATEGORICAL_FEATURES = [
    feature for feature in FEATURES if feature not in NUMERIC_FEATURES
]


def read_adult_file(source, skiprows=0):
    data = pd.read_csv(
        source,
        header=None,
        names=ADULT_COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,
        na_values="?",
    )
    data[TARGET] = data[TARGET].str.strip().str.rstrip(".")
    return data.dropna(subset=[TARGET])


def load_adult_data():
    train_path = DATA_DIR / "adult.data"
    test_path = DATA_DIR / "adult.test"
    if train_path.exists() and test_path.exists():
        train_data = read_adult_file(train_path)
        test_data = read_adult_file(test_path, skiprows=1)
        return train_data, test_data

    with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
        archive_data = response.read()
    with zipfile.ZipFile(BytesIO(archive_data)) as archive:
        train_data = read_adult_file(BytesIO(archive.read("adult.data")))
        test_data = read_adult_file(
            BytesIO(archive.read("adult.test")), skiprows=1
        )
    return train_data, test_data


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def make_model(classifier):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier),
    ])


def positive_class_scores(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        positive_index = list(model.classes_).index(">50K")
        return probabilities[:, positive_index]
    scores = model.decision_function(X)
    if model.classes_[1] != ">50K":
        scores = -scores
    return scores


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train_data, adult_test = load_adult_data()
    test_data = adult_test.sample(
        n=min(TEST_SAMPLE_SIZE, len(adult_test)), random_state=RANDOM_STATE
    )
    test_data.to_csv("test_data.csv", index=False)

    with (MODEL_DIR / "label_info.json").open("w", encoding="utf-8") as file:
        json.dump({
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "classes": sorted(train_data[TARGET].unique().tolist()),
            "positive_class": ">50K",
        }, file, indent=2)

    X_train, y_train = train_data[FEATURES], train_data[TARGET]
    X_test, y_test = test_data[FEATURES], test_data[TARGET]
    models = {
        "Logistic Regression": make_model(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": make_model(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "KNN": make_model(KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": make_model(GaussianNB()),
        "Support Vector Machine": make_model(
            LinearSVC(dual=False, max_iter=5000, random_state=RANDOM_STATE)
        ),
    }

    results, predictions = [], {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        results.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(
                y_test == ">50K", positive_class_scores(model, X_test)
            ),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        })
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)

    results_df = (pd.DataFrame(results)
                  .sort_values("Accuracy", ascending=False).reset_index(drop=True))
    results_df.to_csv("adult_classification_results.csv", index=False)
    print("\n", results_df.round(4).to_string(index=False))
    print("\nMarkdown table for README:\n")
    print(results_df.round(4).to_markdown(index=False))

    labels = sorted(y_train.unique())
    for name, y_pred in predictions.items():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        print(classification_report(y_test, y_pred, zero_division=0))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for index, (name, y_pred) in enumerate(predictions.items()):
        sns.heatmap(
            confusion_matrix(y_test, y_pred, labels=labels),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[index],
        )
        axes[index].set_title(name)
        axes[index].set_xlabel("Predicted income")
        axes[index].set_ylabel("Actual income")
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=120)
    plt.close(fig)

    best = results_df.iloc[0]
    print(f"\nBest model by accuracy: {best['ML Model Name']} ({best['Accuracy']:.4f})")
    print(f"Training records: {len(train_data)} | Test records: {len(test_data)}")


if __name__ == "__main__":
    main()
