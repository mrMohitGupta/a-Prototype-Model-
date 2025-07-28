# heart_disease_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Optional: SHAP explainability
import shap

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal","target"]
    df = pd.read_csv(url, names=cols, na_values='?')
    return df

def preprocess(df):
    df = df.dropna().copy()
    df['target'] = (df['target'] > 0).astype(int)  # binary: 0=no, 1=disease
    # impute continuous
    num = ["age","trestbps","chol","thalach","oldpeak"]
    imp = SimpleImputer(strategy='mean')
    df[num] = imp.fit_transform(df[num])
    # encode categorical via get_dummies
    cat = ["cp","restecg","slope","thal","ca","sex","fbs","exang"]
    df = pd.get_dummies(df, columns=cat, drop_first=True)
    # scale numeric
    scaler = StandardScaler()
    df[num] = scaler.fit_transform(df[num])
    return df

def train_and_evaluate(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                           test_size=0.2, stratify=y, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K‑Nearest Neighbors": KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, pred, output_dict=True)
        auc = roc_auc_score(y_test, proba)
        results[name] = {"report": report, "roc_auc": auc, "model": model}
        print(f"–– {name} ––")
        print(classification_report(y_test, pred))
        print("ROC‑AUC: {:.3f}".format(auc))
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
        print()
    return models, X_train, X_test, y_train, y_test, results

def shap_explain(rf_model, X_train, X_test):
    explainer = shap.Explainer(rf_model, X_train)
    sv = explainer(X_test)
    shap.summary_plot(sv, X_test, plot_type="bar")

def main():
    print("Loading data …")
    df = load_data()
    print("Rows before cleaning:", df.shape[0])
    df = preprocess(df)
    print("Rows after cleaning:", df.shape[0])
    models, X_train, X_test, y_train, y_test, results = train_and_evaluate(df)
    # optional shap for Random Forest
    # shap_explain(models["Random Forest"], X_train, X_test)

if __name__ == "__main__":
 main()
