import json
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("tourism_project/deployment")
TARGET = "ProdTaken"

def main():
    needed = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
    missing = [f for f in needed if not (ARTIFACT_DIR / f).exists()]
    if missing:
        # Workflow download-artifact puts files in the working directory.
        if all(Path(f).exists() for f in needed):
            data_dir = Path(".")
        else:
            raise FileNotFoundError(f"Missing split files: {missing}. Run prep.py first.")
    else:
        data_dir = ARTIFACT_DIR

    X_train = pd.read_csv(data_dir / "Xtrain.csv")
    X_test = pd.read_csv(data_dir / "Xtest.csv")
    y_train = pd.read_csv(data_dir / "ytrain.csv")[TARGET].astype(int)
    y_test = pd.read_csv(data_dir / "ytest.csv")[TARGET].astype(int)

    categorical = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in X_train.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1))
    ])
    param_grid = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(model, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, verbose=1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("tourism-purchase-prediction")
    with mlflow.start_run(run_name="random-forest-grid-search"):
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        probs = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }
        if probs is not None and y_test.nunique() == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics(metrics)
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        print("Best parameters:", search.best_params_)
        print("Test metrics:", json.dumps(metrics, indent=2))
        print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
        mlflow.sklearn.log_model(best, artifact_path="model")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    job_path = MODEL_DIR / "tourism_model.joblib"
    joblib.dump(best, job_path)
    metadata = {
        "target": TARGET,
        "features": X_train.columns.tolist(),
        "best_params": search.best_params_,
        "test_metrics": metrics,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best model to {job_path}")

if __name__ == "__main__":
    main()
