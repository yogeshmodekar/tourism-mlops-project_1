
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "deployment"

TARGET = "ProdTaken"

# Explicitly identify categorical columns in this dataset.
CATEGORICAL_COLUMNS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]


def remove_unnamed_columns(df):
    """Remove accidental CSV index columns such as Unnamed: 0."""
    df.columns = df.columns.astype(str).str.strip()

    unwanted = [
        col for col in df.columns
        if col.strip().lower().startswith("unnamed:")
    ]

    if unwanted:
        print(f"Removing accidental columns: {unwanted}")
        df = df.drop(columns=unwanted)

    return df


def main():
    # --------------------------------------------------
    # 2. Find the prepared data files
    # --------------------------------------------------
    filenames = [
        "Xtrain.csv",
        "Xtest.csv",
        "ytrain.csv",
        "ytest.csv",
    ]

    if all((ARTIFACT_DIR / name).exists() for name in filenames):
        data_dir = ARTIFACT_DIR
    elif all((PROJECT_ROOT / name).exists() for name in filenames):
        data_dir = PROJECT_ROOT
    else:
        missing = [
            name for name in filenames
            if not (ARTIFACT_DIR / name).exists()
            and not (PROJECT_ROOT / name).exists()
        ]
        raise FileNotFoundError(
            f"Missing prepared data files: {missing}. Run prep.py first."
        )

    # --------------------------------------------------
    # 3. Load training and testing data
    # --------------------------------------------------
    X_train = pd.read_csv(data_dir / "Xtrain.csv")
    X_test = pd.read_csv(data_dir / "Xtest.csv")

    y_train_df = pd.read_csv(data_dir / "ytrain.csv")
    y_test_df = pd.read_csv(data_dir / "ytest.csv")

    if TARGET not in y_train_df.columns or TARGET not in y_test_df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from target files.")

    y_train = pd.to_numeric(y_train_df[TARGET], errors="raise").astype(int)
    y_test = pd.to_numeric(y_test_df[TARGET], errors="raise").astype(int)

    # --------------------------------------------------
    # 4. Remove accidental index columns
    # --------------------------------------------------
    X_train = remove_unnamed_columns(X_train)
    X_test = remove_unnamed_columns(X_test)

    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("No training features remain after cleaning.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing features do not match.\n"
            f"Training: {X_train.columns.tolist()}\n"
            f"Testing: {X_test.columns.tolist()}"
        )

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature rows and target rows do not match.")

    # --------------------------------------------------
    # 5. Set categorical and numeric features
    # --------------------------------------------------
    missing_categorical = [
        col for col in CATEGORICAL_COLUMNS
        if col not in X_train.columns
    ]

    if missing_categorical:
        raise ValueError(
            f"Expected categorical columns are missing: {missing_categorical}"
        )

    categorical = CATEGORICAL_COLUMNS
    numeric = [
        col for col in X_train.columns
        if col not in categorical
    ]

    # Ensure numeric columns contain numeric values.
    for col in numeric:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Ensure categorical columns are treated as text.
    for col in categorical:
        X_train[col] = X_train[col].astype("string").astype(object)
        X_test[col] = X_test[col].astype("string").astype(object)

    print("Training data directory:", data_dir)
    print("Training features:", X_train.columns.tolist())
    print("Numeric features:", numeric)
    print("Categorical features:", categorical)

    # --------------------------------------------------
    # 6. Create preprocessing pipeline
    # --------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------
    # 7. Create model pipeline
    # --------------------------------------------------
    model = Pipeline([
        ("preprocess", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ])

    param_grid = {
        "classifier__n_estimators": [150, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    # --------------------------------------------------
    # 8. Train and evaluate model
    # --------------------------------------------------
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("tourism-purchase-prediction")

    with mlflow.start_run(run_name="random-forest-grid-search"):
        search.fit(X_train, y_train)

        best = search.best_estimator_
        predictions = best.predict(X_test)
        probabilities = best.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
        }

        if y_test.nunique() == 2:
            metrics["roc_auc"] = float(
                roc_auc_score(y_test, probabilities)
            )

        mlflow.log_params({
            key: str(value)
            for key, value in search.best_params_.items()
        })
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics(metrics)
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", len(X_train.columns))

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nTest metrics:")
        print(json.dumps(metrics, indent=2))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))

        mlflow.sklearn.log_model(best, artifact_path="model")

    # --------------------------------------------------
    # 9. Save trained model and metadata
    # --------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    job_path = MODEL_DIR / "tourism_model.joblib"
    joblib.dump(best, job_path)

    metadata = {
        "target": TARGET,
        "features": X_train.columns.tolist(),
        "best_params": {
            key: str(value)
            for key, value in search.best_params_.items()
        },
        "test_metrics": metrics,
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nModel training completed successfully.")
    print(f"Saved model to: {job_path}")
    print(f"Saved metadata to: {metadata_path}")
    print("Saved model features:", best.feature_names_in_.tolist())


if __name__ == "__main__":
    main()
