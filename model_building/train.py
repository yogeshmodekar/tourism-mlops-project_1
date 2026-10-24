
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "deployment"

TARGET = "ProdTaken"
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def remove_unnamed_columns(df):
    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    unwanted = [
        column
        for column in df.columns
        if column.lower().startswith("unnamed:")
    ]

    if unwanted:
        print("Removing accidental columns:", unwanted)
        df = df.drop(columns=unwanted)

    return df


def read_csv_artifact(filename):
    path = ARTIFACT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required training artifact not found: {path}. "
            "Run model_building/prep.py first."
        )

    return pd.read_csv(path)


def load_target(filename):
    df = read_csv_artifact(filename)

    if TARGET in df.columns:
        if df.shape[1] != 1:
            raise ValueError(
                f"{filename} must contain only the target column."
            )
        return df[TARGET]

    if df.shape[1] == 1:
        return df.iloc[:, 0]

    raise ValueError(
        f"Cannot identify target column in {filename}."
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Project root:", PROJECT_ROOT)
    print("Artifact directory:", ARTIFACT_DIR)
    print("Model directory:", MODEL_DIR)

    X_train = read_csv_artifact("Xtrain.csv")
    X_test = read_csv_artifact("Xtest.csv")

    y_train = load_target("ytrain.csv")
    y_test = load_target("ytest.csv")

    # Remove unwanted columns before fitting
    X_train = remove_unnamed_columns(X_train)
    X_test = remove_unnamed_columns(X_test)

    # Validate feature columns
    bad_columns = [
        column
        for column in X_train.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]

    if bad_columns:
        raise ValueError(
            f"Unexpected index columns remain in training data: "
            f"{bad_columns}"
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match.\n"
            f"Training: {list(X_train.columns)}\n"
            f"Testing: {list(X_test.columns)}"
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training feature and target row counts do not match."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Testing feature and target row counts do not match."
        )

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    y_train = pd.to_numeric(y_train, errors="raise").astype(int)
    y_test = pd.to_numeric(y_test, errors="raise").astype(int)

    # Determine categorical and numeric columns from actual data
    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X_train.columns
    ]

    missing_categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column not in X_train.columns
    ]

    if missing_categorical:
        print(
            "Configured categorical columns not present:",
            missing_categorical
        )

    numeric_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]

    # Normalize feature types
    for column in numeric_columns:
        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce"
        )

        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce"
        )

    for column in categorical_columns:
        X_train[column] = X_train[column].astype("object")
        X_test[column] = X_test[column].astype("object")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("Numeric columns:", numeric_columns)
    print("Categorical columns:", categorical_columns)

    # --------------------------------------------------
    # PREPROCESSING PIPELINE
    # --------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1
                )
            ),
        ]
    )

    # --------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    # --------------------------------------------------
    # TRAIN AND EVALUATE
    # --------------------------------------------------

    mlflow.set_experiment("Tourism_Prediction")

    with mlflow.start_run():

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print("\nBest parameters:", search.best_params_)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="tourism_model"
        )

        # --------------------------------------------------
        # SAVE MODEL
        # --------------------------------------------------

        model_path = MODEL_DIR / "tourism_model.joblib"

        joblib.dump(
            best_model,
            model_path
        )

        # Reload saved model to verify actual file
        saved_model = joblib.load(model_path)

        saved_features = list(
            saved_model.feature_names_in_
        )

        bad_features = [
            column
            for column in saved_features
            if str(column).strip().lower().startswith("unnamed:")
        ]

        if bad_features:
            raise ValueError(
                f"Saved model contains unwanted columns: "
                f"{bad_features}"
            )

        print("\nSaved model:", model_path.resolve())
        print("Saved model input features:", saved_features)
        print("Saved model verification passed.")

        # Save metadata
        metadata = {
            "target": TARGET,
            "model_type": "RandomForestClassifier",
            "feature_columns": saved_features,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "best_parameters": search.best_params_,
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            },
        }

        metadata_path = MODEL_DIR / "model_metadata.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
                default=str
            )

        print("Metadata saved:", metadata_path.resolve())


if __name__ == "__main__":
    main()
