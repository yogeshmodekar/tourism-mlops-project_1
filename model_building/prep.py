
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Get the repository root directory.
# This assumes prep.py is inside the model_building folder.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / "tourism_project" / "data" / "tourism.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

TARGET = "ProdTaken"
DROP_COLUMNS = ["CustomerID"]


def main():
    # 1. Check whether the original dataset exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}. "
            "Please add tourism.csv to tourism_project/data/."
        )

    # 2. Load the dataset.
    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # 3. Remove accidental CSV index columns, such as Unnamed: 0.
    # These columns are not genuine model features.
    df.columns = df.columns.astype(str).str.strip()

    unnamed_columns = [
        col for col in df.columns
        if col.strip().lower().startswith("unnamed:")
    ]

    if unnamed_columns:
        print(f"Removing accidental index columns: {unnamed_columns}")
        df = df.drop(columns=unnamed_columns)

    # 4. Check that the target column exists.
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in the dataset."
        )

    # 5. Remove duplicate rows.
    df = df.drop_duplicates().copy()

    # 6. Clean and validate the target column.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # 7. Remove identifier columns that should not be model features.
    df = df.drop(
        columns=[col for col in DROP_COLUMNS if col in df.columns],
        errors="ignore",
    )

    # 8. Convert blank text values to missing values.
    # The training pipeline can impute these missing values later.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .astype(object)
        )

    # 9. Separate features (X) and target (y).
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if X.empty or X.shape[1] == 0:
        raise ValueError("No feature columns remain after data cleaning.")

    if len(X) == 0:
        raise ValueError("No valid rows remain after data cleaning.")

    if y.nunique() < 2:
        raise ValueError("Target must contain both classes 0 and 1.")

    # 10. Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 11. Create the artifacts folder.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 12. Save feature splits without row indexes.
    X_train.to_csv(ARTIFACT_DIR / "Xtrain.csv", index=False)
    X_test.to_csv(ARTIFACT_DIR / "Xtest.csv", index=False)

    # Save target splits.
    y_train.to_frame(name=TARGET).to_csv(
        ARTIFACT_DIR / "ytrain.csv", index=False
    )
    y_test.to_frame(name=TARGET).to_csv(
        ARTIFACT_DIR / "ytest.csv", index=False
    )

    # 13. Print a summary.
    print("\nData preparation completed successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature columns: {X.columns.tolist()}")
    print(f"Saved data splits to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
