
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "tourism.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

TARGET = "ProdTaken"
DROP_COLUMNS = ["CustomerID"]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# HELPER: REMOVE ACCIDENTAL CSV INDEX COLUMNS
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


# --------------------------------------------------
# MAIN PREPROCESSING
# --------------------------------------------------

def main():

    print("Project root:", PROJECT_ROOT)
    print("Reading dataset:", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Clean column names and remove CSV index columns
    df = remove_unnamed_columns(df)

    # Check target
    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' is missing. "
            f"Available columns: {list(df.columns)}"
        )

    # Remove duplicate records
    duplicate_count = df.duplicated().sum()

    if duplicate_count:
        print("Removing duplicate rows:", duplicate_count)
        df = df.drop_duplicates()

    # Validate target
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise"
    )

    target_values = set(df[TARGET].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain only 0 and 1. "
            f"Found: {target_values}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target '{TARGET}' contains missing values."
        )

    # Drop identifier columns if present
    existing_drop_columns = [
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ]

    if existing_drop_columns:
        df = df.drop(columns=existing_drop_columns)

    # Clean blank strings
    df = df.replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Final validation
    bad_columns = [
        column
        for column in X.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]

    if bad_columns:
        raise ValueError(
            f"Unexpected index columns in features: {bad_columns}"
        )

    print("Feature columns:", list(X.columns))
    print("Number of features:", len(X.columns))
    print("Target distribution:")
    print(y.value_counts())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Create artifact folder
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save without writing row indexes
    X_train.to_csv(
        ARTIFACT_DIR / "Xtrain.csv",
        index=False
    )

    X_test.to_csv(
        ARTIFACT_DIR / "Xtest.csv",
        index=False
    )

    y_train.to_frame(name=TARGET).to_csv(
        ARTIFACT_DIR / "ytrain.csv",
        index=False
    )

    y_test.to_frame(name=TARGET).to_csv(
        ARTIFACT_DIR / "ytest.csv",
        index=False
    )

    print("\nPreprocessing completed successfully.")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Artifacts saved in:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
