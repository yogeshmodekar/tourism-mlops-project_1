from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
ARTIFACT_DIR = Path("artifacts")
TARGET = "ProdTaken"
DROP_COLUMNS = ["CustomerID"]  # Identifier only; not a predictive feature.

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Add the original tourism.csv file.")
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    # Convert blank strings to missing values and let the model pipeline impute them.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA).astype(object)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    if y.nunique() < 2:
        raise ValueError("Target must contain both classes 0 and 1.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(ARTIFACT_DIR / "Xtrain.csv", index=False)
    X_test.to_csv(ARTIFACT_DIR / "Xtest.csv", index=False)
    y_train.to_frame(TARGET).to_csv(ARTIFACT_DIR / "ytrain.csv", index=False)
    y_test.to_frame(TARGET).to_csv(ARTIFACT_DIR / "ytest.csv", index=False)
    print(f"Preparation complete: train={len(X_train)}, test={len(X_test)}, features={X.shape[1]}")
    print(f"Saved splits to {ARTIFACT_DIR.resolve()}")

if __name__ == "__main__":
    main()
