import argparse
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tourism_project/data/tourism.csv")
    args = parser.parse_args()
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Add tourism.csv to tourism_project/data/.")
    df = pd.read_csv(path)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}\nFound: {list(df.columns)}")
    if "ProdTaken" not in df:
        raise ValueError("Target column ProdTaken is required.")
    print("DATA REGISTRATION PASSED")
    print(f"Path: {path} | Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("\nColumn types:\n", df.dtypes.to_string())
    print("\nMissing values (top):\n", df.isna().sum().sort_values(ascending=False).head(10).to_string())
    print("\nTarget distribution:\n", df["ProdTaken"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
