
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "kg_per_worker"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the tea leaf plucking dataset."""

    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Remove duplicate record IDs, keeping the first occurrence
    if "record_id" in df.columns:
        df = df.drop_duplicates(subset=["record_id"], keep="first")

    # 3. Convert date column to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )

    # 4. Normalize categorical text
    categorical_columns = ["flush", "section"]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    # 5. Standardize known flush abbreviations
    if "flush" in df.columns:
        flush_mapping = {
            "fir.": "first",
            "mon.": "monsoon",
            "sec.": "second",
        }

        df["flush"] = df["flush"].replace(flush_mapping)

    # 6. Treat invalid negative slope values as missing
    if "slope_deg" in df.columns:
        df.loc[df["slope_deg"] < 0, "slope_deg"] = pd.NA

    # 7. Remove rows with missing target values
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])

    # 8. Reset index
    df = df.reset_index(drop=True)
    return df

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = RAW_DIR / "train.csv"

    df = pd.read_csv(train_path)

    print("Original dataset shape:", df.shape)

    cleaned_df = clean_dataset(df)

    print("Cleaned dataset shape:", cleaned_df.shape)
    print("Remaining duplicates:", cleaned_df.duplicated().sum())
    print("Missing values:")
    print(cleaned_df.isnull().sum())

    output_path = PROCESSED_DIR / "train_clean.csv"

    cleaned_df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
