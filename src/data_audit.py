import pandas as pd

DATA_PATH = "data/raw/train.csv"


def audit_dataset():
    df = pd.read_csv(DATA_PATH)

    print("\n========== DATA AUDIT ==========\n")

    # 1. Duplicate rows
    print("1. DUPLICATE ROWS")
    print("Duplicate rows:", df.duplicated().sum())
    print(df[df.duplicated(keep=False)].head(20))

    # 2. Duplicate record IDs
    print("\n2. DUPLICATE RECORD IDs")
    print("Duplicate IDs:", df["record_id"].duplicated().sum())
    print(df[df["record_id"].duplicated(keep=False)].sort_values("record_id"))

    # 3. Unique categorical values
    print("\n3. FLUSH VALUES")
    print(df["flush"].value_counts(dropna=False))

    print("\n4. SECTION VALUES")
    print(df["section"].value_counts(dropna=False))

    # 4. Numeric range checks
    print("\n5. NUMERIC RANGES")
    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:
        print(f"\n{col}")
        print("Minimum:", df[col].min())
        print("Maximum:", df[col].max())
        print("Unique values:", df[col].nunique())

    # 5. Suspicious slope
    print("\n6. SUSPICIOUS SLOPE VALUES")
    print(df[df["slope_deg"] < 0])

    # 6. Target and possible leakage investigation
    print("\n7. TARGET VS FACTORY WEIGHT")
    print(
        df[["kg_weighed_at_factory", "kg_per_worker"]]
        .corr()
    )

    print("\nDifference between factory weight and target:")
    difference = (
        df["kg_weighed_at_factory"] - df["kg_per_worker"]
    )
    print(difference.describe())

    print("\nRows where values are nearly identical:")
    print(
        df[
            (df["kg_weighed_at_factory"] - df["kg_per_worker"]).abs()
            < 0.5
        ].head(20)
    )

    # 7. Date investigation
    print("\n8. DATE INVESTIGATION")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("Invalid dates:", df["date"].isna().sum())
    print("Earliest date:", df["date"].min())
    print("Latest date:", df["date"].max())
    print("Unique dates:", df["date"].nunique())

    print("\nRows per date:")
    print(df["date"].value_counts().sort_index().head(20))

    # 8. Target distribution
    print("\n9. TARGET DISTRIBUTION")
    print(df["kg_per_worker"].describe())

    print("\nPotential target outliers:")
    q1 = df["kg_per_worker"].quantile(0.25)
    q3 = df["kg_per_worker"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    print("Lower bound:", lower)
    print("Upper bound:", upper)

    print(
        df[
            (df["kg_per_worker"] < lower)
            | (df["kg_per_worker"] > upper)
        ]
    )


if __name__ == "__main__":
    audit_dataset()