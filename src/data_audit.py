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
        # Spelling variant investigation

    print("\n3. FLUSH SPELLING VARIANTS")

    print("Raw flush values:")
    print(df["flush"].value_counts(dropna=False))

    flush_normalized = (
        df["flush"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    flush_mapping = {
        "fir.": "first",
        "mon.": "monsoon",
        "sec.": "second",
    }

    flush_normalized = flush_normalized.replace(flush_mapping)

    print("\nNormalized flush values:")
    print(flush_normalized.value_counts(dropna=False))

    print("\nSection values:")
    print(df["section"].value_counts(dropna=False))

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
        # Unit mix-up investigation

    print("\nUNIT MIX-UP INVESTIGATION")

    comparison_columns = [
        "worker_exp_yr",
        "rain_mm_week",
        "slope_deg",
        "temp_c",
        "bush_age_yr",
        "hours_worked",
    ]

    print(
        df.groupby("section")[comparison_columns]
        .agg(["min", "median", "max", "std"])
        .round(2)
        .to_string()
    )

    # 5. Sentinel value investigation

    print("\n6. SENTINEL VALUE INVESTIGATION")

    sentinel_count = (df["slope_deg"] == -1).sum()

    print("slope_deg == -1 count:", sentinel_count)
    print("Percentage:", round(sentinel_count / len(df) * 100, 2), "%")

    print("\nRows containing sentinel value:")
    print(df[df["slope_deg"] == -1])

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
    # Target typo investigation

    print("\nLOWEST 15 TARGET VALUES")

    target_columns = [
        "record_id",
        "date",
        "section",
        "kg_weighed_at_factory",
        "kg_per_worker",
    ]

    print(
        df.nsmallest(15, "kg_per_worker")[target_columns]
        .to_string(index=False)
    )

    print("\nHIGHEST 15 TARGET VALUES")

    print(
        df.nlargest(15, "kg_per_worker")[target_columns]
        .to_string(index=False)
    )
    # 9. Stuck sensor investigation

    print("\n10. STUCK SENSOR INVESTIGATION")

    sensor_columns = [
        "worker_exp_yr",
        "rain_mm_week",
        "slope_deg",
        "temp_c",
        "bush_age_yr",
        "hours_worked",
    ]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    suspicious_streaks = []

    for section in df["section"].dropna().unique():

        section_df = df[df["section"] == section]

        for column in sensor_columns:

            daily_values = (
                section_df.groupby("date")[column]
                .median()
                .dropna()
                .sort_index()
            )

            streak_start = None
            streak_value = None
            streak_length = 0
            previous_date = None

            for date, value in daily_values.items():

                if (
                    streak_value == value
                    and previous_date is not None
                    and (date - previous_date).days == 1
                ):
                    streak_length += 1

                else:
                    if streak_length >= 3:
                        suspicious_streaks.append(
                            {
                                "section": section,
                                "feature": column,
                                "value": streak_value,
                                "start_date": streak_start,
                                "end_date": previous_date,
                                "days": streak_length,
                            }
                        )

                    streak_start = date
                    streak_value = value
                    streak_length = 1

                previous_date = date

            if streak_length >= 3:
                suspicious_streaks.append(
                    {
                        "section": section,
                        "feature": column,
                        "value": streak_value,
                        "start_date": streak_start,
                        "end_date": previous_date,
                        "days": streak_length,
                    }
                )

    if suspicious_streaks:
        print(
            pd.DataFrame(suspicious_streaks)
            .sort_values("days", ascending=False)
            .to_string(index=False)
        )
    else:
        print("No suspicious 3-or-more-day repeated median readings found.")

if __name__ == "__main__":
    audit_dataset()
