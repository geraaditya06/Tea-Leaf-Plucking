import pandas as pd

from src.data_cleaning import clean_dataset


def test_duplicate_rows_are_removed():
    df = pd.DataFrame({
        "record_id": [1, 1],
        "slope_deg": [10, 10],
        "kg_per_worker": [20, 20],
    })

    cleaned = clean_dataset(df)

    assert len(cleaned) == 1


def test_duplicate_record_ids_are_removed():
    df = pd.DataFrame({
        "record_id": [1, 1, 2],
        "slope_deg": [10, 15, 20],
        "kg_per_worker": [20, 25, 30],
    })

    cleaned = clean_dataset(df)

    assert cleaned["record_id"].nunique() == 2
    assert len(cleaned) == 2


def test_negative_slope_is_converted_to_missing():
    df = pd.DataFrame({
        "record_id": [1, 2],
        "slope_deg": [-1, 15],
        "kg_per_worker": [20, 25],
    })

    cleaned = clean_dataset(df)

    assert pd.isna(cleaned.loc[0, "slope_deg"])
    assert cleaned.loc[1, "slope_deg"] == 15


def test_missing_target_rows_are_removed():
    df = pd.DataFrame({
        "record_id": [1, 2],
        "slope_deg": [10, 15],
        "kg_per_worker": [20, None],
    })

    cleaned = clean_dataset(df)

    assert len(cleaned) == 1
    assert cleaned["kg_per_worker"].notna().all()


def test_categorical_values_are_normalized():
    df = pd.DataFrame({
        "record_id": [1],
        "flush": ["  First Flush  "],
        "section": ["  SECTION A  "],
        "slope_deg": [10],
        "kg_per_worker": [20],
    })

    cleaned = clean_dataset(df)

    assert cleaned.loc[0, "flush"] == "first flush"
    assert cleaned.loc[0, "section"] == "section a"