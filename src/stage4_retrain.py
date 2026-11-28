
# Stage 4 - Chronological Retraining and Model Evaluation

from pathlib import Path
import pandas as pd
import mlflow
import mlflow.pyfunc

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.train import build_pipeline, PipelineModel

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"

TARGET = "kg_per_worker"
DROP_COLUMNS = [
    "record_id",
    "date",
    "kg_weighed_at_factory",
]

MODEL_NAME = "tea-leaf-plucking-model"


def main():
    # Load original cleaned training data
    original = pd.read_csv(TRAIN_PATH)

    # Load and chronologically sort later data
    later = pd.read_csv(LATER_PATH)
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date", kind="stable").reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    check = later.iloc[split_index:].copy()

    # Prepare combined v2 training dataset
    combined = pd.concat(
        [original, later_train],
        ignore_index=True
    )

    X_train = combined.drop(
        columns=[TARGET] + DROP_COLUMNS,
        errors="ignore"
    )
    y_train = combined[TARGET]

    X_check = check.drop(
        columns=[TARGET] + DROP_COLUMNS,
        errors="ignore"
    )
    y_check = check[TARGET]

    print("Original training rows:", len(original))
    print("Later training rows:", len(later_train))
    print("Combined v2 training rows:", len(combined))
    print("Check rows:", len(check))
    print("Check dates:", check["date"].min(), "to", check["date"].max())

    # Train v2 using the same algorithm as the existing v1 model
    estimator = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    pipeline = build_pipeline(X_train, estimator)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_check)
    v2_mae = mean_absolute_error(y_check, predictions)

    print("\nV2 MAE ON CHRONOLOGICAL CHECK SET:", v2_mae)

    # Log and register v2 in MLflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("Tea-Leaf-Plucking")

    with mlflow.start_run(run_name="Stage4_GradientBoosting_V2") as run:
        mlflow.log_params({
            "model_name": "GradientBoosting_V2",
            "training_rows_original": len(original),
            "training_rows_later": len(later_train),
            "total_training_rows": len(combined),
            "check_rows": len(check),
            "chronological_split": 0.70,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 42,
            "target": TARGET,
        })

        mlflow.log_metric("check_MAE", v2_mae)

        mlflow.pyfunc.log_model(
            name="model",
            python_model=PipelineModel(pipeline),
        )

        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\nV2 REGISTERED SUCCESSFULLY")
    print("Run ID:", run_id)
    print("Model name:", registered.name)
    print("Model version:", registered.version)
    print("V2 check MAE:", v2_mae)


if __name__ == "__main__":
    main()

