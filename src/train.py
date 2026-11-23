
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"

TARGET = "kg_per_worker"

# Exclude identifiers, date, and potentially leakage-related column
DROP_COLUMNS = [
    "record_id",
    "date",
    "kg_weighed_at_factory",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET]

    return X, y


def build_pipeline(X, model):

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def main():

    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("Target:", TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    mlflow.set_tracking_uri(
        f"sqlite:///{ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment("Tea-Leaf-Plucking")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for model_name, estimator in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = build_pipeline(X_train, estimator)

        with mlflow.start_run(run_name=model_name) as run:

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test,
            )

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
            )

            results.append({
                "model_name": model_name,
                "run_id": run.info.run_id,
                "pipeline": pipeline,
                **metrics,
            })

            print("Metrics:", metrics)

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON")
    print(
        results_df[
            ["model_name", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )

    best_model = results_df.sort_values(
        "RMSE"
    ).iloc[0]

    print("\nBest model:", best_model["model_name"])
    print("Best RMSE:", best_model["RMSE"])

    best_run_id = best_model["run_id"]

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="tea-leaf-plucking-model",
    )

    print("\nModel registered successfully!")
    print("Registered model name:", registered_model.name)
    print("Registered model version:", registered_model.version)


if __name__ == "__main__":
    main()
