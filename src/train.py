
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.pyfunc

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# CONFIGURATION
# =========================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "train_clean.csv"

TARGET = "kg_per_worker"

DROP_COLUMNS = [
    "record_id",
    "date",
    "kg_weighed_at_factory",
]


# =========================
# MLFLOW PYFUNC WRAPPER
# =========================

class PipelineModel(mlflow.pyfunc.PythonModel):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input, params=None):
        return self.pipeline.predict(model_input)


# =========================
# LOAD DATA
# =========================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=[TARGET] + DROP_COLUMNS,
        errors="ignore",
    )

    y = df[TARGET]

    return X, y


# =========================
# PREPROCESSING
# =========================

def build_pipeline(X, estimator):

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])

    return pipeline


# =========================
# EVALUATION
# =========================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# =========================
# MAIN TRAINING
# =========================

def main():

    # Load dataset
    X, y = load_data()

    print("Dataset shape:", X.shape)
    print("Target:", TARGET)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # MLflow configuration
    mlflow.set_tracking_uri(
        f"sqlite:///{ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "Tea-Leaf-Plucking"
    )

    # Models
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

    # Train models
    for model_name, estimator in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = build_pipeline(
            X_train,
            estimator,
        )

        with mlflow.start_run(
            run_name=model_name
        ) as run:

            # Train
            pipeline.fit(
                X_train,
                y_train,
            )

            # Evaluate
            metrics = evaluate_model(
                pipeline,
                X_test,
                y_test,
            )

            # Log parameters
            mlflow.log_params({
                "model_name": model_name,
                "test_size": 0.2,
                "random_state": 42,
                "n_estimators": 200,
                "target": TARGET,
            })

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model through PyFunc
            mlflow.pyfunc.log_model(
                name="model",
                python_model=PipelineModel(pipeline),
            )

            # Save results
            results.append({
                "model_name": model_name,
                "run_id": run.info.run_id,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            })

            print("Metrics:", metrics)

    # =========================
    # MODEL COMPARISON
    # =========================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    print(
        results_df[
            ["model_name", "MAE", "RMSE", "R2"]
        ].to_string(index=False)
    )

    # Select best model using RMSE
    best_model = results_df.sort_values(
        by="RMSE",
        ascending=True,
    ).iloc[0]

    print("\nBest model:", best_model["model_name"])
    print("Best RMSE:", best_model["RMSE"])

    # =========================
    # MODEL REGISTRY
    # =========================

    best_run_id = best_model["run_id"]

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="tea-leaf-plucking-model",
    )

    print("\n" + "=" * 50)
    print("MODEL REGISTRATION SUCCESSFUL")
    print("=" * 50)

    print("Registered model:", registered_model.name)
    print("Registered version:", registered_model.version)


if __name__ == "__main__":
    main()
