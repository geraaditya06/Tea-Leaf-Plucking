from pathlib import Path
import logging

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================
# CONFIGURATION
# =========================

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "tea-leaf-plucking-model"

MODEL_PATH = ROOT / "mlruns" / "1" / "models" / "m-d504de84954e4225a9d7659d14df5be7" / "artifacts"

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =========================
# FASTAPI APPLICATION
# =========================

app = FastAPI(
    title="Tea Leaf Plucking Prediction API",
    description="API for predicting tea leaf plucking productivity using a machine learning model.",
    version="1.0.0",
)


# =========================
# INPUT SCHEMA
# =========================

class PredictionInput(BaseModel):

    worker_exp_yr: float = Field(ge=0)
    rain_mm_week: float = Field(ge=0)
    slope_deg: float = Field(ge=0, le=90)
    temp_c: float
    bush_age_yr: float = Field(ge=0)
    hours_worked: float = Field(gt=0, le=24)

    flush: str
    section: str


# =========================
# LOAD MODEL
# =========================

@app.on_event("startup")
def load_model():
    global model

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        logger.info("MLflow model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load MLflow model")
        raise RuntimeError(f"Could not load MLflow model: {e}")


# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")
def home():

    return {
        "message": "Tea Leaf Plucking Prediction API",
        "status": "running",
        "version": "1.0.0",
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# =========================
# PREDICTION ENDPOINT
# =========================

@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:

        input_data = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_data)

        predicted_value = float(prediction[0])

        logger.info(
            "Prediction successful: %.4f kg per worker",
            predicted_value,
        )

        return {
            "prediction": predicted_value,
            "unit": "kg_per_worker",
            "message": "Prediction successful",
        }

    except Exception as e:

        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )
