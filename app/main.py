from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.model_service import FEATURE_COLUMNS, ModelService
from app.schemas import MetricsResponse, PredictionResponse, StudentFeatures, TrainingResponse


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_training_data.csv"

model_service = ModelService(MODEL_DIR)

app = FastAPI(
    title="Student Performance Predictor API",
    version="1.0.0",
    description="Predict student final scores from learning behavior features.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_ready": model_service.has_model(),
        "required_features": FEATURE_COLUMNS,
    }


@app.post("/train/sample", response_model=TrainingResponse)
def train_with_sample_data() -> TrainingResponse:
    if not SAMPLE_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found.")

    try:
        training_result = model_service.train_from_csv(SAMPLE_DATA_PATH)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return TrainingResponse(
        message="Model trained successfully with sample dataset.",
        rows_used=training_result.trained_rows,
    )


@app.post("/train/upload", response_model=TrainingResponse)
async def train_with_uploaded_csv(file: UploadFile = File(...)) -> TrainingResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        content = await file.read()
        with NamedTemporaryFile(delete=True, suffix=".csv") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            dataframe = pd.read_csv(tmp_file.name)
        training_result = model_service.train_from_dataframe(dataframe)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Training failed: {error}") from error

    return TrainingResponse(
        message="Model trained successfully with uploaded dataset.",
        rows_used=training_result.trained_rows,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_score(features: StudentFeatures) -> PredictionResponse:
    try:
        predicted_score = model_service.predict(features.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return PredictionResponse(predicted_score=predicted_score)


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    try:
        metrics = model_service.get_metrics()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return MetricsResponse(**metrics)
