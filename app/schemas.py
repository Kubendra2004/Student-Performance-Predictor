from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    study_hours: float = Field(..., ge=0, le=24)
    attendance_percent: float = Field(..., ge=0, le=100)
    previous_grade: float = Field(..., ge=0, le=100)
    assignments_completed: int = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=24)


class PredictionResponse(BaseModel):
    predicted_score: float


class TrainingResponse(BaseModel):
    message: str
    rows_used: int


class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    r2: float
    trained_rows: int
