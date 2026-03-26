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
    version_id: str


class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    r2: float
    trained_rows: int


class ModelVersionMetadata(BaseModel):
    version_id: str
    created_at: str
    trained_rows: int
    mae: float
    rmse: float
    r2: float
    model_path: str


class ModelVersionsResponse(BaseModel):
    active_version_id: str | None
    versions: list[ModelVersionMetadata]


class CsvValidationResponse(BaseModel):
    total_rows: int
    missing_columns: list[str]
    null_counts: dict[str, int]
    duplicate_rows: int
    out_of_range_counts: dict[str, int]
    is_valid_for_training: bool
    messages: list[str]
