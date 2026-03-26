from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "study_hours",
    "attendance_percent",
    "previous_grade",
    "assignments_completed",
    "sleep_hours",
]
TARGET_COLUMN = "final_score"


@dataclass
class TrainingResult:
    mae: float
    rmse: float
    r2: float
    trained_rows: int


class ModelService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_path = model_dir / "student_predictor.joblib"
        self.metrics_path = model_dir / "metrics.joblib"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def has_model(self) -> bool:
        return self.model_path.exists()

    def train_from_dataframe(self, dataframe: pd.DataFrame) -> TrainingResult:
        self._validate_dataframe(dataframe)

        x = dataframe[FEATURE_COLUMNS]
        y = dataframe[TARGET_COLUMN]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "trained_rows": int(len(dataframe)),
        }

        joblib.dump(model, self.model_path)
        joblib.dump(metrics, self.metrics_path)

        return TrainingResult(**metrics)

    def train_from_csv(self, csv_path: Path) -> TrainingResult:
        dataframe = pd.read_csv(csv_path)
        return self.train_from_dataframe(dataframe)

    def predict(self, features: Dict[str, float]) -> float:
        if not self.has_model():
            raise FileNotFoundError("Model is not trained yet.")

        model = joblib.load(self.model_path)
        ordered_features = [[features[column] for column in FEATURE_COLUMNS]]
        prediction = float(model.predict(ordered_features)[0])
        return round(prediction, 2)

    def get_metrics(self) -> Dict[str, float]:
        if not self.metrics_path.exists():
            raise FileNotFoundError("Metrics are not available. Train the model first.")
        return joblib.load(self.metrics_path)

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = [column for column in expected_columns if column not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        if len(dataframe) < 20:
            raise ValueError("At least 20 rows are required for training.")

        null_counts = dataframe[expected_columns].isnull().sum()
        if int(null_counts.sum()) > 0:
            raise ValueError("Input dataset has null values in required columns.")
