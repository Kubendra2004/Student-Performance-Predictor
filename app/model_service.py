from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import UTC, datetime
from typing import Dict

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
    version_id: str


class ModelService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.registry_path = model_dir / "model_registry.json"
        self.active_model_path = model_dir / "active_model.json"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")

    def has_model(self) -> bool:
        try:
            self.get_active_metadata()
            return True
        except FileNotFoundError:
            return False

    def train_from_dataframe(self, dataframe: pd.DataFrame) -> TrainingResult:
        report = self.build_validation_report(dataframe)
        self._validate_training_report(report)

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

        version_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        model_path = self.model_dir / f"student_predictor_{version_id}.joblib"
        created_at = datetime.now(UTC).isoformat()

        metadata = {
            "version_id": version_id,
            "created_at": created_at,
            "trained_rows": metrics["trained_rows"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "model_path": str(model_path),
        }

        joblib.dump(model, model_path)
        self._append_registry(metadata)
        self.active_model_path.write_text(
            json.dumps({"version_id": version_id}, indent=2),
            encoding="utf-8",
        )

        return TrainingResult(**metrics, version_id=version_id)

    def train_from_csv(self, csv_path: Path) -> TrainingResult:
        dataframe = pd.read_csv(csv_path)
        return self.train_from_dataframe(dataframe)

    def predict(self, features: Dict[str, float]) -> float:
        if not self.has_model():
            raise FileNotFoundError("Model is not trained yet.")

        active_metadata = self.get_active_metadata()
        model = joblib.load(active_metadata["model_path"])
        ordered_features = pd.DataFrame(
            [{column: features[column] for column in FEATURE_COLUMNS}],
            columns=FEATURE_COLUMNS,
        )
        prediction = float(model.predict(ordered_features)[0])
        return round(prediction, 2)

    def get_metrics(self) -> Dict[str, float]:
        active_metadata = self.get_active_metadata()
        return {
            "mae": active_metadata["mae"],
            "rmse": active_metadata["rmse"],
            "r2": active_metadata["r2"],
            "trained_rows": active_metadata["trained_rows"],
        }

    def list_versions(self) -> list[Dict[str, float | int | str]]:
        registry = self._load_registry()
        return list(reversed(registry))

    def get_active_metadata(self) -> Dict[str, float | int | str]:
        if not self.active_model_path.exists():
            raise FileNotFoundError("No active model metadata found.")

        active_data = json.loads(self.active_model_path.read_text(encoding="utf-8"))
        version_id = active_data.get("version_id")
        if not version_id:
            raise FileNotFoundError("Active model version is not set.")

        registry = self._load_registry()
        for item in registry:
            if item.get("version_id") == version_id:
                return item

        raise FileNotFoundError("Active model metadata not present in registry.")

    @staticmethod
    def build_validation_report(dataframe: pd.DataFrame) -> Dict[str, object]:
        expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = [column for column in expected_columns if column not in dataframe.columns]

        available_columns = [column for column in expected_columns if column in dataframe.columns]
        null_counts = (
            dataframe[available_columns].isnull().sum().astype(int).to_dict()
            if available_columns
            else {}
        )

        ranges = {
            "study_hours": (0, 24),
            "attendance_percent": (0, 100),
            "previous_grade": (0, 100),
            "assignments_completed": (0, 100),
            "sleep_hours": (0, 24),
            "final_score": (0, 100),
        }

        out_of_range_counts: dict[str, int] = {}
        for column, (minimum, maximum) in ranges.items():
            if column in dataframe.columns:
                invalid = int(((dataframe[column] < minimum) | (dataframe[column] > maximum)).sum())
                out_of_range_counts[column] = invalid

        duplicate_rows = int(dataframe.duplicated().sum())
        total_rows = int(len(dataframe))

        messages: list[str] = []
        if missing_columns:
            messages.append(f"Missing required columns: {', '.join(missing_columns)}")
        if total_rows < 20:
            messages.append("At least 20 rows are required for training.")
        null_total = int(sum(null_counts.values()))
        if null_total > 0:
            messages.append("Null values found in required columns.")
        out_of_range_total = int(sum(out_of_range_counts.values()))
        if out_of_range_total > 0:
            messages.append("Some values are outside allowed ranges.")
        if duplicate_rows > 0:
            messages.append("Duplicate rows detected.")

        is_valid_for_training = (
            not missing_columns
            and total_rows >= 20
            and null_total == 0
            and out_of_range_total == 0
        )

        if not messages:
            messages.append("Dataset looks good for training.")

        return {
            "total_rows": total_rows,
            "missing_columns": missing_columns,
            "null_counts": null_counts,
            "duplicate_rows": duplicate_rows,
            "out_of_range_counts": out_of_range_counts,
            "is_valid_for_training": is_valid_for_training,
            "messages": messages,
        }

    def _append_registry(self, metadata: Dict[str, float | int | str]) -> None:
        registry = self._load_registry()
        registry.append(metadata)
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def _load_registry(self) -> list[Dict[str, float | int | str]]:
        if not self.registry_path.exists():
            return []
        text = self.registry_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return data

    @staticmethod
    def _validate_training_report(report: Dict[str, object]) -> None:
        if not bool(report.get("is_valid_for_training", False)):
            messages = report.get("messages", [])
            if isinstance(messages, list) and messages:
                raise ValueError(" ".join(str(message) for message in messages))
            raise ValueError("Dataset is not valid for training.")
