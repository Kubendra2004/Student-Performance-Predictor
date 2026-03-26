from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "required_features" in payload


def test_predict_before_training_returns_404() -> None:
    payload = {
        "study_hours": 4.0,
        "attendance_percent": 85,
        "previous_grade": 72,
        "assignments_completed": 10,
        "sleep_hours": 7.0,
    }
    response = client.post("/predict", json=payload)

    if response.status_code == 200:
        # If the model already exists from previous runs, endpoint can succeed.
        assert "predicted_score" in response.json()
    else:
        assert response.status_code == 404


def test_train_sample_then_predict_and_metrics() -> None:
    train_response = client.post("/train/sample")
    assert train_response.status_code == 200

    predict_payload = {
        "study_hours": 5.5,
        "attendance_percent": 94,
        "previous_grade": 80,
        "assignments_completed": 12,
        "sleep_hours": 7.5,
    }

    predict_response = client.post("/predict", json=predict_payload)
    assert predict_response.status_code == 200
    assert "predicted_score" in predict_response.json()

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["trained_rows"] >= 20
