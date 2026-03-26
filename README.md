# Student Performance Predictor API + Frontend

A full Python project that predicts student final score using learning behavior features.

## What This Project Includes

- FastAPI backend for training and prediction
- ML model pipeline (RandomForestRegressor)
- Sample dataset (350 rows) for quick training
- Browser frontend that calls backend APIs
- Persistent model artifacts and metrics
- Docker and docker-compose setup
- Pytest API test suite

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- scikit-learn, pandas, numpy, joblib
- pytest for tests
- Frontend: HTML, CSS, Vanilla JavaScript (Fetch API)
- Containers: Docker + docker-compose + Nginx (for static frontend)

## Project Structure

```text
Student Performance Predictor/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model_service.py
│   └── schemas.py
├── data/
│   └── sample_training_data.csv
├── frontend/
│   ├── app.js
│   ├── index.html
│   ├── nginx.conf
│   └── style.css
├── tests/
│   └── test_api.py
├── models/
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## API Endpoints

- `GET /health`
  - Checks service status and model availability.

- `POST /train/sample`
  - Trains model using built-in sample CSV.

- `POST /train/upload`
  - Trains model from uploaded CSV file.
  - File field name: `file`

- `POST /predict`
  - Predicts final score for one student.

- `GET /metrics`
  - Returns latest MAE, RMSE, and R2 from training.

## Required CSV Columns

- `study_hours`
- `attendance_percent`
- `previous_grade`
- `assignments_completed`
- `sleep_hours`
- `final_score`

## Local Setup (Inside This Folder Only)

All commands below are executed from this project directory only and install dependencies in a local virtual environment (`.venv`) inside this folder.

```bash
cd "/mnt/kubendra/Project Programs/Student Performance Predictor"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Backend

1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start API

```bash
uvicorn app.main:app --reload
```

4. Open Swagger docs

- http://127.0.0.1:8000/docs

## Run Frontend

Open the file below in your browser:

- `frontend/index.html`

The default API base URL is already set to `http://127.0.0.1:8000`.

## Frontend Working Status

Yes, the frontend is wired to the API and working from a code perspective:

- Calls `GET /health`
- Calls `POST /train/sample`
- Calls `POST /train/upload`
- Calls `POST /predict`
- Calls `GET /metrics`
- Includes interactive score gauge, quick student profile presets, live metric cards, and activity console logging

If backend is running on `http://127.0.0.1:8000`, the UI can train and predict immediately.

## Run Tests

```bash
cd "/mnt/kubendra/Project Programs/Student Performance Predictor"
source .venv/bin/activate
pytest
```

## Run with Docker

```bash
cd "/mnt/kubendra/Project Programs/Student Performance Predictor"
docker compose up --build
```

Services:

- Backend API: http://127.0.0.1:8000
- Frontend UI: http://127.0.0.1:8080

## Frontend Use Cases

1. Health check
- Click `Check Health` to verify API and model status.

2. Quick training
- Click `Train with Sample Data` for instant setup.

3. Custom training
- Upload your CSV and click `Train with Uploaded CSV`.

4. Prediction
- Fill student features and submit `Predict`.

5. Metrics review
- Click `Fetch Metrics` to inspect model quality.

## Update Log

This section tracks implementation updates as the project evolves.

### 2026-03-26

- Initialized complete project scaffold.
- Implemented FastAPI backend with `/health`, `/train/sample`, `/train/upload`, `/predict`, `/metrics`.
- Added model persistence and training metrics storage.
- Added sample CSV dataset for instant training.
- Built frontend dashboard that calls all backend APIs.
- Added setup and execution instructions.
- Added Dockerfile, docker-compose, and Nginx frontend serving.
- Added pytest test suite and pytest config.
- Expanded sample training dataset to 350 rows with realistic variation.
- Redesigned frontend with professional visual style, motion effects, and richer interactions.

## Next Suggested Improvements

- Add authentication and role-based access for training endpoints.
- Add dataset drift checks and model versioning.
- Add automated tests with pytest.
- Add Docker and docker-compose for one-command startup.
