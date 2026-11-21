# G100 — MP030 Helpline Call Length

SCSE3040 MLOps mini project for predicting `handle_min`, the duration of a
Lucknow Citizen Helpline call.

## Group

- S24CSEU2174 — Krish Singh
- S24CSEU2336 — Angad Deval
- E23CSEU1145 — Mayank Sehrawat

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
dvc pull
```

## Run

```bash
python -m src.mp030.audit
python -m src.mp030.train
python -m src.mp030.monitor
pytest
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API exposes `GET /health` and `POST /predict`. Training experiments are
stored in the committed SQLite MLflow database `mlflow.db`.
