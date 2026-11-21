import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"level": record.levelname, "event": record.getMessage()})


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp030.predictions")
logger.handlers = [handler]
logger.setLevel(logging.INFO)
logger.propagate = False


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queue_wait_sec: float = Field(..., ge=0)
    agent_exp_months: float = Field(..., le=150)
    previous_calls: float
    issue_count: float
    hour: float
    caller_age: float
    language: str
    team: str


app = FastAPI(title="MP030 Helpline Call Length")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models/model.joblib"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if model is None:
        raise RuntimeError("Model artifact is not available")
    frame = pd.DataFrame([request.model_dump()])
    prediction = float(model.predict(frame)[0])
    logger.info("prediction_generated")
    return {"prediction": prediction}
