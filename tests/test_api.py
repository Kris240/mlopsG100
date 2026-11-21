from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_invalid_inputs_return_422():
    payload = {"queue_wait_sec": -1, "agent_exp_months": 151, "previous_calls": 2, "issue_count": 1, "hour": 10, "caller_age": 30, "language": "hindi", "team": "Team Alpha"}
    assert client.post("/predict", json=payload).status_code == 422


def test_valid_prediction():
    payload = {"queue_wait_sec": 120, "agent_exp_months": 24, "previous_calls": 2, "issue_count": 1, "hour": 10, "caller_age": 30, "language": "hindi", "team": "Team Alpha"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
