import pytest
import os
# import numpy as np
from fastapi.testclient import TestClient
from src.serve_model import create_app

# ==== Fixtures ====
@pytest.fixture(scope="module", autouse=True)
def setup_env():
    os.environ["MLFLOW_TRACKING_URI"] = "fake_tracking_uri"
    os.environ["MODEL_URI"] = "models:/fake_model/Production"
    yield

@pytest.fixture(scope="module")
def test_app():
    app = create_app()
    client = TestClient(app)
    return client

# ==== Test cases ====

def test_health_ok(test_app):
    response = test_app.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert json_data["status"] in ["ok", "model_error"]
    assert "health" in json_data

def test_metrics(test_app):
    response = test_app.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    assert b"predict_requests_total" in response.content

def test_predict_success(monkeypatch, test_app):
    from src import serve_model
    import numpy as np

    class FakeModel:
        def predict(self, df):
            return np.array([[0.1, 0.2, 0.7]])

    monkeypatch.setattr(serve_model, "model", FakeModel())
    monkeypatch.setattr(serve_model, "MODEL_VERSION_INFO", "test-version")

    response = test_app.post("/predict", json={"text": "This is a test"})
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_model_not_loaded(monkeypatch, test_app):
    from src import serve_model

    monkeypatch.setattr(serve_model, "model", None)
    monkeypatch.setattr(serve_model, "load_and_register_model", lambda: None)

    response = test_app.post("/predict", json={"text": "Test"})
    assert response.status_code == 500


def test_reload_model(monkeypatch, test_app):
    from src import serve_model
    monkeypatch.setattr(serve_model, "load_and_register_model", lambda: None)
    monkeypatch.setattr(serve_model, "MODEL_VERSION_INFO", "fake-version")
    response = test_app.post("/reload-model")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "reloaded"
    assert json_data["model_version"] == "fake-version"
