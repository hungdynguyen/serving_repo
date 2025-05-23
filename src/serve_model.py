import os
import time
import threading
import logging

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ==== Logging ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Globals & Lock ====
model = None
MODEL_VERSION_INFO = "unknown"
model_lock = threading.Lock()

# ==== Prometheus metrics ====
PREDICT_COUNTER = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["model_version", "outcome", "predicted_label"]
)

PREDICT_DURATION = Histogram(
    "predict_duration_seconds",
    "Prediction duration",
    ["model_version"]
)

MODEL_INFO = Gauge(
    "model_serving_info",
    "Indicates model version/URI being served (1 if active)",
    ["version_uri"]
)

RELOAD_COUNTER = Counter(
    "reload_model_requests_total",
    "Total model reload requests",
    ["outcome"]
)

RELOAD_DURATION = Histogram(
    "reload_model_duration_seconds",
    "Model reload duration"
)

HEALTH_STATUS = Gauge(
    "service_health_status",
    "Health status of the service (1=OK, 0=error)"
)

INPUT_LENGTH_HIST = Histogram(
    "input_text_length",
    "Length of input text",
    buckets=[0, 10, 20, 50, 100, 200, 500]
)

LABEL_MAP = {0: "neutral", 1: "negative", 2: "positive"}


def load_and_register_model():
    global model, MODEL_VERSION_INFO

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MODEL_URI = os.getenv("MODEL_URI")

    if not MLFLOW_TRACKING_URI or not MODEL_URI:
        logger.error("MLFLOW_TRACKING_URI and MODEL_URI must be set")
        raise RuntimeError("Missing MLFLOW_TRACKING_URI or MODEL_URI")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with model_lock:
        logger.info(f"Loading model from URI: {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)

        if MODEL_URI.startswith("models:/"):
            client = mlflow.tracking.MlflowClient()
            model_name = MODEL_URI.split("/")[1]
            prod = client.get_latest_versions(model_name, 
                                            stages=["Production"])[0]
            MODEL_VERSION_INFO = prod.version
        else:
            MODEL_VERSION_INFO = MODEL_URI

        MODEL_INFO.clear()
        MODEL_INFO.labels(version_uri=str(MODEL_VERSION_INFO)).set(1)
        HEALTH_STATUS.set(1)
        logger.info(f"Model loaded, version: {MODEL_VERSION_INFO}")


class InputText(BaseModel):
    text: str


def create_app() -> FastAPI:
    app = FastAPI(title="Sentiment API")

    @app.post("/predict")
    def predict(input_data: InputText):
        global model
        if model is None:
            load_and_register_model()

        start = time.time()
        outcome = "success"
        pred_label = "unknown"

        try:
            df = pd.DataFrame({"text": [input_data.text]})
            probs = model.predict(df)
            idx = int(np.argmax(probs[0]))
            pred_label = LABEL_MAP.get(idx, "unknown")
            return {
                "prediction": pred_label,
                "probs": probs[0].tolist(),
                "model_version": MODEL_VERSION_INFO,
            }
        except Exception as e:
            outcome = "error"
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            duration = time.time() - start
            PREDICT_DURATION.labels(model_version=MODEL_VERSION_INFO).observe(duration)
            PREDICT_COUNTER.labels(
                model_version=MODEL_VERSION_INFO,
                outcome=outcome,
                predicted_label=pred_label if outcome == "success" else "error",
            ).inc()

    @app.post("/reload-model")
    def reload_model():
        start = time.time()
        outcome = "success"

        try:
            load_and_register_model()
            return {"status": "reloaded", "model_version": MODEL_VERSION_INFO}
        except Exception as e:
            outcome = "error"
            HEALTH_STATUS.set(0)
            logger.error(f"Reload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            RELOAD_DURATION.observe(time.time() - start)
            RELOAD_COUNTER.labels(outcome=outcome).inc()

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health():
        ok = model is not None
        HEALTH_STATUS.set(1 if ok else 0)
        return {"status": "ok" if ok else "model_error", "health": ok}

    return app
