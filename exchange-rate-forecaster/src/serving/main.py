"""
FastAPI serving layer.

Endpoints:
  GET  /health          — liveness probe
  GET  /ready           — readiness probe (checks model loaded)
  POST /predict         — single prediction
  GET  /metrics         — Prometheus text format metrics
  POST /reload          — hot-reload the model from registry (admin)

Prometheus metrics exposed:
  - prediction_requests_total (counter, labelled by pair)
  - prediction_latency_seconds (histogram)
  - model_loaded (gauge)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.serving.predictor import load_model, model_loaded, predict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ── Prometheus metrics ──────────────────────────────────────────────────────
REQUEST_COUNTER = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["pair", "status"],
)
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["pair"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
MODEL_LOADED_GAUGE = Gauge("model_loaded", "1 if model is loaded and ready")


# ── Lifecycle ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model(TRACKING_URI)
        MODEL_LOADED_GAUGE.set(1)
    except Exception:
        logger.exception("Failed to load model at startup")
        MODEL_LOADED_GAUGE.set(0)
    yield


app = FastAPI(
    title="Exchange Rate LSTM API",
    version="1.0.0",
    description="Forecasts currency exchange rates using a trained LSTM model.",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    pair: str = Field("USD/BRL", description="Currency pair, e.g. USD/BRL")
    features: list[list[float]] = Field(
        ...,
        description="Sequence of feature vectors. Shape: (seq_len, n_features).",
    )

    model_config = {"json_schema_extra": {"example": {"pair": "USD/BRL", "features": [[0.5] * 25]}}}


class PredictResponse(BaseModel):
    pair: str
    predicted_rate: float
    latency_ms: float


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready() -> dict[str, Any]:
    if not model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict_endpoint(req: PredictRequest) -> PredictResponse:
    if not model_loaded():
        REQUEST_COUNTER.labels(pair=req.pair, status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        features = np.array(req.features, dtype=np.float32)
        rate = predict(features)
        latency = (time.perf_counter() - t0) * 1000
        LATENCY.labels(pair=req.pair).observe(latency / 1000)
        REQUEST_COUNTER.labels(pair=req.pair, status="ok").inc()
        return PredictResponse(pair=req.pair, predicted_rate=rate, latency_ms=round(latency, 3))
    except Exception as exc:
        REQUEST_COUNTER.labels(pair=req.pair, status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics", tags=["ops"])
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload", tags=["ops"])
async def reload_model() -> dict[str, Any]:
    try:
        load_model(TRACKING_URI)
        MODEL_LOADED_GAUGE.set(1)
        return {"status": "reloaded"}
    except Exception as exc:
        MODEL_LOADED_GAUGE.set(0)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
