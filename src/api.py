from math import *
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predictor import DraftPredictor


app = FastAPI(
    title="Dota Draft AI",
    version="0.1.0",
    description="Draft-only Dota 2 win probability baseline.",
)


class DraftRequest(BaseModel):
    radiant: list[int] = Field(min_length=5, max_length=5)
    dire: list[int] = Field(min_length=5, max_length=5)


@lru_cache(maxsize=1)
def get_predictor() -> DraftPredictor:
    return DraftPredictor()


@app.get("/health")
def health():
    try:
        predictor = get_predictor()
        return {"status": "ok", "metrics": predictor.metrics}
    except FileNotFoundError as e:
        return {"status": "model_missing", "detail": str(e)}


@app.post("/predict")
def predict(req: DraftRequest):
    try:
        return get_predictor().predict(req.radiant, req.dire)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
