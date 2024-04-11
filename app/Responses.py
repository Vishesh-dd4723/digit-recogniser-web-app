from typing import Any
from pydantic import BaseModel
from digitRecognition.modelNames import Models


class InitializationResponse(BaseModel):
    summary: Any


class TestResponse(BaseModel):
    accuracy: float
    loss: float


class PredictionResponse(BaseModel):
    prediction: int
