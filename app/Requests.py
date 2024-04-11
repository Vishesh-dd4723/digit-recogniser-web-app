from pydantic import BaseModel
from digitRecognition.modelNames import Models
from resources import config


class InitializationRequest(BaseModel):
    input_shape: str
    output_shape: int
    name: Models


class TrainRequest(BaseModel):
    epochs: int
    batch_size: int
    csv_file: str
    reset: bool = False
    to_compile: bool = True
    optimizer: str = config.optimizer
    loss: str = config.loss
    metrics: list = config.metrics
