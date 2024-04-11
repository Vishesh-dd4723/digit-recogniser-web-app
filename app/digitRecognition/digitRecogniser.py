from keras import Model
from typing import Union
from digitRecognition.models import ResNet50 as resNet, SimpleNN
from resources.modelParams import ResNet50
from digitRecognition.modelNames import Models
from resources.logger import log
from digitRecognition.utils import process_image, process_csv, process_file
from fastapi import UploadFile
from resources import config
import numpy as np
from Requests import TrainRequest


class DigitRecogniser:
    def __init__(self, model_name: Models, input_shape, output_shape) -> None:
        self.model_name = model_name
        self.is_trained = False
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.metrics = None
        self.loss_fn = None
        self.optimizer = None
        self.history = None
        self.model = None
        self.define()

    def define(self) -> Model:
        log.info(f"Defining {self.model_name} model")
        if self.model_name == Models.ResNet:
            self.model = resNet(self.input_shape, self.output_shape).build()
        elif self.model_name == Models.SimpleNN:
            self.model = SimpleNN(self.input_shape, self.output_shape).build()
        else:
            log.error(f"Model: {self.model_name} not supported")
            raise ModuleNotFoundError
        self.is_trained = False
        self.set_compile_params()
        return self.summary()

    def summary(self):
        return self.model.summary()

    def set_compile_params(self, optimizer=None, loss_fn=None, metrics=None):
        self.optimizer = optimizer if optimizer else config.optimizer
        self.loss_fn = loss_fn if loss_fn else config.loss
        self.metrics = metrics if metrics else config.metrics

    def load_model_weights(self, name: str = config.dr_weights_name) -> bool:
        try:
            log.info(f"Loading model weights: {name}")
            self.model.load_weights(name)
            self.compile()
            self.is_trained = True
        except Exception as e:
            log.error(f"Model weights: {name} can't be loaded with error: {e}")
            return False
        return True

    def compile(self, optimizer=None, loss=None, metrics=None):
        self.set_compile_params(optimizer, loss, metrics)
        log.info(
            f"Compiling {self.model_name} with optimizer: {self.optimizer}, "
            f"loss function: {self.loss_fn} and measuring: {self.metrics}")
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

    def train(self, request: TrainRequest):
        if request.reset or not self.model:
            self.define()
            log.info(f"Resetting the model: {self.model_name}")
        if request.to_compile:
            self.compile(request.optimizer, request.loss, request.metrics)

        input_features, labels, _, _, _, _ = self.process(request.csv_file)
        self.history = self.model.fit(input_features, labels, epochs=request.epochs, batch_size=request.batch_size)
        self.is_trained = True
        self.save_model_weights()

    def save_model_weights(self, name=config.dr_weights_name) -> bool:
        self.model.save_weights(name)
        log.info(f"Model weights saved successfully in {name}")
        return True

    def test(self, csv_file: Union[UploadFile, str]):
        _, _, input_features, labels, _, _ = self.process(csv_file)
        predictions = self.model.evaluate(input_features, labels, return_dict=True)
        log.info(f"{predictions}")
        return predictions

    def predict(self, image: UploadFile):
        if not self.is_trained:
            log.error("Can't Predict! Model not trained")
            raise Exception()
        input_features = self.process_image(image)
        predicted_value = np.argmax(self.model.predict(input_features))
        log.info(f"Predicted value: {predicted_value}")
        return predicted_value

    def process_image(self, image: UploadFile):
        return process_image(self.input_shape, image)

    def process(self, file: UploadFile):
        log.info(f"Processing file: {file.filename} for model: {self.model_name}")
        return process_file(file)

    def process(self, csv: str):
        log.info(f"Processing CSV file: {csv}")
        return process_csv(csv)
