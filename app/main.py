from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, UploadFile, File

from Requests import InitializationRequest, TrainRequest
from Responses import InitializationResponse, TestResponse, PredictionResponse
from digitRecognition.digitRecogniser import DigitRecogniser
from resources.logger import log

app = FastAPI()
baseUrl = '/digitRecogniser'
model: Optional[DigitRecogniser] = None


@app.get(baseUrl+"/")
def homepage():
    return "Welcome"


@app.post(baseUrl+"/initialize", response_model=InitializationResponse)
def init(request: InitializationRequest):
    global model
    log.debug(f"Processing initialization request:{request}")
    model = DigitRecogniser(request.name, eval(request.input_shape), request.output_shape)
    response = InitializationResponse(summary=str(model.summary()))
    return response


@app.post(baseUrl+"/uploadFile")
def upload_file():
    # TODO: Process to add files
    pass


@app.post(baseUrl+"/load/{model}")
def load_model(file: Optional[str]):
    global model
    return model.load_model_weights(file)


@app.post(baseUrl+"/train")
def train(request: TrainRequest):
    global model
    log.debug(f"Processing training request:{request}")
    model.train(request)
    return True


@app.post(baseUrl+"/test", response_model=TestResponse)
def test(file: Union[UploadFile, str]):
    global model
    response = model.test(file)
    response = TestResponse.parse_obj(response)
    return response


@app.get(baseUrl+"/predict", response_model=PredictionResponse)
def predict(image: UploadFile):
    global model
    pred_val = model.predict(image)
    response = PredictionResponse(prediction=pred_val)
    return response


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
