import os.path
from fastapi import UploadFile, HTTPException
from resources.logger import log
import cv2
import numpy as np
import io
import pandas as pd
from resources.config import defaultCsvFile, dataset_split_ratio


def process_image(input_shape, file: UploadFile):
    filename = file.filename
    file_extension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image.resize(input_shape)
    image = np.squeeze(image)
    image = np.expand_dims(image, axis=0)
    return image


def process_file(file: UploadFile):
    csv = file.filename
    if csv.split('.')[-1] != 'csv':
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    try:
        data = pd.read_csv(csv)
        return process(data)
    except FileNotFoundError:
        log.debug(f"{csv} file not found, going with default file")
        dir_path = os.path.dirname(__file__).replace('\\', '/')
        csv = dir_path + defaultCsvFile
    return process_csv(csv)


def process_csv(csv: str):
    data = pd.read_csv(csv)
    return process(data)


def process(data, train_dev_split_ratio=dataset_split_ratio):
    train_images_labels = np.array(data.iloc[0:, :785])
    np.random.shuffle(train_images_labels)
    total_entries = train_images_labels.shape[0]
    training_size = int(total_entries*train_dev_split_ratio)
    dev_size = training_size + (total_entries-training_size)//2
    x_train = np.array(train_images_labels[:training_size, 1:]) / 255.
    x_dev = np.array(train_images_labels[training_size:dev_size, 1:]) / 255.
    x_test = np.array(train_images_labels[dev_size:, 1:]) / 255.

    x_train = x_train.reshape((training_size, 28, 28, 1))
    x_dev = x_dev.reshape((dev_size-training_size, 28, 28, 1))
    x_test = x_test.reshape((total_entries-dev_size, 28, 28, 1))

    y_train = one_hot_matrix(np.array(train_images_labels[:training_size, 0]), 10)
    y_dev = one_hot_matrix(np.array(train_images_labels[training_size:dev_size, 0]), 10)
    y_test = one_hot_matrix(np.array(train_images_labels[dev_size:, 0]), 10)
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def one_hot_matrix(y, classes):
    return np.eye(classes)[y.reshape(-1)]
