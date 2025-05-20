from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import os
import requests

app = FastAPI()

MODEL_ID = "1z5Ddw2pXgd4JEfZiNAeVrRSUudx4Hb9V"
MODEL_PATH = "resnet50_trashnet_finetuned.h5"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model indiriliyor...")
        download_file_from_google_drive(MODEL_ID, MODEL_PATH)
        size = os.path.getsize(MODEL_PATH)
        print(f"Model indirildi, dosya boyutu: {size} byte")
    else:
        size = os.path.getsize(MODEL_PATH)
        print(f"Model zaten mevcut, dosya boyutu: {size} byte")

def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_data = preprocess_image(image)

    predictions = model.predict(input_data)
    predicted_class = int(np.argmax(predictions, axis=1)[0])

    return JSONResponse(content={"prediction": predicted_class})
