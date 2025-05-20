from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import os
import requests
from tensorflow.keras.applications.resnet50 import preprocess_input

app = FastAPI()

DROPBOX_URL = "https://www.dropbox.com/scl/fi/kbet8wqtoazc40fmmz1c6/resnet50_trashnet_finetuned.h5?rlkey=egok5st4mlclaqb9kotvf4gn7&st=sl1zmtib&dl=1"
MODEL_PATH = "resnet50_trashnet_finetuned.h5"

def download_file_from_dropbox(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model Dropbox'tan indiriliyor...")
        download_file_from_dropbox(DROPBOX_URL, MODEL_PATH)
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
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_data = preprocess_image(image)

    predictions = model.predict(input_data)
    predicted_class = int(np.argmax(predictions, axis=1)[0])

    return JSONResponse(content={
        "prediction": predicted_class,
        "probabilities": predictions[0].tolist()
    })
