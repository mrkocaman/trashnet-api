from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import requests
import os

app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?id=1z5Ddw2pXgd4JEfZiNAeVrRSUudx4Hb9V"
MODEL_PATH = "resnet50_trashnet_finetuned.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model indiriliyor...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model indirildi.")

def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(image: Image.Image):
    # ResNet50 ile uyumlu preprocess örneği
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # normalize et
    img_array = np.expand_dims(img_array, axis=0)  # batch boyutu ekle
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_data = preprocess_image(image)

    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return JSONResponse(content={"prediction": int(predicted_class)})
