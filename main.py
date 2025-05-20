from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import os
import requests

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
    # Debug: gelen görselin bilgisi
    print(f"Görsel orijinal boyut: {image.size}, mod: {image.mode}")

    # Model için doğru boyut ve normalize etme
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32) / 255.0
    print(f"Preprocess sonrası shape: {img_array.shape}, dtype: {img_array.dtype}")

    # Kanal sırası doğru mu kontrol et
    if img_array.shape[2] != 3:
        raise ValueError("Görüntü 3 kanallı olmalı (RGB)")

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Görseli API tarafında kaydedip inceleyebilirsin (debug için)
        # image.save("debug_uploaded_image.jpg")

        input_data = preprocess_image(image)

        predictions = model.predict(input_data)
        print(f"Model çıktısı (olasılıklar): {predictions}")

        predicted_class = int(np.argmax(predictions, axis=1)[0])

        return JSONResponse(content={
            "prediction": predicted_class,
            "probabilities": predictions[0].tolist()
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
