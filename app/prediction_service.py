# app/prediction_service.py
import os
import json
import io
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")

# Load once
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.reshape(1, 224, 224, 3)


def predict_disease(image_bytes: bytes):
    image = preprocess_image(image_bytes)
    predictions = model.predict(image)
    idx = str(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "disease_name": class_indices.get(idx, "Unknown Disease"),
        "confidence": confidence
    }
