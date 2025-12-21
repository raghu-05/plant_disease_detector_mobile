# app/prediction_service.py
import os
import json
import io
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "plant_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "..", "models", "class_indices.json")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {int(k): v for k, v in class_indices.items()}


def predict_disease(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "disease_name": index_to_class.get(idx, "Unknown Disease"),
        "confidence": confidence
    }
