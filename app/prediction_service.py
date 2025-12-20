import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json

MODEL_PATH = "models/plant_disease_model.keras"

_model = None
_class_names = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

def get_class_names():
    global _class_names
    if _class_names is None:
        with open("models/class_indices.json") as f:
            class_indices = json.load(f)
        # invert mapping: index → class name
        _class_names = {v: k for k, v in class_indices.items()}
    return _class_names


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_disease(image_bytes):
    model = get_model()
    class_names = get_class_names()

    img = preprocess_image(image_bytes)
    preds = model.predict(img)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "disease_name": class_names.get(idx, "Unknown Disease"),
        "confidence": confidence
    }
