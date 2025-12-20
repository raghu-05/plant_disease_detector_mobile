import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json

MODEL_PATH = "models/plant_disease_model.keras"
CLASS_INDEX_PATH = "models/class_indices.json"

_model = None
CLASS_MAP = None


def load_model_and_classes():
    global _model, CLASS_MAP

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if CLASS_MAP is None:
        with open(CLASS_INDEX_PATH, "r") as f:
            raw = json.load(f)
            # 🔥 REVERSE mapping {0: "Apple___Apple_scab"}
            CLASS_MAP = {v: k for k, v in raw.items()}


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_disease(image_bytes: bytes) -> dict:
    load_model_and_classes()

    image = preprocess_image(image_bytes)

    preds = _model.predict(image)
    class_index = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    disease_name = CLASS_MAP.get(class_index, "Unknown Disease")

    return {
        "disease_name": disease_name,
        "confidence": confidence
    }
