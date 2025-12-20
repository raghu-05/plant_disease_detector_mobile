import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

MODEL_PATH = "models/plant_disease_model.keras"

_model = None
_class_indices = None


def load_model_and_classes():
    global _model, _class_indices

    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    if _class_indices is None:
        with open("models/class_indices.json") as f:
            _class_indices = {str(k): v for k, v in json.load(f).items()}


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_disease(image_bytes):
    load_model_and_classes()

    image = preprocess_image(image_bytes)

    preds = _model.predict(image)   # ✅ WORKS NOW
    class_index = str(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "disease_name": _class_indices.get(class_index, "Unknown"),
        "confidence": confidence
    }
