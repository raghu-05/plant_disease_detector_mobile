from PIL import Image
import numpy as np
import io
import tensorflow as tf

MODEL_PATH = "models/plant_disease_model.keras"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid or unsupported image format")

    img = img.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(image_bytes):
    model = get_model()
    img = preprocess_image(image_bytes)

    preds = model.predict(img)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "disease_name": str(class_index),  # TEMP (mapping later)
        "confidence": confidence
    }
