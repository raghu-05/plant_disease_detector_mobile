import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json

# --------------------------------------------------
# 1. Load model (lazy loading – SAFE for Render)
# --------------------------------------------------
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_INDEX_PATH = "models/class_indices.json"

_model = None
_class_map = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        print("✅ Lightweight model loaded")
    return _model


def get_class_map():
    """
    Handles BOTH formats safely:
    1) { "Apple___Black_rot": 1 }
    2) { "1": "Apple___Black_rot" }
    """
    global _class_map
    if _class_map is None:
        with open(CLASS_INDEX_PATH, "r") as f:
            raw = json.load(f)

        # If keys are digits → already index → name
        if all(str(k).isdigit() for k in raw.keys()):
            _class_map = {int(k): v for k, v in raw.items()}
        else:
            # Otherwise → name → index → invert
            _class_map = {v: k for k, v in raw.items()}

        print("✅ Class map loaded:", _class_map)

    return _class_map



# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------
def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --------------------------------------------------
# 3. Prediction
# --------------------------------------------------
def predict_disease(image_bytes: bytes) -> dict:
    model = get_model()
    class_map = get_class_map()

    image = preprocess_image(image_bytes)

    preds = model.predict(image)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    disease_name = class_map.get(class_id, "Unknown Disease")

    return {
        "disease_name": disease_name,
        "confidence": round(confidence, 4)
    }
