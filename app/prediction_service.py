import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json

# --- 1. Load Model and Class Indices ---
# It's efficient to load these once when the application starts.
model = tf.keras.models.load_model('models/plant_disease_model.h5')
with open('models/class_indices.json', 'r') as f:
    # Keras's flow_from_directory uses string indices, so we ensure they match.
    class_indices = {str(k): v for k, v in json.load(f).items()}

# --- 2. Preprocessing Function ---
# This function must exactly match the preprocessing you used for training.
def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """Converts image bytes to a preprocessed numpy array for the model."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array.astype('float32') / 255.0  # Rescale
    return img_array

# --- 3. Prediction Function ---
def predict_disease(image_bytes: bytes) -> dict:
    """Takes image bytes and returns the predicted disease and confidence."""
    # Preprocess the image
    processed_img = preprocess_image(image_bytes)

    # Make prediction
    prediction = model.predict(processed_img)
    
    # Get the top prediction
    predicted_class_index = str(np.argmax(prediction, axis=1)[0])
    disease_name = class_indices.get(predicted_class_index, "Unknown Disease")
    confidence = float(np.max(prediction))

    return {"disease_name": disease_name, "confidence": confidence}