import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(
    "models/plant_disease_model.keras",
    compile=False
)

# ✅ Export as TensorFlow SavedModel (Render-compatible)
model.export("models/plant_disease_model_tf")

print("✅ Model exported successfully in TF SavedModel format")
