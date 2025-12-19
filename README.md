# AgroDoctor - Plant Disease Prediction API

This is the backend server for the AgroDoctor final year project. It uses a TensorFlow/Keras model to predict plant diseases, OpenCV for severity analysis, and Google's Gemini API for generating treatment plans.

## Project Structure

- **/app**: Contains the core application logic (prediction, severity, treatment, database).
- **/models**: Stores the pre-trained `.h5` model and class indices.
- **/test_images**: Sample images for testing.
- **main.py**: The main FastAPI application file.
- **requirements.txt**: Project dependencies.
- **database.db**: The SQLite database file (will be created automatically).

## Setup and Installation

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    - Open `app/treatment_service.py`.
    - Replace the placeholder `"YOUR_GEMINI_API_KEY"` with your actual API key from Google AI Studio.

4.  **Place your model:**
    - Ensure your trained `plant_disease_model.h5` and `class_indices.json` files are in the `models/` directory.

## Running the Server

To start the API server, run the following command from the root directory (`plant_disease_backend/`):

```bash
uvicorn main:app --reload