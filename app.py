import base64
import os
import sys

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from emotion_model import EmotionDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = "uploads"

# Create uploads folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Lazy load detector (only initialize when first used)
detector = None

def get_detector():
    """Lazy load the emotion detector to save memory"""
    global detector
    if detector is None:
        print("Initializing Emotion Detector...")
        detector = EmotionDetector(model_path="face_model.h5")
        print("Emotion Detector ready!")
    return detector


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # Predict emotion using the cleaner EmotionDetector class
        result = get_detector().predict_emotion(filepath)        # Read image and convert to base64 for display
        with open(filepath, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        # Clean up uploaded file
        os.remove(filepath)

        if not result["success"]:
            return jsonify({"error": result.get("error", "Unknown error")}), 400

        # Format response for frontend
        response = {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "all_predictions": result["all_predictions"],
            "face_detected": True,
            "image": f"data:image/jpeg;base64,{img_data}",
        }

        return jsonify(response)


@app.route("/webcam")
def webcam():
    """Render webcam page"""
    return render_template("webcam.html")


@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    """Handle webcam image prediction"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Remove data URL prefix
        image_data = image_data.split(",")[1] if "," in image_data else image_data

        # Decode base64 image
        import io

        from PIL import Image

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert PIL image to OpenCV format
        import cv2
        import numpy as np

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Predict emotion from frame
        result = get_detector().predict_from_frame(frame)
        
        if not result["success"]:
            return jsonify({"error": result.get("error", "No face detected")}), 400

        # Return first face result (for simplicity)
        if result.get("faces"):
            face_result = result["faces"][0]
            return jsonify(
                {
                    "success": True,
                    "emotion": face_result["emotion"],
                    "confidence": face_result["confidence"],
                }
            )
        else:
            return jsonify({"error": "No face detected"}), 400

    except Exception as e:
        return jsonify({"error": f"Error processing webcam image: {str(e)}"}), 500


if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    # Get port from environment variable (for Render) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
