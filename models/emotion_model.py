# model/emotion_model.py

import numpy as np
import cv2
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

class EmotionDetector:
    def __init__(self, model_path=None):
        """
        Initialize Emotion Detector
        
        Args:
            model_path: Path to trained model (.h5 file)
                       If None, will try to find model in common locations
        """
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.img_size = 48
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"Emotion Detector initialized with model: {self.model is not None}")
    
    def _load_model(self, model_path):
        """Load the emotion detection model"""
        
        # If specific path provided, use it
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            return load_model(model_path)
        
        # Try common model locations
        possible_paths = [
            'models/face_model.h5',
            'face_model.h5'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found model at: {path}")
                return load_model(path)
        
        # If no model found, create a dummy model for demo
        print("WARNING: No trained model found. Creating dummy model for demo.")
        print("Please train a model or download a pretrained model.")
        return self._create_dummy_model()
    
    def _create_dummy_model(self):
        """
        Create a dummy model for demonstration purposes
        WARNING: This will give random results!
        """
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=(self.img_size, self.img_size, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model prediction
        
        Args:
            face_img: Grayscale face image (numpy array)
            
        Returns:
            Preprocessed image ready for model
        """
        # Resize to model input size
        face_img = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        
        # Convert to array and add dimensions
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def detect_faces(self, image):
        """
        Detect faces in image
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            List of face coordinates (x, y, w, h)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def predict_emotion(self, image_path):
        """
        Predict emotion from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                return {'success': False, 'error': 'Could not read image file'}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detect_faces(img)
            
            if len(faces) == 0:
                return {
                    'success': False, 
                    'error': 'No face detected in the image. Please upload a clear image with a visible face.'
                }
            
            # Get the largest face (assuming it's the main subject)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = self.preprocess_face(face_roi)
            
            # Predict emotion
            predictions = self.model.predict(processed_face, verbose=0)[0]
            
            # Get emotion with highest probability
            emotion_idx = np.argmax(predictions)
            emotion = self.emotion_labels[emotion_idx]
            confidence = float(predictions[emotion_idx])
            
            # Get all predictions sorted by confidence
            all_predictions = {
                self.emotion_labels[i]: float(predictions[i]) 
                for i in range(len(self.emotion_labels))
            }
            
            # Sort predictions by confidence
            all_predictions = dict(
                sorted(all_predictions.items(), 
                      key=lambda x: x[1], 
                      reverse=True)
            )
            
            return {
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'face_location': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            }
            
        except Exception as e:
            return {
                'success': False, 
                'error': f'Error processing image: {str(e)}'
            }
    
    def predict_from_frame(self, frame):
        """
        Predict emotion from video frame (for webcam support)
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return {'success': False, 'error': 'No face detected'}
            
            results = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = self.preprocess_face(face_roi)
                
                # Predict emotion
                predictions = self.model.predict(processed_face, verbose=0)[0]
                
                emotion_idx = np.argmax(predictions)
                emotion = self.emotion_labels[emotion_idx]
                confidence = float(predictions[emotion_idx])
                
                results.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'location': (x, y, w, h)
                })
            
            return {'success': True, 'faces': results}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =====================================
# Test the detector
# =====================================
if __name__ == "__main__":
    # Test the detector
    detector = EmotionDetector()
    
    print("\nEmotion Detector Test")
    print("=" * 50)
    print(f"Model loaded: {detector.model is not None}")
    print(f"Emotion labels: {detector.emotion_labels}")
    print("\nReady to detect emotions!")
    
    # Example usage:
    # result = detector.predict_emotion('path/to/image.jpg')
    # if result['success']:
    #     print(f"Detected emotion: {result['emotion']}")
    #     print(f"Confidence: {result['confidence']:.2%}")