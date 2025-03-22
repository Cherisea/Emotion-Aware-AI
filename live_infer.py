import cv2.data
import torch
from models.Resnet import ResEmoteNet
from pathlib import Path
from torchvision import transforms
import cv2
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import requests
from dotenv import load_dotenv
import gdown
import gc
import logging
from typing import List, Tuple, Optional
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Emotion Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_CACHE_DIR = "model_cache"
MODEL_FILENAME = "best_model.pth"
EMOTIONS = ['angry', 'frustration', 'boredom', 'happy', 'sad', 'surprise', 'neutral']

# Configure device
device = "cpu"

def setup_model() -> ResEmoteNet:
    """
    Initialize and load the model from Google Drive.
    Returns:
        ResEmoteNet: Loaded model
    Raises:
        HTTPException: If model loading fails
    """
    try:
        # Create cache directory
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_CACHE_DIR, MODEL_FILENAME)

        # Initialize model
        model = ResEmoteNet().to(device)

        # Download model if not in cache
        if not os.path.exists(model_path):
            logger.info("Downloading model weights from Google Drive...")
            gdrive_url = os.getenv('GDRIVE_MODEL_URL')
            if not gdrive_url:
                raise ValueError("GDRIVE_MODEL_URL environment variable not set")
            
            try:
                gdown.download(gdrive_url, model_path, quiet=False)
                logger.info("Model weights downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise HTTPException(status_code=500, detail="Failed to download model weights")

        # Load model weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise HTTPException(status_code=500, detail="Failed to load model weights")

    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize model at startup
model = setup_model()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Instantiate a haar cascade classifier with a pretrained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades 
                                        + "haarcascade_frontalface_default.xml")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model input.
    Args:
        image: PIL Image
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to preprocess image")

def infer_emotion(image: Image.Image) -> List[float]:
    """
    Infer emotions from an image.
    Args:
        image: PIL Image
    Returns:
        List[float]: Emotion probabilities
    """
    try:
        # Clear memory
        gc.collect()
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
        
        # Convert to probabilities
        scores = probs.numpy().flatten()
        rounded_scores = [round(score, 2) for score in scores]
        
        # Clear memory
        gc.collect()
        
        return rounded_scores
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a frame.
    Args:
        frame: OpenCV image frame
    Returns:
        List[Tuple[int, int, int, int]]: List of face bounding boxes
    """
    try:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_img, 1.1, 5, minSize=(40, 40))
        return faces
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return []

def process_frame(frame: np.ndarray) -> Tuple[str, List[float]]:
    """
    Process a frame and return the predicted emotion and probabilities.
    Args:
        frame: OpenCV image frame
    Returns:
        Tuple[str, List[float]]: Predicted emotion and probabilities
    """
    try:
        faces = detect_faces(frame)
        if not faces:
            return None, None

        # Get the first face detected
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(face_img)
        
        # Get emotion probabilities
        probabilities = infer_emotion(pil_img)
        predicted_emotion = EMOTIONS[np.argmax(probabilities)]
        
        return predicted_emotion, probabilities
    except Exception as e:
        logger.error(f"Frame processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_emotion(request: Request):
    """
    Endpoint to predict emotion from an image frame.
    Expected JSON payload:
    {
        "frame": "base64_encoded_image_data"
    }
    Returns:
    {
        "emotion": "predicted_emotion",
        "probabilities": {
            "angry": 0.1,
            "frustration": 0.2,
            ...
        }
    }
    """
    try:
        # Get the frame data from the request
        data = await request.json()
        frame_data = data.get("frame")
        if not frame_data:
            raise HTTPException(status_code=400, detail="No frame data provided")

        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data.split(",")[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process frame
        emotion, probabilities = process_frame(frame)

        return {
            "emotion": emotion,
            "probabilities": dict(zip(EMOTIONS, probabilities)) if probabilities else None
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}