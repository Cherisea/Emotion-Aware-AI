# Model Deployment

A FastAPI-based web service built from a Docker image residing in Docker Hub and hosted on Digital Ocean that performs real-time emotion detection from webcam frames. The service accepts base64-encoded images and returns predicted emotions with their probabilities.

## Files

- `live_infer.py`: Main FastAPI application that handles image processing and emotion detection
  - Provides `/predict` endpoint for emotion detection
  - Includes `/health` endpoint for monitoring
  - Handles face detection and emotion classification
  - Uses a pre-trained ResEmoteNet model

- `deploy_test.py`: Client script to test the API
  - Captures webcam frames
  - Converts frames to base64
  - Sends frames to API at specified intervals
  - Displays emotion detection results

- `/models/Resnet`: Definition of model architecture 

## API Endpoints

### `POST /predict`
Accepts base64-encoded image data and returns detected emotions:
```json
{
    "frame": "data:image/jpeg;base64,<encoded_image_data>"
}
```

Returns:
```json
{
    "emotion": "happy",
    "probabilities": {
        "angry": 0.1,
        "frustration": 0.2,
        "boredom": 0.1,
        "happy": 0.4,
        "sad": 0.1,
        "surprise": 0.05,
        "neutral": 0.05
    }
}
```

### `GET /health`
Health check endpoint that returns service status.

## Environment Variables
- `GDRIVE_MODEL_URL`: URL to download the trained model weights
- `API_URL`: Base URL of the deployed API (for test script)

## Docker Deployment
The service can be containerized and deployed to cloud platforms like Digital Ocean. The Dockerfile sets up the necessary environment and dependencies for running the emotion detection service. Docker image should then be pushed to an online registry, such as Docker Hub for retrieval by your host platform. 

> [!CAUTION]
> Make sure to build your image with a flag `--platform=linux/amd64` since their APP platform only supports binaries built for the AMD64 CPU architecture. See [details](https://docs.digitalocean.com/products/app-platform/details/limits/#:~:text=built%20for%20the-,AMD64,-CPU%20architecture). Requirement may vary. Consult documentation of related host platforms if you'd like to deploy it on other places.