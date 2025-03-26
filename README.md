# Emotion-Aware-AI
This research project aims to increase user engagement of learning games with a topic in computer science by dynamically adjusting the difficulty level through an emotion-aware NPC (Non-player character). The NPC, powered by an facial emotion recognition (FER) algorithm that categorizes facial expressions into 8 emotion types (including 3 custom ones), detects the current state of players via a combination of camera feed and in-game text sent between players.

## Data
Following the standard of well-known dataset FER-2013 created by Pierre Luc Carrier and Aaron Courville, integer labels are used to classify facial images expressing eight emotions, three of which (in italics) are tailored to the needs of this project:
- 0: Angry -- 4953
- 1: *Frustration* -- 265
- 2: *Boredom* -- 238
- ~~2: *Distracted*~~
- 3: Happy -- 8989
- 4: Sad -- 6077
- 5: Surprise -- 4002
- 6: Neutral -- 6198
> [!Note]
> *Distracted* is temporarily removed from our training dataset due to a failure in collecting meaningful images expressing this particular emotion. It may be added back in future if we're able to gather relevant images from other sources. Class labels are adjusted accordingly.

### How images are collected for custom emotions
Facial images representing frustration, distracted and boredom are fetched from Google Images using API calls. To ensure the diversity of our images, a basic search string is combined with other keywords like "men", "women" etc. before being passed to an API call. Faces from returned results are then extracted and cropped to a standard size of 48x48 with OpenCV, followed by a conversion to pixel string before being added to a csv file.

Finally, we manually inspect every pixel string to remove incorectly detected faces and dupliate images. This process is performed in each API call that collects 50 images and again every two API calls when a total of 100 images are fetched. 

### How image labels are assigned
We intentionally assigned our custom emotions to the exact two that are removed from the original FER2013 datasets -- disgust and fear, closely aligning our dataset to established practices.

### How to run our image collection program
Since custom images are collected via Google Images API, make sure to load your google API key and custom search engine ID into a .env file in the directory where main.py is located. See Google doc [here](https://developers.google.com/custom-search/v1/overview). Then, simply run `python main.py` in [data](data/main.py) and choose the first option on a menu that pops up.
> [!Note]
> Make sure to comment out the code in `main.py` that checks for `.env` credentials if you don't intend to run this program for image collection. 

> [!CAUTION]
> Other modes in the menu may break since they have been deprecated upon finishing data collection. Please see comments in [main.py](data/main.py). 

## Emotion Detection API

A FastAPI-based web service hosted on Digital Ocean that performs real-time emotion detection from webcam frames. The service accepts base64-encoded images and returns predicted emotions with their probabilities.

### Files

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

### API Endpoints

### POST /predict
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

### GET /health
Health check endpoint that returns service status.

## Environment Variables
- `GDRIVE_MODEL_URL`: URL to download the trained model weights
- `API_URL`: Base URL of the deployed API (for test script)

## Docker Deployment
The service can be containerized and deployed to cloud platforms like Digital Ocean. The Dockerfile sets up the necessary environment and dependencies for running the emotion detection service.

> [!CAUTION]
> Make sure to build your image with a flag `--platform=linux/amd64` since their APP platform only supports binaries built for the AMD64 CPU architecture. See [details](https://docs.digitalocean.com/products/app-platform/details/limits/#:~:text=built%20for%20the-,AMD64,-CPU%20architecture). Requirement may vary. Consult documentation of related host platforms if you'd like to deploy it on other places.