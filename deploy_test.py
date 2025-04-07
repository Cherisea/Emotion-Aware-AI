"""
    This script is used to demonstrate how to get predictions from our deployed model.
    It captures frames from your webcam and sends them to the API for prediction at a
    specified interval.
"""

import cv2
import base64
import requests
import json
from dotenv import load_dotenv
import os
import time
import numpy as np

load_dotenv()

# Get API URL from .env file
API_URL = os.getenv("API_URL")

def encode_frame(frame):
    """Convert OpenCV frame to base64 string."""
    # Convert frame to JPEG format and get a numpy array buffer
    _, buffer = cv2.imencode('.jpg', frame)

    # Encode the buffer in base64 binary format and decode it to a string
    return base64.b64encode(buffer).decode('utf-8')

def draw_emotion_results(frame, emotion, probabilities):
    """Draw emotion prediction results on the frame."""
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Define colors for different emotions (BGR format)
    colors = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 128, 0),    # Dark green
        'fear': (128, 0, 128),     # Purple
        'happy': (0, 255, 0),      # Green
        'neutral': (255, 255, 0),  # Cyan
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 255, 255)  # Yellow
    }
    
    # Get color for the detected emotion
    emotion_color = colors.get(emotion, (255, 255, 255))  # Default to white if emotion not in colors
    
    # Draw the main emotion text
    cv2.putText(annotated_frame, f"Emotion: {emotion.upper()}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
    
    # Draw probability bars for each emotion
    y_position = 80
    for i, (emotion_name, prob) in enumerate(probabilities.items()):
        # Get color for this emotion
        color = colors.get(emotion_name, (255, 255, 255))
        
        # Draw emotion name and probability
        cv2.putText(annotated_frame, f"{emotion_name}: {prob:.2f}", 
                    (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Draw probability bar
        bar_width = int(prob * 200)  # Scale probability to bar width
        cv2.rectangle(annotated_frame, (150, y_position - 15), 
                     (150 + bar_width, y_position), color, -1)
        
        y_position += 30
    
    return annotated_frame

def test_emotion_detection(interval=1):
    """
    Test the emotion detection model by capturing frames from the webcam
    and sending them to the API for prediction. 

    Args:
        interval (int): The interval between frames in seconds. Default is 1.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Initialize variables for prediction
    last_prediction_time = 0
    current_emotion = "Loading..."
    current_probabilities = {}
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Check if it's time to make a new prediction
            current_time = time.time()
            if current_time - last_prediction_time >= interval:
                # Convert frame to base64 string
                frame_data = encode_frame(frame)
                
                # Prepare the request
                payload = {
                    "frame": f"data:image/jpeg;base64,{frame_data}"
                }
                
                try:
                    # Make a post request to the /predict endpoint
                    response = requests.post(
                        API_URL + "/predict",
                        json=payload
                    )
                    
                    # Process the result
                    if response.status_code == 200:
                        result = response.json()
                        current_emotion = result.get('emotion', 'unknown')
                        current_probabilities = result.get('probabilities', {})
                        last_prediction_time = current_time
                    else:
                        print(f"Error: Server returned status code {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
            
            # Draw emotion results on the frame
            annotated_frame = draw_emotion_results(frame, current_emotion, current_probabilities)
            
            # Display the annotated frame
            cv2.imshow('Emotion Detection', annotated_frame)
            
            # Wait for keyboard input for 1ms and break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_emotion_detection(5)