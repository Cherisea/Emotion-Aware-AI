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

load_dotenv()

# Get API URL from .env file
API_URL = os.getenv("API_URL")

def encode_frame(frame):
    """Convert OpenCV frame to base64 string."""
    # Convert frame to JPEG format and get a numpy array buffer
    _, buffer = cv2.imencode('.jpg', frame)

    # Encode the buffer in base64 binary format and decode it to a string
    return base64.b64encode(buffer).decode('utf-8')

def test_emotion_detection(interval=5):
    """
    Test the emotion detection model by capturing frames from the webcam
    and sending them to the API for prediction. 

    Args:
        interval (int): The interval between frames in seconds. Default is 5.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break
                
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
                
                # Print detailed response information
                print(f"Status Code: {response.status_code}")
                
                # Print the result
                if response.status_code == 200:
                    result = response.json()
                    print(f"Detected emotion: {result.get('emotion')}")
                    print("Probabilities:", result.get('probabilities'))
                else:
                    print(f"Error: Server returned status code {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

            # Wait for interval seconds before next frame
            time.sleep(interval)
            
            # Wait for keyboard input for 1ms and break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_emotion_detection()