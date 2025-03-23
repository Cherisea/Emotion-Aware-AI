import cv2
import base64
import requests
import json

def encode_frame(frame):
    """Convert OpenCV frame to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def test_emotion_detection():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break
                
            # Convert frame to base64
            frame_data = encode_frame(frame)
            
            # Prepare the request
            payload = {
                "frame": f"data:image/jpeg;base64,{frame_data}"
            }
            
            # Make the request to your local server
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
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
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_emotion_detection()