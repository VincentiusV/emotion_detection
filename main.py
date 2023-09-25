import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import psutil
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your Emotion Detection model (.h5 file)
emotion_model = keras.models.load_model('C:/Kecilin/emotion_detection/V1/checkpoint/best_model.h5')

# Define emotion labels (adjust these labels to match your model's output)
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

# Create a VideoCapture object to access the webcam (usually the default camera, index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the frame width and height (you can adjust these values)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize to match the input size of your emotion model
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add a channel dimension (grayscale)

        # Normalize the input
        face_roi = face_roi / 255.0

        # Make an emotion prediction using your model
        emotion_prediction = emotion_model.predict(np.array([face_roi]))
        emotion_index = np.argmax(emotion_prediction[0])
        emotion_text = emotion_labels[emotion_index]

        # Draw a rectangle around the detected face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Get CPU usage
    cpu_usage = psutil.cpu_percent()

    # Display FPS and CPU usage on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU: {cpu_usage:.2f}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()