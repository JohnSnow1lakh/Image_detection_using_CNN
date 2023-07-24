import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf


# Load the pre-trained CNN model
model = load_model(r"C:/Users/Harshitkumar Dinesh/Desktop/Emotion_Detection_CNN-main/emotion_detection_model.h5")


# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(r"C:/Users/Harshitkumar Dinesh/Desktop/Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml")



# Function to detect and classify emotions
def detect_emotion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to match the input size of the model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape(1, 48, 48, 1)
        face_roi = face_roi.astype('float32')
        face_roi /= 255

        # Predict the emotion
        emotion_probabilities = model.predict(face_roi)[0]
        emotion_index = np.argmax(emotion_probabilities)
        emotion_label = emotion_labels[emotion_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted emotion above the face rectangle
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# Open the video capture
video_capture = cv2.VideoCapture(0)

# Continuously process frames from the video capture
while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Detect and classify emotions in the frame
    frame = detect_emotion(frame)

    # Display the frame with emotion information
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
