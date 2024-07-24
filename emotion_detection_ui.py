import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Load the model from JSON file
with open('model_b.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('model_b.h5')

# Emotion labels dictionary
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to preprocess the face image
def preprocess_face(image):
    # Resize the image to the required size
    image = cv2.resize(image, (48, 48))
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Repeat the grayscale channel to create an RGB image
    image = np.stack([image] * 3, axis=-1)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    return image

# Function to capture video and predict emotion
def video_stream():
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        emotion_prediction = model.predict(preprocessed_face)
        predicted_class = np.argmax(emotion_prediction, axis=1)[0]
        predicted_emotion = emotion_labels[predicted_class]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, video_stream)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create the main window
root = tk.Tk()
root.title("Emotion Detection")

# Create a label to display the video feed
lmain = Label(root)
lmain.pack()

# Start the video stream
video_stream()

# Start the Tkinter main loop
root.mainloop()

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
