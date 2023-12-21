# importing libraries
import streamlit as st
import numpy as np
import cv2
import cv2.data
from PIL import Image
from keras_facenet import FaceNet
import pickle


# Initializations
facenet = FaceNet()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('./FRS_Encoder.pkl', 'rb') as ce:
    encoder = pickle.load(ce)
with open('./FRS_SVM_Model.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def extract_face(filename):
    # define the required output size
    required_size = (160, 160)
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # perform face detection
    results = detector.detectMultiScale(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def get_embedding(model, face):
    # Scale pixel values
    face_pixels = face.astype('float32')
    # Transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # Make prediction to get embedding
    yhat = model.embeddings(samples)
    return yhat[0]


# main program: Create a streamlit interface
def main():
    # Setting a title to our app
    st.title('Face Recognition System using FaceNet')
    # Load picture you want to recognize
    filename = st.file_uploader("Upload the picture you want to recognize:")
    if filename is not None:
        face_array = extract_face(filename)
        # Get the embedding of the face
        embedding = get_embedding(facenet, face_array)
        # Reshape the embedding
        embedding = embedding.reshape(1, -1)
        # Predict the label of the face
        label = svm_model.predict(embedding)
        # Get the name of the person
        name = encoder.inverse_transform(label)
        # Display the image
        st.image(filename, use_column_width=True)
        # Display the name of the person
        st.success(f'The person in the picture is {name[0]}')
    else:
        st.error("No image uploaded")


# Run the main
if __name__ == "__main__":
    main()
