import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


# Image parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224


# Get the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"


# Load the pre-trained model
model = tf.keras.models.load_model(model_path)


# Load the class names file
classes_path = f"{working_dir}/classes.json"
classes = json.load(open(classes_path))


# Function to Load and Preprocess the Image
def load_and_preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image(img_path):
    preprocessed_img = load_and_preprocess_image(img_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = classes[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App

# Title of the app
st.title('🌿 Plant Disease Predictor')

# Image upload dropbox
uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    col1, col2 = st.columns(2)

    # Show the image
    with col1:
        resized_img = image.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(resized_img)

    # Display the predict button and the results
    with col2:
        if st.button('Predict'):
            prediction = predict_image(uploaded_img)
            plant_name, disease_name = prediction.split('___')

            st.markdown("### Prediction 🌱")
            st.markdown(f"**Plant:** {plant_name.replace('_', ' ')}")
            if disease_name == "healthy":
                st.markdown(f"**Disease:** None (Healthy)")
            else:
                st.markdown(f"**Disease:** {disease_name.replace('_', ' ')}")
