import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

# Load the saved model
@st.cache_resource
def load_mnist_model():
    return load_model("Model/mnist_cnn_model.h5")

model = load_mnist_model()

# Center align title and description
st.markdown(
    """
    <h1 style="text-align:center;">MNIST Digit Classifier</h1>
    <p style="text-align:center;">Choose a digit (0–9) or upload your own image</p>
    """,
    unsafe_allow_html=True,
)

# Sidebar for sample images
st.sidebar.header("Sample Images")
sample_dir = "test_images"

# Map filenames to numbers for sorting
file_order = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
}

sample_files = []
if os.path.exists(sample_dir):
    sample_files = sorted(
        os.listdir(sample_dir),
        key=lambda x: file_order.get(os.path.splitext(x)[0].lower(), 999)
    )

selected_sample = st.sidebar.selectbox(
    "Pick a sample digit image:", ["None"] + sample_files
)

uploaded_file = None

# If user picks a sample → load it
if selected_sample != "None":
    uploaded_file = os.path.join(sample_dir, selected_sample)
    st.sidebar.image(uploaded_file, caption="Original Image")  

# Or allow user upload
uploaded_upload = st.file_uploader("Upload your own digit...", type=["png", "jpg", "jpeg"])
if uploaded_upload is not None:
    uploaded_file = uploaded_upload
    st.sidebar.image(uploaded_file, caption="Original Image")  

# Run prediction if we have an image
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert("L")  

    # Preprocess the image
    img_resized = ImageOps.invert(image)  
    img_resized = img_resized.resize((28, 28))  
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Processed Image")
        st.image(img_resized, caption="Resized Input", use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.write(f"**Predicted Digit:** {predicted_class}")
        st.bar_chart(prediction[0])
