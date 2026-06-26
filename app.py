import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺")

st.title("🩺 Breast Cancer Detection Using CNN")
st.write("Upload a histopathology image to predict Benign or Malignant.")

model = tf.keras.models.load_model("breast_cancer_cnn_model.h5")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write("Prediction Score:", float(prediction))

    if prediction > 0.5:
        st.error("Result: Malignant Cancer")
    else:
        st.success("Result: Benign Cancer")