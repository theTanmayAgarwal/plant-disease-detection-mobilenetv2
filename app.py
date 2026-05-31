import streamlit as st
import tensorflow as tf
import numpy as np
import json

from tensorflow.keras.preprocessing import image

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌱",
    layout="centered"
)

# Title
st.title("🌱 Plant Disease Detection")

st.info(
    "This model was trained using MobileNetV2 Transfer Learning on the PlantVillage dataset."
)

st.write(
    "Upload a plant leaf image and the model will identify the disease."
)

# Load Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "plant_disease_model.keras"
    )

model = load_my_model()

# Load Class Names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# File Upload
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Uploaded Image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=300
    )

    # Load Image
    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    prediction = prediction[0]

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    display_name = predicted_class.replace(
        "___",
        " - "
    ).replace(
        "_",
        " "
    )

    confidence = float(
        np.max(prediction)
    )

    # Main Prediction
    st.success(
        f"Prediction: {display_name}"
    )

    # Confidence
    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.progress(confidence)

    # Top 3 Predictions
    st.subheader("Top 3 Predictions")

    top3_idx = np.argsort(
        prediction
    )[-3:][::-1]

    for idx in top3_idx:

        disease_name = class_names[idx]

        disease_name = disease_name.replace(
            "___",
            " - "
        ).replace(
            "_",
            " "
        )

        score = prediction[idx] * 100

        st.write(
            f"• {disease_name}: {score:.2f}%"
        )