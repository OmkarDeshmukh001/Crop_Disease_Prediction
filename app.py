import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Crop Disease AI", layout="centered")

MODEL_PATH = "plant_disease_model.keras"

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

LABELS = ["Early Blight", "Late Blight", "Healthy"]

DISEASE_INFO = {
    "Potato___Early_blight": {"status": "Early Blight", "severity": "Moderate"},
    "Potato___Late_blight":  {"status": "Late Blight",  "severity": "High"},
    "Potato___healthy":      {"status": "Healthy",      "severity": "None"}
}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
IMG_SIZE = model.input_shape[1:3]


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


# ── UI ──────────────────────────────────────
st.title("🌿 Crop Disease Detection")
st.caption("AI-powered Potato Leaf Diagnosis")
st.divider()

uploaded_file = st.file_uploader(
    "Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(file_bytes))

    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            preds = model.predict(preprocess_image(file_bytes))[0]
            class_idx = np.argmax(preds)
            confidence = float(np.max(preds)) * 100
            info = DISEASE_INFO[CLASS_NAMES[class_idx]]

        st.divider()
        st.subheader("Result")
        st.write(f"**Diagnosis:** {info['status']}")
        st.write(f"**Severity:** {info['severity']}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(confidence / 100)

        if confidence < 70:
            st.warning("Low confidence — try a clearer image.")

        st.divider()
        st.subheader("Prediction Breakdown")
        fig, ax = plt.subplots()
        ax.bar(LABELS, preds * 100, color=["#f4a261", "#e76f51", "#2a9d8f"])
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim([0, 100])
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig)

st.divider()
st.caption("Built with TensorFlow + Streamlit")
