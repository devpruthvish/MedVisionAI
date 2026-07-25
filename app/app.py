import os
import torch
import streamlit as st
from PIL import Image

from model import load_model
from utils import preprocess_image, CLASS_NAMES

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("🩺 Medical AI")

st.sidebar.markdown("---")

st.sidebar.header("Project Information")

st.sidebar.write("""
**Model:** ResNet18 (Transfer Learning)

**Framework:** PyTorch

**Frontend:** Streamlit

**Classes:**
- NORMAL
- PNEUMONIA

**Dataset:**
Chest X-Ray Images
""")

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

# =====================================================
# Main Title
# =====================================================

st.title("🩺 AI Pneumonia Detection using ResNet18")

st.markdown(
    "Upload a Chest X-Ray image to predict whether the patient has **Pneumonia** or is **Normal**."
)

st.markdown("---")

# =====================================================
# Load Model
# =====================================================

MODEL_PATH = os.path.join("models", "resnet18_pneumonia.pth")

model = load_model(MODEL_PATH)

# =====================================================
# Upload Image
# =====================================================

uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# Prediction
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            use_container_width=True
        )

    image_tensor = preprocess_image(image)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[prediction.item()]

    normal_prob = probabilities[0][0].item() * 100
    pneumonia_prob = probabilities[0][1].item() * 100

    with col2:

        st.subheader("Prediction Result")

        if predicted_class == "NORMAL":
            st.success(f"Prediction : {predicted_class}")
        else:
            st.error(f"Prediction : {predicted_class}")

        st.write(f"### Confidence : {confidence.item()*100:.2f}%")

        st.progress(float(confidence.item()))

        st.markdown("---")

        st.subheader("Class Probabilities")

        st.write(f"🟢 NORMAL : {normal_prob:.2f}%")

        st.progress(normal_prob / 100)

        st.write(f"🔴 PNEUMONIA : {pneumonia_prob:.2f}%")

        st.progress(pneumonia_prob / 100)

st.markdown("---")

st.caption("Developed using PyTorch • ResNet18 • Streamlit")