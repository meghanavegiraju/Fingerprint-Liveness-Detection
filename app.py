import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Fingerprint Liveness Detection",
    page_icon="🔐",
    layout="centered"
)

st.title("🔐 Fingerprint Liveness Detection")
st.write("Upload a fingerprint image to detect whether it is LIVE or SPOOF.")

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "models/liveness_model.pth"

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

st.success("✅ Model Loaded Successfully")

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a fingerprint image",
    type=["jpg", "jpeg", "png", "bmp"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

    classes = ["LIVE", "SPOOF"]

    predicted_class = classes[prediction.item()]
    confidence_score = confidence.item() * 100

    st.markdown("---")

    if predicted_class == "LIVE":
        st.success(f"✅ Prediction: {predicted_class}")
    else:
        st.error(f"❌ Prediction: {predicted_class}")

    st.metric("Confidence", f"{confidence_score:.2f}%")

    st.progress(float(confidence.item()))

    st.write("### Class Probabilities")
    st.write(f"🟢 LIVE : {probabilities[0][0].item()*100:.2f}%")
    st.write(f"🔴 SPOOF : {probabilities[0][1].item()*100:.2f}%")