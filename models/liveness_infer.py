import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/liveness_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

parser = argparse.ArgumentParser(description="Fingerprint Liveness Inference")
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

img = Image.open(args.image).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(img)
    probs = torch.softmax(out, dim=1)[0]
    pred = torch.argmax(probs).item()

classes = ["LIVE", "SPOOF"]
confidence = probs[pred].item() * 100

print("=" * 40)
print("Fingerprint Liveness Prediction")
print("=" * 40)
print(f"Image      : {args.image}")
print(f"Prediction : {classes[pred]}")
print(f"Confidence : {confidence:.2f}%")
print("=" * 40)