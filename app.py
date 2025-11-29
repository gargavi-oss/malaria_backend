import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torchvision.models as models

# ---------------------
# FASTAPI INSTANCE
# ---------------------
app = FastAPI()

# Allow CORS (useful for React/Vercel frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# MODEL DEFINITION
# ---------------------
def get_resnet18(num_classes=2):
    model = models.resnet18(pretrained=False)

    # Freeze layers except layer4
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------
# LOAD MODEL
# ---------------------
model_path = "resnet18_malaria_weights.pth"

model = get_resnet18(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

class_names = ["Parasitized", "Uninfected"]

# ---------------------
# IMAGE TRANSFORM
# ---------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------------
# PREDICT FUNCTION
# ---------------------
def predict_image(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    return class_names[pred_idx], confidence


# ---------------------
# FASTAPI ENDPOINT
# ---------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    label, confidence = predict_image(image)

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }


@app.get("/")
def home():
    return {"message": "Malaria ResNet18 API is running!"}
