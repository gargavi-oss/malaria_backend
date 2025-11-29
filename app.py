import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_resnet18(num_classes=2):
    model = models.resnet18(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

model_path = "resnet18_malaria_weights.pth"
model = get_resnet18(2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

class_names = ["Parasitized", "Uninfected"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def validate_rbc_image(image: Image.Image):
    w, h = image.size
    if w > 600 or h > 600:
        raise HTTPException(
            status_code=400,
            detail="Please upload an RBC microscope image (not a normal photo)."
        )
    ratio = max(w, h) / min(w, h)
    if ratio > 1.5:
        raise HTTPException(
            status_code=400,
            detail="Image does not look like an RBC cell crop. Please upload a valid microscope image."
        )
    return True

def predict_image(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()
    return class_names[pred_idx], confidence

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax()
        output[0, class_idx].backward()
        grads = self.gradients           # [C,H,W]
        acts = self.activations          # [C,H,W]
        weights = torch.mean(grads, dim=(1, 2))  # GAP over gradients
        cam = torch.sum(weights[:, None, None] * acts, dim=0)  # Weighted sum
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-9
        return cam.detach().cpu().numpy()


def encode_heatmap(cam):
    plt.figure(figsize=(3, 3))
    plt.imshow(cam, cmap='jet', alpha=1.0)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    validate_rbc_image(image)
    label, confidence = predict_image(image)
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    validate_rbc_image(image)
    label, confidence = predict_image(image)
    img_tensor = transform(image).unsqueeze(0)
    gradcam = GradCAM(model, model.layer4[-1])
    cam = gradcam.generate(img_tensor)
    cam_base64 = encode_heatmap(cam)
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "gradcam": cam_base64
    }

@app.get("/")
def home():
    return {"message": "Malaria API is running!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
