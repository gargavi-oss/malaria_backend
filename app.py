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
            detail="Image does not look like an RBC cell crop."
        )

def predict_image(image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        idx = torch.argmax(probs).item()
        conf = probs[0][idx].item()
    return class_names[idx], conf

class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, module, inp, out):
        self.activations = out
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def generate(self, input_tensor, class_idx=None):
        torch.set_grad_enabled(True)
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        output[:, class_idx].backward(retain_graph=True)
        A = self.activations[0]   
        dA = self.gradients[0]    
        dA2 = dA ** 2
        dA3 = dA ** 3
        S = A.sum(dim=(1, 2), keepdim=True)
        eps = 1e-7
        alpha = dA2 / (2 * dA2 + S * dA3 + eps)
        weights = (alpha * torch.relu(dA)).sum(dim=(1, 2))
        cam = (weights[:, None, None] * A).sum(dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.detach().cpu().numpy()


def encode_heatmap(cam):
    plt.figure(figsize=(3, 3))
    plt.imshow(cam, cmap="jet")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    validate_rbc_image(image)
    label, conf = predict_image(image)
    return {
        "prediction": label,
        "confidence": round(conf * 100, 2)
    }

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    validate_rbc_image(image)
    label, conf = predict_image(image)
    img_tensor = transform(image).unsqueeze(0)
    cam_generator = GradCAMPP(model, model.layer4[-1])
    cam = cam_generator.generate(img_tensor)
    return {
        "prediction": label,
        "confidence": round(conf * 100, 2),
        "gradcam": encode_heatmap(cam)
    }

@app.get("/")
def home():
    return {"message": "Malaria API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
