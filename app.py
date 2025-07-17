# app.py

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import base64

# New imports for lifespan manager
from contextlib import asynccontextmanager

# Import model-specific libraries
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your custom UNet from the new file
from model import UNet

# --- Application Setup ---
# Global dictionary to hold the models
MODELS = {}
# This line ensures PyTorch runs on CPU if CUDA is not available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Step 1: Define the Lifespan Context Manager (New Method) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Models are loaded on startup and cleared on shutdown.
    """
    # --- Startup Logic ---
    print("Loading models... This may take a moment.")
    
    # Load Custom UNet
    custom_unet = UNet(in_channels=3, num_classes=1)
    custom_unet.load_state_dict(torch.load("model_weights/best_model.pth", map_location=DEVICE))
    MODELS["custom_unet"] = custom_unet.to(DEVICE).eval()

    # Load SMP UNet
    smp_unet = smp.Unet(encoder_name="efficientnet-b7", encoder_weights=None, in_channels=3, classes=1)
    smp_unet.load_state_dict(torch.load("model_weights/best_model_smp.pth", map_location=DEVICE))
    MODELS["smp_unet"] = smp_unet.to(DEVICE).eval()

    # Load SegFormer
    segformer = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True
    )
    segformer.load_state_dict(torch.load("model_weights/best_segformer_model.pth", map_location=DEVICE))
    MODELS["segformer"] = segformer.to(DEVICE).eval()

    print(f"Models loaded successfully on device: {DEVICE}")

    yield # The application runs after this point

    # --- Shutdown Logic ---
    print("Clearing models and releasing resources.")
    MODELS.clear()


# --- Step 2: Instantiate the FastAPI App with Lifespan ---
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- Preprocessing and Postprocessing Functions (Unchanged) ---
def preprocess_unet(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image.to(DEVICE)

def preprocess_segformer(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    processed = transform(image=image)
    return processed['image'].unsqueeze(0).to(DEVICE)

def postprocess_mask(raw_pred, model_type):
    if model_type == 'segformer':
        pred_mask = torch.argmax(raw_pred, dim=1).squeeze(0)
    else:
        pred_mask = (torch.sigmoid(raw_pred) > 0.5).squeeze(0).squeeze(0)
    mask_np = pred_mask.cpu().numpy().astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_np, mode='L')
    buff = io.BytesIO()
    mask_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# --- API Endpoints (Unchanged) ---
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, model_choice: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    model = MODELS[model_choice]
    
    if model_choice == 'segformer':
        input_tensor = preprocess_segformer(image_bytes)
        with torch.no_grad():
            low_res_logits = model(pixel_values=input_tensor).logits
            outputs = F.interpolate(
                low_res_logits,
                size=input_tensor.shape[-2:], # Dynamically get H, W from input
                mode='bilinear',
                align_corners=False
            )
    else:
        input_tensor = preprocess_unet(image_bytes)
        with torch.no_grad():
            outputs = model(input_tensor)
            
    output_mask_base64 = postprocess_mask(outputs, model_choice)
    input_image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "input_image": input_image_base64,
        "output_mask": output_mask_base64,
        "model_used": model_choice
    }

# --- Main entry point (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
