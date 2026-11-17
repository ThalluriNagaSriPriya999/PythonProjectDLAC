
import io
import torch
import torch.nn as nn

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms


app = FastAPI(
    title="Skin Lesion Triage Gateway"
)

# 1. Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. HAM10000 classes
classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# 3. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Load trained ResNet18 model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

try:
    checkpoint = torch.load(
        "models/skin_lesion_resnet18.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    model_loaded = True

except (FileNotFoundError, RuntimeError, ValueError):
    model_loaded = False


@app.get("/")
def home():
    return {
        "project": "Skin Lesion Triage",
        "message": "Skin Lesion FastAPI Gateway is running!",
        "status": "success",
        "device": device,
        "model_loaded": model_loaded,
        "available_endpoints": [
            "/health",
            "/detect",
            "/docs"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": device
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # 1. Validate request
    if file.content_type not in (
        "image/jpeg",
        "image/png"
    ):
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image"
        )

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Trained model weights not found"
        )

    # 2. Decode image
    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be 10 MB or smaller"
            )

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

    # 3. Preprocess image
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # 4. Model inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities, dim=1
    )

    predicted_class = classes[predicted_index.item()]

    # 5. Format predictions
    predictions = {
        classes[i]: round(
            probabilities[0][i].item(), 4
        )
        for i in range(len(classes))
    }

    return {
        "project": "Skin Lesion Triage",
        "image_size": {
            "width": image.width,
            "height": image.height
        },
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "all_class_probabilities": predictions,
        "triage_note": (
            "Research-only prediction. "
            "This result is not a medical diagnosis. "
            "Seek assessment from a qualified dermatologist."
        )
    }
