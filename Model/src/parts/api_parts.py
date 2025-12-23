from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pathlib import Path
from PIL import Image
import io
from typing import Dict, Any

from ultralytics import YOLO
from src.parts.part_knowledge import PART_PURPOSE


app = FastAPI(title="Parts Identification API", version="2.0")

BASE_DIR = Path(__file__).resolve().parents[2]
DEVICE = "cuda"
def normalize(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


CAR_MODELS = {
    "engine": BASE_DIR / "runs/classify/engine_v13/weights/best.pt",
    "braking": BASE_DIR / "runs/classify/braking_v12/weights/best.pt",
    "electrical": BASE_DIR / "runs/classify/electrical_v1/weights/best.pt",
    "fuel": BASE_DIR / "runs/classify/fuel_v1/weights/best.pt",
}

BIKE_MODELS = {
    "engine": BASE_DIR / "runs/classify/engine_bike_v1/weights/best.pt",
    "braking": BASE_DIR / "runs/classify/braking_bike_v1/weights/best.pt",
    "fuel": BASE_DIR / "runs/classify/Bike_FUEL_v1/weights/best.pt",
    "cooling": BASE_DIR / "runs/classify/cooling_bike_v12/weights/best.pt",
    "electrical": BASE_DIR / "runs/classify/electrical_bike_v1/weights/best.pt",
    "lights": BASE_DIR / "runs/classify/lights_bike_v12/weights/best.pt",
    "transmission": BASE_DIR / "runs/classify/transmission_bike_v15/weights/best.pt",
    "body": BASE_DIR / "runs/classify/body_v1/weights/best.pt",
}


def load_models(model_paths: Dict[str, Path]) -> Dict[str, YOLO]:
    models = {}
    for k, p in model_paths.items():
        if p.exists():
            models[k] = YOLO(str(p))
    return models


BACKENDS = {
    "car": load_models(CAR_MODELS),
    "bike": load_models(BIKE_MODELS),
}


def load_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    return Image.open(io.BytesIO(data)).convert("RGB")


def infer_yolo(model: YOLO, image: Image.Image, topk: int) -> Dict[str, Any]:
    res = model.predict(image, device=DEVICE, verbose=False)
    probs = res[0].probs
    names = res[0].names

    scores = probs.data.cpu().numpy()
    best_idx = int(probs.top1)

    idxs = scores.argsort()[-topk:][::-1]

    return {
        "part": names[best_idx],
        "confidence": float(probs.top1conf),
        "alternatives": [
            {"part": names[int(i)], "confidence": float(scores[int(i)])}
            for i in idxs
        ],
    }


def auto_detect(image: Image.Image, models: Dict[str, YOLO], topk: int):
    best = None
    best_type = None

    for part_type, model in models.items():
        res = infer_yolo(model, image, topk=1)
        if best is None or res["confidence"] > best["confidence"]:
            best = res
            best_type = part_type

    if best is None:
        raise HTTPException(status_code=500, detail="Inference failed")

    best["part_type"] = best_type
    return best


@app.post("/identify-part")
def identify_part(
    vehicle_type: str = Query(..., description="car or bike"),
    image: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=10),
):
    vehicle_type = vehicle_type.lower().strip()

    if vehicle_type not in BACKENDS:
        raise HTTPException(status_code=400, detail="vehicle_type must be car or bike")

    models = BACKENDS[vehicle_type]
    if not models:
        raise HTTPException(status_code=500, detail="No models loaded")

    img = load_image(image)
    result = auto_detect(img, models, topk)

    part = result["part"]
    purpose = (
    PART_PURPOSE
    .get(vehicle_type, {})
    .get(part)
    or PART_PURPOSE
    .get(vehicle_type, {})
    .get(part.lower())
    or "Purpose not available."
)


    return {
        "vehicle_type": vehicle_type,
        "system": result["part_type"],
        "part": part,
        "confidence": result["confidence"],
        "purpose": purpose,
        "alternatives": result["alternatives"],
        "method": "yolo-auto",
    }
