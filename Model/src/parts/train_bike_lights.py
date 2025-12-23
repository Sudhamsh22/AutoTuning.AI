from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def main():
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data= "Bike_LIGHTS_output",
        epochs=60,
        imgsz=224,
        batch=8,
        device="cuda",
        augment=True,
        name="lights_bike_v1"
    )

if __name__ == "__main__":
    main()
