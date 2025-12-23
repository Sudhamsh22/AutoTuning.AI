from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def main():
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data= "ELECTRICAL_output",
        epochs=60,
        imgsz=224,
        batch=8,
        device="cpu",
        augment=True,
        name="electrical_v1"
    )

if __name__ == "__main__":
    main()
