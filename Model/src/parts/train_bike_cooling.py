from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

def main():
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data=str(BASE / "Bike_cooling_output"),
        epochs=60,
        imgsz=224,
        batch=8,
        device=0,
        augment=True,
        name="cooling_bike_v1"

    )

if __name__ == "__main__":
    main()
