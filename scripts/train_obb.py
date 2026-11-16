"""Fine-tune YOLOv8n-OBB on the Roboflow polygons (classes: noir, rouge)."""
import sys
from ultralytics import YOLO

data = sys.argv[1]
model = YOLO("models_raw/yolov8n-obb.pt")       # DOTA-pretrained oriented-box model
model.train(
    data=data, imgsz=416, epochs=70, patience=20, batch=16, device="cpu", workers=2,
    project="runs_obb", name="v1", exist_ok=True, seed=0,
    fliplr=0.0, flipud=0.0,      # a mirror would put red on the left: forbidden
    degrees=180.0,               # meters are photographed at any angle
    hsv_v=0.4, scale=0.4, mosaic=1.0, close_mosaic=10, plots=True,
)
