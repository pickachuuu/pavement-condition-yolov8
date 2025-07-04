from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())         # should return: True
    print(torch.cuda.get_device_name(0))     # should return: NVIDIA RTX 4060
    # Initialize a new YOLO model from scratch (no pre-trained weights)
    model = YOLO(model="yolov8m.yaml")

    # Start training on your custom dataset
    model.train(
        data="data.yaml",  # replace with actual path to your YAML file
        epochs=50,
        imgsz=640,
        batch=8,         # recommended for yolov8m with 8GB VRAM
        cache="disk",       # optional: speeds up training after first epoch
        workers=4
    )