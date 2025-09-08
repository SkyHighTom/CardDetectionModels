from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="D:/ImageWarping/SegmentationTraining/data.yaml", 
        epochs=20,
        imgsz=640,
        batch=16,
        device=0
    )