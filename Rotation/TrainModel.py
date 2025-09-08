from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-cls.pt')

    model.train(
        data='.',
        epochs=20,
        imgsz=320,
        batch=8,
        device=0,
        plots=False
    )

if __name__ == "__main__":
    main()
