from ultralytics import YOLO
import torch
import numpy as np
import os
import functools
import sys
import warnings
import cv2
import matplotlib.pyplot as plt

# Optional: suppress warnings
warnings.filterwarnings("ignore")

# Utility to suppress print output
def disable_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result
    return wrapper

@disable_print
class YOLOSegmenter:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @disable_print
    def detect_objects(self, img, scoreThreshold=0.25):
        results = self.model.predict(source=img, conf=scoreThreshold, verbose=False)

        if not results or not results[0].masks:
            return []

        predictions = results[0]
        bboxes = predictions.boxes.xyxy.cpu().numpy()  # (N, 4)
        scores = predictions.boxes.conf.cpu().numpy()  # (N,)
        masks = predictions.masks.data.cpu().numpy()   # (N, H, W)

        detections = []
        for i in range(len(scores)):
            if scores[i] >= scoreThreshold:
                detection = {
                    'bbox': list(map(int, bboxes[i])),  # [x1, y1, x2, y2]
                    'score': float(scores[i]),
                    'mask': masks[i]
                }
                detections.append(detection)

        return detections

