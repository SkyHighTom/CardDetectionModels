import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import Detector
import Scanner
import matplotlib.pyplot as plt
from PIL import Image

# Settings
weights = "YugiohModel/runs/segment/train/weights/best.pt"
rotation_weights = "Rotation/runs/classify/train8/weights/best.pt"
rotation_model = YOLO(rotation_weights)
image_folder = 'YugiohCardTests'
width = 480
height = 640
scoreThreshold = 0.4
save_image = False
mirror = False
include_flipped = True  # currently not used, but left for consistency

def show_each_detection(detections, image_title=""):
    for i, detection in enumerate(detections):
        if 'card_images' in detection:
            card_rgb = cv2.cvtColor(detection['card_images'], cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(card_rgb)
            plt.axis('off')
            title = f'{image_title} - Detection {i + 1}' if image_title else f'Detection {i + 1}'
            plt.title(title)
            plt.show()

def main():
    print("Starting batch inference on folder")

    # Initialize the DetInferencer once
    detector = Detector.YOLOSegmenter(weights)

    # Loop through each file in the folder
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif')):
            continue  # Skip non-image files

        image_path = os.path.join(image_folder, filename)
        print(f"\nProcessing: {filename}")

        image_original = Scanner.read_image(image_path, width, height)
        image_copy = image_original.copy()

        detections = detector.detect_objects(image_original, scoreThreshold)

        Scanner.process_masks_to_cards(image_original, detections, rotation_model)
        show_each_detection(detections, image_title=filename)

        Scanner.draw_boxes(image_copy, detections)
        Scanner.draw_masks(image_copy, detections)

        if save_image:
            output_path = os.path.join('NewScans', f'processed_{filename}')
            cv2.imwrite(output_path, image_copy)
            print(f'Saved processed image: {output_path}')

        Scanner.show_image_wait(image_copy)

if __name__ == '__main__':
    main()
