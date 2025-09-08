import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Paths
image_dir = "SegmentationTraining/images/train"
label_dir = "SegmentationTraining/labels/train"

for i in range(10):
    image_path = os.path.join(image_dir, f"synthetic_{i:04d}.jpg")
    label_path = os.path.join(label_dir, f"synthetic_{i:04d}.txt")

    print(f"Testing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        continue

    height, width, _ = image.shape

    # Make a copy to draw all annotations on
    annotated_image = image.copy()

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                coords = parts[1:]

                if len(coords) % 2 != 0:
                    print(f"Line {line_num} in {label_path} has invalid number of coordinates.")
                    continue

                # Convert normalized coords to pixel coords
                points = []
                for j in range(0, len(coords), 2):
                    x = int(coords[j] * width)
                    y = int(coords[j + 1] * height)
                    points.append((x, y))

                # Draw polygon and points
                cv2.polylines(annotated_image, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                for (x, y) in points:
                    cv2.circle(annotated_image, (x, y), 3, (255, 0, 0), -1)

                # Draw class label near first point
                cv2.putText(annotated_image, str(cls), (points[0][0], points[0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"YOLO Segmentation: synthetic_{i:04d}.jpg")
    plt.show()
