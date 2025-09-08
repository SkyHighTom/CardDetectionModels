import os
import json
import cv2

image_dir = "TrainingData/images/val"
annotation_dir = "TrainingData/annotations"
label_output_dir = "TrainingData/labels/val"
os.makedirs(label_output_dir, exist_ok=True)

for image_file in os.listdir(image_dir):
    if not image_file.endswith(".jpg"):
        continue

    base_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(image_dir, image_file)
    json_path = os.path.join(annotation_dir, base_name + ".json")
    label_path = os.path.join(label_output_dir, base_name + ".txt")

    if not os.path.exists(json_path):
        print(f"No annotation for {image_file}")
        continue

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for obj in data.get("objects", []):
        corners = obj["corners"]
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        box_width = (x_max - x_min) / w
        box_height = (y_max - y_min) / h

        class_id = 0  # assuming one class
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

print("✅ YOLO annotations generated.")
