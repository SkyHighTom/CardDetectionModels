from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a trained YOLOv8 segmentation model
model = YOLO("runs/segment/train9/weights/best.pt")  # Note: 'segment', not 'detect'

# Run prediction on an image
results = model.predict(source="images/test/IMG_6419.jpg", conf=0.10)

# Extract and plot segmentation result
annotated_frame = results[0].plot()  # includes masks, labels, and boxes

# Show the prediction using matplotlib
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLOv8 Segmentation Prediction")
plt.show()
