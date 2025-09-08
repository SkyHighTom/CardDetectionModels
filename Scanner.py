
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json
import datetime

# def find_matchBak(hash_a):
#     hash_dict = get_hashes('hashes64x64.txt')
#     best_match = None
#     min_similarity = 14 # lower is more exact
#
#     for card_id, hash_b in hash_dict.items():
#         similarity = hash_a - hash_b # lower = better
#         if similarity < min_similarity:
#             min_similarity = similarity
#             best_match = card_id
#
#     if best_match is not None:
#         hash_b = hash_dict[best_match]
#         similarity = hash_a - hash_b
#     return best_match


def write_track_id(image, detections):
    for detection in detections:
        bbox = detection['bbox']
        track_id = detection.get('track_id')
        if track_id is None:
            continue
        x1, y1, x2, y2 = bbox
        # cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def write_card_labels(image, detections):
    for detection in detections:
        if 'match' not in detection or detection['match'] is None:
            continue
        bbox = detection['bbox']
        # Calculate the center of the bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        center = (center_x, center_y)
        write_label(image, center, detection['match'])


def write_label_rotated(img, loc, text, rotation=-20):
    # Define the main text and its properties
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .2
    thickness = 2
    text_color = (255, 255, 255)

    # Calculate the size of the main text
    text_size, _ = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Define the position and rotation of the main text
    [main_x, main_y] = loc

    # Calculate the rotation matrix for the main text
    rotation_matrix = cv2.getRotationMatrix2D((main_x, main_y), rotation, 1)

    # Create a black image with the same size as the input image
    text_img = np.zeros_like(img)

    # Add the main text to the text image with rotation
    cv2.putText(text_img, text, (main_x, main_y), font_face, font_scale, text_color, thickness, cv2.LINE_AA)
    rotated_text_img = cv2.warpAffine(text_img, rotation_matrix, (img.shape[1], img.shape[0]))

    # Overlay the rotated main text on the input image
    result = cv2.add(img, rotated_text_img)

    return result


def write_label(image, loc, text):
    [center_x, center_y] = loc
    font=cv2.FONT_HERSHEY_SIMPLEX
    font_scale=.6
    text_color=(0, 0, 0)
    background_color=(255, 255, 255)
    thickness=2

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position to start drawing the text (centered)
    text_x = int(center_x - text_size[0] / 2)
    text_y = int(center_y + text_size[1] / 2)

    # Calculate the bounding box for the background rectangle
    background_left = text_x
    background_top = text_y - text_size[1]
    background_right = text_x + text_size[0]
    background_bottom = text_y + 2

    # Draw the filled rectangle as the background
    cv2.rectangle(image, (background_left, background_top), (background_right, background_bottom), background_color, -1)

    # Write the text on top of the background
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness)


def draw_boxes(image, detections):
    for detection in detections:
        draw_box(image, detection['bbox'])


def draw_masks(image, detections):
    for detection in detections:
        draw_mask(image, detection['mask'])


def process_masks_to_cards(image, detections, rotation_model):
    for detection in detections:
        # Skip if match is already found and tracked
        if 'match' in detection:
            continue
        card_images = perspective_transform(image, detection['mask'], detection['bbox'], rotation_model)
        if card_images is not None:
            #print('not none')
            detection['card_images'] = card_images

def read_frame(camera, size):
    # Read a frame from the webcam
    ret, frame = camera.read()

    # Check if the frame is successfully read
    if not ret:
        print("Failed to read frame from the webcam")
        return None

    # Resize the frame to the desired size
    resized_frame = cv2.resize(frame, (size, size))
    return resized_frame


# Read and resize the image
def read_image(img, width, height):
    img = cv2.imread(img)
    img = cv2.resize(img, (height, width))
    return img


def show_contour(contour, original_image):
    # Create a blank canvas with the same size and aspect ratio as the original image
    canvas = np.zeros_like(original_image)

    # Draw the contour on the canvas
    cv2.drawContours(canvas, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Show the canvas
    cv2.imshow('Contour', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspective_transform(image, mask, bbox, rotation_model):
    # Convert the boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255

    # Visualize the mask
    h, w = mask_uint8.shape
    mask_visualized = np.zeros((h, w, 3), dtype=np.uint8)
    # Set the background pixels to black
    mask_visualized[:, :] = [0, 0, 0]
    # Set the object mask pixels to red
    mask_visualized[mask_uint8 != 0] = [0, 0, 255]
    # viewer.VideoFrameBuilder().add_image(mask_visualized, 2, "Segmentation Mask")

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area from largest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_with_detections = image.copy()

    # Approximate the contours to get the four corners of the card
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # Rearrange the points in the approx array if needed
            approx = reorder_points(approx)

            # Draw the contour on the image
            cv2.drawContours(image_with_detections, [approx], -1, (0, 0, 255), 2)

            # Draw the corners on the image
            circle_size = 8
            for point in approx:
                x, y = point.ravel()
                cv2.circle(image_with_detections, (x, y), circle_size, (0, 255, 0), -1)

            # viewer.VideoFrameBuilder().add_image(image_with_detections, 2, "Edge & Corner Detection")

            # Perform perspective transform
            width, height = get_card_dimensions(approx)
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(approx.astype(np.float32), dst)

            warped = cv2.warpPerspective(image, M, (int(width), int(height)))

            # Rotate the warped image to ensure portrait orientation
            #if width > height:
            #    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            # Resize the warped image to fill the canvas without maintaining aspect ratio
            warped_stretched = cv2.resize(warped, (250, 350))
            results = rotation_model.predict(source=Image.fromarray(warped_stretched), verbose=False)
            predicted_class = int(results[0].probs.top1)
            if predicted_class == 3:
                warped_stretched = cv2.rotate(warped_stretched, cv2.ROTATE_90_CLOCKWISE)
                warped_stretched = cv2.resize(warped_stretched, (250, 350))
            elif predicted_class == 1:
                warped_stretched = cv2.rotate(warped_stretched, cv2.ROTATE_90_COUNTERCLOCKWISE)
                warped_stretched = cv2.resize(warped_stretched, (250, 350))
            elif predicted_class == 2:
                warped_stretched = cv2.rotate(warped_stretched, cv2.ROTATE_180)
                warped_stretched = cv2.resize(warped_stretched, (250, 350))
            return warped_stretched

    return None


def reorder_points(points):
    # Calculate centroids
    centroids = np.mean(points, axis=0)

    # Sort points based on distance from centroids
    points_sorted = sorted(points, key=lambda x: np.arctan2(x[0][1] - centroids[0][1], x[0][0] - centroids[0][0]))

    return np.array(points_sorted)


def get_card_dimensions(corners):
    # Calculate the width and height of the card
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])
    return width, height


def draw_box(img, bbox):
    xmin, ymin, xmax, ymax = map(int, bbox)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


def draw_mask(original_image, mask):
    color = (0, 0, 255)
    alpha = 0.4

    # Convert mask to binary mask (0s and 255s)
    binary_mask = np.array(mask, dtype=np.uint8)

    # Create a copy of the original image
    overlay = original_image.copy()

    # Apply the color to the mask region
    overlay[binary_mask != 0] = color

    # Blend the overlay with the original image using alpha blending
    result = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)

    # Update the original image with the result
    original_image[:, :] = result


def save_screenshot(img):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Write the image to a file
    screenshot_filename = f'output/screenshots/screenshot_{current_time}.png'
    cv2.imwrite(screenshot_filename, img)
    print(f"Screenshot saved to '{screenshot_filename}'")


import matplotlib.pyplot as plt
import numpy as np

def show_image_wait(img):

    # Convert BGR (OpenCV default) to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis
    plt.title('Image')
    plt.show()