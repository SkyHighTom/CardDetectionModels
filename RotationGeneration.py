import os
import cv2
import random
import numpy as np

# Input
CARD_DIR = ["./images", "./CutMTGimages"]
NUM_IMAGES = 15000

# Output structure
BASE_DIR = "./Rotation"
SPLITS = ["train", "val"]
ROTATION_CLASSES = ["0", "1", "2", "3"]

# Create directory structure: Rotation/train/0, Rotation/train/1, ..., Rotation/val/3
for split in SPLITS:
    for cls in ROTATION_CLASSES:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Load from multiple folders
def load_images_from_folders(folders, valid_exts=('.png', '.jpg', '.jpeg')):
    all_files = []
    for folder in folders:
        all_files.extend(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
        )
    return all_files

def apply_jpeg_compression(img, quality=25):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def apply_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img

def degrade_image_randomly(img):
    choice = random.choice(["jpeg", "blur", "noise", "blur", "none", "none", "none", "jpeg", "jpeg", "blur", "blur", "none", "none", "none", "jpeg"])
    if choice == "jpeg":
        return apply_jpeg_compression(img, quality=random.randint(15, 40))
    elif choice == "blur":
        return apply_blur(img, kernel_size=random.choice([3, 5, 7]))
    elif choice == "noise":
        return apply_noise(img, std=random.randint(1, 5))
    else:
        return img  # no degradation


cards = load_images_from_folders(CARD_DIR)

# Helper to rotate card and assign label
def rotate_card(card):
    rand = random.random()
    if rand < 0.25:
        return card, 0
    elif rand < 0.5:
        return cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE), 1
    elif rand < 0.75:
        return cv2.rotate(card, cv2.ROTATE_180), 2
    else:
        return cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE), 3

# Generate images
for i in range(NUM_IMAGES):
    split = "train" if i < int(NUM_IMAGES * 0.8) else "val"

    # Get a valid card image
    while True:
        card_path = random.choice(cards)
        card = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
        if card is not None and card.shape[2] == 4:  # Ensure RGBA
            break

    rotated, tag = rotate_card(card)
    resized = cv2.resize(rotated, (250, 350))
    degraded = degrade_image_randomly(resized)

    # Save image to class-labeled folder
    class_dir = os.path.join(BASE_DIR, split, str(tag))
    out_path = os.path.join(class_dir, f"synthetic_{i:04d}.jpg")
    cv2.imwrite(out_path, degraded)

    if i % 100 == 0:
        print(f"Generated {i} images...")

print("Done generating dataset.")
