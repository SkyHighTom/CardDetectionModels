import os
import cv2
import json
import random
import numpy as np
from PIL import Image

# Input directories
BACKGROUND_DIR = "./BackgroundImages"
CARD_DIR = "./CutMTGimages"
EFFECTS_DIR = "./Effects"
NUM_IMAGES = 75000

# Output directories
BASE_DIR = "./MTGModel"
OUTPUT_DIRS = {
    "train": {
        "images": os.path.join(BASE_DIR, "images/train"),
        "labels": os.path.join(BASE_DIR, "labels/train"),
        "masks": os.path.join(BASE_DIR, "masks/train")
    },
    "val": {
        "images": os.path.join(BASE_DIR, "images/val"),
        "labels": os.path.join(BASE_DIR, "labels/val"),
        "masks": os.path.join(BASE_DIR, "masks/val")
    }
}

# Create directories
for split in OUTPUT_DIRS.values():
    for path in split.values():
        os.makedirs(path, exist_ok=True)

def load_images_from_folder(folder, valid_exts=('.png', '.jpg', '.jpeg')):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]

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

def random_transform(card_img, num_cards, output_size=(480, 640)):
    h, w = card_img.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Apply random perspective skew
    max_skew = 20
    dst_pts = src_pts + np.random.uniform(-max_skew, max_skew, src_pts.shape).astype(np.float32)

    # ---- RANDOM SCALING ----
    if num_cards == 1:
        scale = np.random.uniform(0.6, 1.1)
    else:
        scale = np.random.uniform(1/(num_cards*2)+.1, 2/(num_cards*2)+.1)

    # Scale relative to the center of the transformed card
    center = dst_pts.mean(axis=0)
    dst_pts = (dst_pts - center) * scale + center

    # Compute width/height of the transformed card
    card_width = dst_pts[:, 0].max() - dst_pts[:, 0].min()
    card_height = dst_pts[:, 1].max() - dst_pts[:, 1].min()

    # ---- RANDOM ROTATION (in-plane) ----
    angle_deg = random.uniform(0, 360)
    angle_rad = np.deg2rad(angle_deg)

    # Rotation matrix around the center of the transformed card
    center_x = dst_pts[:, 0].mean()
    center_y = dst_pts[:, 1].mean()
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ], dtype=np.float32)

    dst_pts_centered = dst_pts - [center_x, center_y]
    dst_pts_rotated = np.dot(dst_pts_centered, rotation_matrix.T) + [center_x, center_y]

    dst_pts = dst_pts_rotated

    # ---- RANDOM TRANSLATION ----
    # More uniform randomness across the output image
    max_dx = max(0, output_size[0] - card_width)
    max_dy = max(0, output_size[1] - card_height)

    dx = np.random.uniform(0, max_dx)
    dy = np.random.uniform(0, max_dy)
    translation = np.array([[dx - dst_pts[:, 0].min(), dy - dst_pts[:, 1].min()]], dtype=np.float32)

    dst_pts += translation

    # Final perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return dst_pts.tolist(), M


def apply_transform(card_img, M, size=(480, 640)):
    return cv2.warpPerspective(card_img, M, size, borderValue=(0, 0, 0, 0))

def paste_card(bg, card_rgb, mask, M):
    warped_card = cv2.warpPerspective(card_rgb, M, (bg.shape[1], bg.shape[0]), borderValue=0)
    warped_mask = cv2.warpPerspective(mask, M, (bg.shape[1], bg.shape[0]), borderValue=0)

    inv_mask = cv2.bitwise_not(warped_mask)
    bg_part = cv2.bitwise_and(bg, bg, mask=inv_mask)
    fg_part = cv2.bitwise_and(warped_card, warped_card, mask=warped_mask)
    return cv2.add(bg_part, fg_part), warped_mask

# Prepare data
backgrounds = load_images_from_folder(BACKGROUND_DIR)
cards = load_images_from_folder(CARD_DIR)

# Shuffle indices for randomization
indices = list(range(1000))
random.shuffle(indices)

# Split indices for training and validation (80/20 split)
train_idx = indices[:int(0.8 * len(indices))]
val_idx = indices[int(0.8 * len(indices)):]

for i in range(int(NUM_IMAGES * .8)):
    bg_path = random.choice(backgrounds)
    bg = cv2.imread(bg_path)
    if bg is None:
        continue
    bg = cv2.resize(bg, (480, 640))
    canvas = bg.copy()
    mask_canvas = np.zeros((640, 480), dtype=np.uint8)
    yolo_lines = []
    existing_mask = np.zeros((640, 480), dtype=np.uint8)
    num_cards = int(random.uniform(1, 6))
    for x in range(num_cards):
        card_path = random.choice(cards)
        # Retry loop for invalid card images
        while True:
            card = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
            card = degrade_image_randomly(card)
            if card is None or card.shape[2] != 4:
                card_path = random.choice(cards)
                continue
            break

        card_rgb = card[:, :, :3]
        card_mask = (card[:, :, 3] > 0).astype(np.uint8) * 255
        num_tries = 0
        while(True):
            num_tries += 1
            if num_tries >= 20:
                if x == 0:
                    card_path = random.choice(cards)
                    card = cv2.imread(card_path)  # RGBA

                    # Optional: apply degradation *after* resizing so it looks realistic
                    card = degrade_image_randomly(card)

                    # Separate channels
                    card_rgb = card[:, :, :3]
                    h, w = card_rgb.shape[:2]

                    if card.shape[2] == 4:
                        card_mask = (card[:, :, 3] > 0).astype(np.uint8) * 255
                    else:
                        card_mask = np.ones((h, w), dtype=np.uint8) * 255
                    num_tries = 0
                else:
                    break
            corners, M = random_transform(card, num_cards, output_size=(480, 640))
            corners_np = np.array(corners, dtype=np.float32)
            # Step 1: Warp the card's mask into the output canvas space
            _, warped_card_mask = paste_card(np.zeros_like(canvas), card_rgb, card_mask, M)

            # Step 2: Check if it fits in bounds
            if not (0 <= corners_np[:, 0]).all() or not (corners_np[:, 0] < 480).all():
                continue
            if not (0 <= corners_np[:, 1]).all() or not (corners_np[:, 1] < 640).all():
                continue

            # Step 3: Check for overlap (up to 20% of the card area is allowed)
            overlap_mask = cv2.bitwise_and(existing_mask, warped_card_mask)
            overlap_area = np.count_nonzero(overlap_mask)
            card_area = np.count_nonzero(warped_card_mask)

            allowed_overlap = random.uniform(0.0, 0.20)  # e.g., up to 20% overlap randomly per card
            if card_area == 0 or (overlap_area / card_area) > allowed_overlap:
                continue

            # Step 4: Safe to paste
            break
        if num_tries >= 20:
            continue

        canvas, warped_card_mask = paste_card(canvas, card_rgb, card_mask, M)
        existing_mask = cv2.bitwise_or(existing_mask, warped_card_mask)
        # 25% chance: Add a slightly larger sleeve using scaled corners
        if random.random() < 0.25:
            sleeve_color = tuple(np.random.randint(0, 255, size=3).tolist())
            sleeve_shape = card_rgb.shape[:2]  # (height, width)

            sleeve_img = np.full((sleeve_shape[0], sleeve_shape[1], 3), sleeve_color, dtype=np.uint8)
            sleeve_mask = np.full((sleeve_shape[0], sleeve_shape[1]), 255, dtype=np.uint8)

            # Scale corners outward from the center
            corners_np = np.array(corners, dtype=np.float32)
            center = np.mean(corners_np, axis=0)
            enlarged_corners = (corners_np - center) * 1.05 + center

            src_pts = np.array([
                [0, 0],
                [sleeve_shape[1], 0],
                [sleeve_shape[1], sleeve_shape[0]],
                [0, sleeve_shape[0]]
            ], dtype=np.float32)

            sleeve_M = cv2.getPerspectiveTransform(src_pts, enlarged_corners)

            canvas, _ = paste_card(canvas, sleeve_img, sleeve_mask, sleeve_M)
        canvas, single_mask = paste_card(canvas, card_rgb, card_mask, M)
        mask_canvas = cv2.bitwise_or(mask_canvas, single_mask)


        # Add glare
        """if random.random() < 0.25:
            # Load glare image
            glare_img = Image.open(os.path.join(EFFECTS_DIR, "Glare/glare1.jpg")).convert("RGBA")

            # Compute bounding box of the card
            card_corners_np = np.array(corners, dtype=np.float32)
            min_x = min(card_corners_np[:, 0])
            max_x = max(card_corners_np[:, 0])
            min_y = min(card_corners_np[:, 1])
            max_y = max(card_corners_np[:, 1])
            
            card_w = max_x - min_x
            card_h = max_y - min_y

            # Choose random scale
            scale = random.uniform(0.5, 0.9)
            target_w = int(card_w * scale)
            target_h = int(card_h * scale)

            # Resize glare
            glare_resized = glare_img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

            # Random rotation
            angle = random.uniform(0, 360)
            glare_rotated = glare_resized.rotate(angle, expand=True)

            # Convert to OpenCV
            glare_cv = cv2.cvtColor(np.array(glare_rotated), cv2.COLOR_RGBA2BGRA)

            # Pick a random position inside card bounding box
            top_left_x = int(min_x + (card_w - glare_cv.shape[1]) * random.uniform(0, 1))
            top_left_y = int(min_y + (card_h - glare_cv.shape[0]) * random.uniform(0, 1))

            # Create mask and paste using alpha blending
            overlay = canvas.copy()
            x1, y1 = top_left_x, top_left_y
            x2, y2 = x1 + glare_cv.shape[1], y1 + glare_cv.shape[0]

            # Ensure dimensions match
            if x2 > overlay.shape[1] or y2 > overlay.shape[0] or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                continue  # skip if glare would go out of bounds

            roi = overlay[y1:y2, x1:x2]
            alpha = glare_cv[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + glare_cv[:, :, c] * alpha

            overlay[y1:y2, x1:x2] = roi
            canvas = overlay"""

        # Normalize for YOLO: x1, y1, x2, y2, ...
        height, width = 640, 480
        corners_np = np.array(corners, dtype=np.float32)

        # Flatten and normalize
        norm_corners = []
        for a, (b, c) in enumerate(corners_np):
            x_norm = round(b / width, 6)
            y_norm = round(c / height, 6)
            norm_corners.extend([x_norm, y_norm])

        # Format as YOLO segmentation line
        yolo_line = f"0 {' '.join(map(str, norm_corners))}\n"
        yolo_lines.append(yolo_line)

    # Save files
    base_name = f"synthetic_{i:04d}"
    cv2.imwrite(os.path.join(OUTPUT_DIRS["train"]["images"], f"{base_name}.jpg"), canvas)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["train"]["masks"], f"{base_name}.png"), mask_canvas)
    with open(os.path.join(OUTPUT_DIRS["train"]["labels"], f"{base_name}.txt"), "w") as f:
        f.writelines(yolo_lines)

    if i % 100 == 0:
        print(f"Generated {i} samples...")

# Generate validation images
for i in range(int(NUM_IMAGES * .8), NUM_IMAGES):
    bg_path = random.choice(backgrounds)
    bg = cv2.imread(bg_path)
    if bg is None:
        continue
    bg = cv2.resize(bg, (480, 640))
    canvas = bg.copy()
    mask_canvas = np.zeros((640, 480), dtype=np.uint8)
    yolo_lines = []
    existing_mask = np.zeros((640, 480), dtype=np.uint8)
    num_cards = int(random.uniform(1, 6))
    for x in range(num_cards):
        card_path = random.choice(cards)
        # Retry loop for invalid card images
        while True:
            card = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
            card = degrade_image_randomly(card)
            if card is None or card.shape[2] != 4:
                card_path = random.choice(cards)
                continue
            break

        card_rgb = card[:, :, :3]
        card_mask = (card[:, :, 3] > 0).astype(np.uint8) * 255
        num_tries = 0
        while(True):
            num_tries += 1
            if num_tries >= 20:
                if x == 0:
                    card_path = random.choice(cards)
                    card = cv2.imread(card_path)  # RGBA

                    # Optional: apply degradation *after* resizing so it looks realistic
                    card = degrade_image_randomly(card)

                    # Separate channels
                    card_rgb = card[:, :, :3]
                    h, w = card_rgb.shape[:2]

                    if card.shape[2] == 4:
                        card_mask = (card[:, :, 3] > 0).astype(np.uint8) * 255
                    else:
                        card_mask = np.ones((h, w), dtype=np.uint8) * 255
                    num_tries = 0
                else:
                    break
            corners, M = random_transform(card, num_cards, output_size=(480, 640))
            corners_np = np.array(corners, dtype=np.float32)
            # Step 1: Warp the card's mask into the output canvas space
            _, warped_card_mask = paste_card(np.zeros_like(canvas), card_rgb, card_mask, M)

            # Step 2: Check if it fits in bounds
            if not (0 <= corners_np[:, 0]).all() or not (corners_np[:, 0] < 480).all():
                continue
            if not (0 <= corners_np[:, 1]).all() or not (corners_np[:, 1] < 640).all():
                continue

            # Step 3: Check for overlap (up to 20% of the card area is allowed)
            overlap_mask = cv2.bitwise_and(existing_mask, warped_card_mask)
            overlap_area = np.count_nonzero(overlap_mask)
            card_area = np.count_nonzero(warped_card_mask)

            allowed_overlap = random.uniform(0.0, 0.20)  # e.g., up to 20% overlap randomly per card
            if card_area == 0 or (overlap_area / card_area) > allowed_overlap:
                continue

            # Step 4: Safe to paste
            break
        if num_tries >= 20:
            continue
        
        canvas, warped_card_mask = paste_card(canvas, card_rgb, card_mask, M)
        existing_mask = cv2.bitwise_or(existing_mask, warped_card_mask)
        # 25% chance: Add a slightly larger sleeve using scaled corners
        if random.random() < 0.25:
            sleeve_color = tuple(np.random.randint(0, 255, size=3).tolist())
            sleeve_shape = card_rgb.shape[:2]  # (height, width)

            sleeve_img = np.full((sleeve_shape[0], sleeve_shape[1], 3), sleeve_color, dtype=np.uint8)
            sleeve_mask = np.full((sleeve_shape[0], sleeve_shape[1]), 255, dtype=np.uint8)

            # Scale corners outward from the center
            corners_np = np.array(corners, dtype=np.float32)
            center = np.mean(corners_np, axis=0)
            enlarged_corners = (corners_np - center) * 1.05 + center

            src_pts = np.array([
                [0, 0],
                [sleeve_shape[1], 0],
                [sleeve_shape[1], sleeve_shape[0]],
                [0, sleeve_shape[0]]
            ], dtype=np.float32)

            sleeve_M = cv2.getPerspectiveTransform(src_pts, enlarged_corners)

            canvas, _ = paste_card(canvas, sleeve_img, sleeve_mask, sleeve_M)
        canvas, single_mask = paste_card(canvas, card_rgb, card_mask, M)
        mask_canvas = cv2.bitwise_or(mask_canvas, single_mask)


        # Add glare
        """if random.random() < 0.25:
            # Load glare image
            glare_img = Image.open(os.path.join(EFFECTS_DIR, "Glare/glare1.jpg")).convert("RGBA")

            # Compute bounding box of the card
            card_corners_np = np.array(corners, dtype=np.float32)
            min_x = min(card_corners_np[:, 0])
            max_x = max(card_corners_np[:, 0])
            min_y = min(card_corners_np[:, 1])
            max_y = max(card_corners_np[:, 1])
            
            card_w = max_x - min_x
            card_h = max_y - min_y

            # Choose random scale
            scale = random.uniform(0.5, 0.9)
            target_w = int(card_w * scale)
            target_h = int(card_h * scale)

            # Resize glare
            glare_resized = glare_img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

            # Random rotation
            angle = random.uniform(0, 360)
            glare_rotated = glare_resized.rotate(angle, expand=True)

            # Convert to OpenCV
            glare_cv = cv2.cvtColor(np.array(glare_rotated), cv2.COLOR_RGBA2BGRA)

            # Pick a random position inside card bounding box
            top_left_x = int(min_x + (card_w - glare_cv.shape[1]) * random.uniform(0, 1))
            top_left_y = int(min_y + (card_h - glare_cv.shape[0]) * random.uniform(0, 1))

            # Create mask and paste using alpha blending
            overlay = canvas.copy()
            x1, y1 = top_left_x, top_left_y
            x2, y2 = x1 + glare_cv.shape[1], y1 + glare_cv.shape[0]

            # Ensure dimensions match
            if x2 > overlay.shape[1] or y2 > overlay.shape[0] or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                continue  # skip if glare would go out of bounds

            roi = overlay[y1:y2, x1:x2]
            alpha = glare_cv[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + glare_cv[:, :, c] * alpha

            overlay[y1:y2, x1:x2] = roi
            canvas = overlay"""

        # Normalize for YOLO: x1, y1, x2, y2, ...
        height, width = 640, 480
        corners_np = np.array(corners, dtype=np.float32)

        # Flatten and normalize
        norm_corners = []
        for a, (b, c) in enumerate(corners_np):
            x_norm = round(b / width, 6)
            y_norm = round(c / height, 6)
            norm_corners.extend([x_norm, y_norm])

        # Format as YOLO segmentation line
        yolo_line = f"0 {' '.join(map(str, norm_corners))}\n"
        yolo_lines.append(yolo_line)

    # Save files for validation data
    base_name = f"synthetic_{i:04d}"
    cv2.imwrite(os.path.join(OUTPUT_DIRS["val"]["images"], f"{base_name}.jpg"), canvas)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["val"]["masks"], f"{base_name}.png"), mask_canvas)
    with open(os.path.join(OUTPUT_DIRS["val"]["labels"], f"{base_name}.txt"), "w") as f:
        f.writelines(yolo_lines)

    if i % 100 == 0:
        print(f"[VALIDATION] Generated {i-int(NUM_IMAGES*.8)} samples...")

print("Dataset generation complete!")
