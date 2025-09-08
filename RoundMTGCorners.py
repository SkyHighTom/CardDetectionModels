import os
from PIL import Image, ImageDraw

input_folder = "MTGimages"
output_folder = "CutMTGimages"
corner_radius = 23  # exact radius for 488x680 MTG card

os.makedirs(output_folder, exist_ok=True)

def apply_card_shape(image, radius):
    # Create a blank mask the same size as the image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw the rounded rectangle in pure white (inside of card shape)
    draw.rounded_rectangle(
        (0, 0, image.size[0], image.size[1]),
        radius=radius,
        fill=255
    )

    # Ensure mask is strictly 0 or 255 (no semi-transparent edges left)
    mask = mask.point(lambda p: 255 if p > 128 else 0)

    # Force the image to RGBA and apply the mask
    image = image.convert("RGBA")
    image.putalpha(mask)
    return image

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        shaped = apply_card_shape(img, corner_radius)

        # Save as PNG to keep transparency
        base, _ = os.path.splitext(filename)
        shaped.save(os.path.join(output_folder, base + ".png"))
