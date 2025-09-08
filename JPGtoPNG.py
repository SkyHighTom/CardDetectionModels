import os
from PIL import Image

input_folder = "YugiohImages"
output_folder = "YugiohPNGImages"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGBA")  # Keep alpha support
        base, _ = os.path.splitext(filename)
        img.save(os.path.join(output_folder, base + ".png"))
