import httpx
import time
from PIL import Image
from io import BytesIO
import json
import re

url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
all_cards = httpx.get(url).json()["data"]

print(f"Loaded {len(all_cards)} cards")
for i in range(len(all_cards)):
    card = all_cards[i]
    if card["name"] == "Dark Magician":
        print(card)
        print(type(card))
        print(card["name"])
        print(card["id"])
        for i in range(len(card["card_sets"])):
            print(card["card_sets"][i]["set_name"])
            print(card["card_sets"][i]["set_rarity"])
            print(int(re.search(r'\d+$', card["card_sets"][i]["set_code"]).group()))
        time.sleep(0.1)
        for i in range(len(card["card_images"])):
            image_url = card["card_images"][i]["image_url_small"]
            img = Image.open(BytesIO(httpx.get(image_url).content))
            img.show()