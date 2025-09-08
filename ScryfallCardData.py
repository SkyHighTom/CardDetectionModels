import httpx
import time
from PIL import Image
from io import BytesIO
import gzip
import json

headers = {
    "User-Agent": "TradeTracker/1.0",
    "Accept": "application/json;q=0.9,*/*;q=0.8"
}

def get_all_cards_from_set(set_code):
    url = f"https://api.scryfall.com/cards/search?q=set:{set_code}"
    all_cards = []

    while url:
        r = httpx.get(url, headers=headers).json()
        all_cards.extend(r['data'])

        # Stop if no more pages
        if r.get('has_more'):
            url = r['next_page']
        else:
            url = None

        time.sleep(0.1)  # polite delay

    return all_cards

"""cards = get_all_cards_from_set("znc")
print(f"Fetched {len(cards)} cards")
# Loop through the results
for i in range(3):
    card = cards[i]


    if "image_uris" in card:
        image_url = card["image_uris"]["png"]
    else:
        image_url = card["card_faces"][0]["image_uris"]["png"]
    img = Image.open(BytesIO(httpx.get(image_url, headers=headers).content))
    img.show()

    print(card["name"])
    print(card["id"])
    print(image_url)
    print(card["set"])
    print(card["collector_number"])
    print(card["rarity"])
    time.sleep(0.1)  # polite delay"""
"""# Let's say you already have the JSON from the Scryfall bulk-data endpoint
url = "https://api.scryfall.com/bulk-data"
bulk_data_json = httpx.get(url, headers=headers).json()

# Step 1: Find the Oracle Cards entry
oracle_entry = bulk_data_json["data"][0]

download_url = oracle_entry['download_uri']
print("Oracle Cards download URL:", download_url)

# Step 2: Download the JSON file
oracle_cards = httpx.get(download_url).json()"""
# Open and parse JSON
with open("default-cards-20250824091023.json", "r", encoding="utf-8") as file:
    oracle_cards = json.load(file)

# Step 4: Use the data
print(type(oracle_cards))  # should be <class 'list'> or <class 'dict'>
print(f"Loaded {len(oracle_cards)} cards")
for i in range(len(oracle_cards)):
    card = oracle_cards[i]
    if card["name"] == "Ugin, Eye of the Storms":
        print(card["name"])
        print(card["id"])
        print(card["set"])
        print(card["collector_number"])
        print(card["rarity"])
        time.sleep(0.1)  # polite delay
        image_url = card["image_uris"]["png"]
        img = Image.open(BytesIO(httpx.get(image_url, headers=headers).content))
        img.show()

file.close()