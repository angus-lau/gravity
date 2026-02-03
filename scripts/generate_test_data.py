#!/usr/bin/env python3
"""Generate synthetic campaign data for testing (no API required)."""
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ADVERTISERS = [
    "Nike", "Adidas", "Under Armour", "New Balance", "Brooks",
    "Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo",
    "Marriott", "Hilton", "Airbnb", "Expedia", "Booking.com",
    "Whole Foods", "Trader Joe's", "Starbucks", "Chipotle",
    "Chase", "Citi", "Capital One", "American Express",
    "Peloton", "Lululemon", "Fitbit", "Garmin",
    "Tesla", "Ford", "Toyota", "Honda", "BMW",
    "Netflix", "Spotify", "Disney+", "HBO Max",
    "Coursera", "Udemy", "LinkedIn Learning", "Skillshare",
    "IKEA", "Wayfair", "Home Depot", "Lowe's",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Los Angeles, CA", "Chicago, IL",
    "Seattle, WA", "Austin, TX", "Boston, MA", "Denver, CO",
    "Miami, FL", "Portland, OR", "Atlanta, GA", "Phoenix, AZ",
]

INTERESTS = [
    "fitness", "running", "technology", "travel", "cooking", "gaming",
    "fashion", "music", "photography", "outdoor activities", "reading",
    "sports", "finance", "home improvement", "wellness", "education",
]


def load_categories():
    with open(DATA_DIR / "categories.json") as f:
        data = json.load(f)
    cats = []
    for vertical in data["verticals"]:
        for cat in vertical["categories"]:
            cats.append({
                "id": cat["id"],
                "name": cat["name"],
                "keywords": cat.get("keywords", []),
                "vertical": vertical["name"],
            })
    return cats


def generate_campaign(idx: int, category: dict) -> dict:
    advertiser = random.choice(ADVERTISERS)
    keywords = category["keywords"][:5] + [category["name"].lower()]

    title_templates = [
        f"{advertiser} - {category['name']} Sale",
        f"Shop {category['name']} at {advertiser}",
        f"Best {category['name']} from {advertiser}",
        f"{advertiser} {category['name']} Collection",
        f"Discover {category['name']} by {advertiser}",
    ]

    desc_templates = [
        f"Find the best {category['name'].lower()} at {advertiser}. Quality products with fast shipping.",
        f"Shop our selection of {category['name'].lower()}. {advertiser} offers top-rated products.",
        f"{advertiser} brings you premium {category['name'].lower()}. Browse our collection today.",
        f"Looking for {category['name'].lower()}? {advertiser} has what you need at great prices.",
    ]

    return {
        "campaign_id": f"camp_{idx:05d}",
        "advertiser": advertiser,
        "title": random.choice(title_templates),
        "description": random.choice(desc_templates),
        "categories": [category["name"]],
        "keywords": list(set(keywords)),
        "targeting": {
            "locations": random.sample(LOCATIONS, random.randint(2, 5)),
            "age_range": [random.randint(18, 25), random.randint(45, 65)],
            "genders": random.choice(["all", "male", "female"]),
            "interests": random.sample(INTERESTS, random.randint(2, 4)),
        },
        "bid_amount": round(random.uniform(0.5, 15.0), 2),
    }


def main():
    categories = load_categories()
    campaigns = []

    # Generate ~100 campaigns per category (100 cats * 100 = 10,000)
    campaigns_per_cat = 100
    idx = 1

    for cat in categories:
        for _ in range(campaigns_per_cat):
            campaigns.append(generate_campaign(idx, cat))
            idx += 1

    random.shuffle(campaigns)
    for i, c in enumerate(campaigns):
        c["campaign_id"] = f"camp_{i + 1:05d}"

    with open(DATA_DIR / "campaigns.json", "w") as f:
        json.dump(campaigns, f, indent=2)

    print(f"Generated {len(campaigns)} campaigns")


if __name__ == "__main__":
    main()
