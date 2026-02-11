#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-4o-mini"

CAMPAIGNS_PER_VERTICAL = 1000
BATCH_SIZE = 20
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.5

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CATEGORIES_PATH = DATA_DIR / "categories.json"
CAMPAIGNS_PATH = DATA_DIR / "campaigns.json"
CHECKPOINT_PATH = DATA_DIR / "campaigns.checkpoint.json"


class CampaignTargeting(BaseModel):
    locations: list[str] = Field(min_length=1, max_length=10)
    age_range: list[int] = Field(min_length=2, max_length=2)
    genders: str | list[str]
    interests: list[str] = Field(min_length=1, max_length=10)

    @field_validator("age_range")
    @classmethod
    def validate_age_range(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or v[0] > v[1] or v[0] < 13 or v[1] > 100:
            raise ValueError("invalid age range")
        return v

    @field_validator("genders")
    @classmethod
    def normalize_genders(cls, v: str | list[str]) -> str:
        return v[0] if isinstance(v, list) and len(v) == 1 else (v if isinstance(v, str) else "all")


class GeneratedCampaign(BaseModel):
    campaign_id: str
    advertiser: str = Field(min_length=1, max_length=150)
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=1000)
    categories: list[str] = Field(min_length=1, max_length=5)
    keywords: list[str] = Field(min_length=2, max_length=20)
    targeting: CampaignTargeting
    bid_amount: float = Field(ge=0.10, le=100.0)

    @field_validator("campaign_id")
    @classmethod
    def validate_campaign_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("campaign_id cannot be empty")
        return v.strip()


def load_checkpoint() -> tuple[list[dict], int]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
            return checkpoint.get("campaigns", []), checkpoint.get("last_vertical_index", -1)
    return [], -1


def save_checkpoint(campaigns: list[dict], vertical_index: int) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({
            "campaigns": campaigns,
            "last_vertical_index": vertical_index,
            "timestamp": datetime.utcnow().isoformat(),
        }, f)


def build_prompt(vertical: dict, batch_num: int, batch_size: int) -> str:
    category_names = [c["name"] for c in vertical["categories"]]
    vertical_abbrev = vertical["id"][:4]
    start_num = batch_num * batch_size + 1

    return f"""Generate exactly {batch_size} realistic advertising campaigns for the "{vertical['name']}" vertical.

Use these categories (pick 1-3 per campaign): {', '.join(category_names)}

Each campaign MUST follow this EXACT JSON structure:
{{
  "campaign_id": "camp_{vertical_abbrev}_{{number}}",
  "advertiser": "Brand Name Here",
  "title": "Compelling headline (5-12 words)",
  "description": "Campaign description with value proposition (20-60 words)",
  "categories": ["category1", "category2"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "targeting": {{
    "locations": ["City, State"],
    "age_range": [min_age, max_age],
    "genders": "all" or "male" or "female",
    "interests": ["interest1", "interest2", "interest3"]
  }},
  "bid_amount": 2.50
}}

Requirements:
- campaign_id: Use format "camp_{vertical_abbrev}_{{number}}" where number starts at {start_num}
- advertiser: Mix of realistic brand names (both well-known style and fictional)
- title: Compelling, action-oriented headlines
- description: Clear value proposition, 20-60 words
- categories: 1-3 from the provided list
- keywords: 5-10 relevant search terms
- targeting.locations: 1-5 US cities (format: "City, State")
- targeting.age_range: [min, max] between 18-65
- targeting.genders: "all", "male", or "female"
- targeting.interests: 2-5 relevant interest tags
- bid_amount: Realistic CPM bid between $0.50-$15.00

IMPORTANT: Return ONLY a valid JSON array with exactly {batch_size} campaign objects. No markdown, no explanation, just the JSON array."""


def extract_json_from_response(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start_idx = 1
        end_idx = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "```":
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "campaigns" in data:
            return data["campaigns"]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    raise ValueError("Could not parse JSON array from response")


def validate_campaigns(campaigns: list[dict], vertical_id: str) -> list[dict]:
    validated = []
    for i, campaign in enumerate(campaigns):
        try:
            if "campaign_id" not in campaign or not campaign["campaign_id"]:
                campaign["campaign_id"] = f"camp_{vertical_id[:4]}_{len(validated) + 1}"
            validated.append(GeneratedCampaign(**campaign).model_dump())
        except ValidationError as e:
            print(f"  skip campaign {i}: {e.errors()[0]['msg']}")
    return validated


async def generate_batch(client: httpx.AsyncClient, prompt: str, vertical_id: str) -> list[dict]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 4096,
                },
                timeout=120.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            campaigns = extract_json_from_response(content)
            validated = validate_campaigns(campaigns, vertical_id)
            if validated:
                return validated
        except (httpx.HTTPStatusError, json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = 5 if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429 else 2 ** attempt
                await asyncio.sleep(wait)
            else:
                print(f"  batch failed after {MAX_RETRIES} attempts: {e}")
                return []
    return []


async def main() -> None:
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set")
        sys.exit(1)

    with open(CATEGORIES_PATH) as f:
        categories = json.load(f)
    verticals = categories["verticals"]

    campaigns, last_idx = load_checkpoint()
    start_idx = last_idx + 1
    if campaigns:
        print(f"resuming: {len(campaigns)} campaigns from vertical {start_idx}")

    async with httpx.AsyncClient() as client:
        for v_idx in range(start_idx, len(verticals)):
            vertical = verticals[v_idx]
            print(f"[{v_idx + 1}/{len(verticals)}] {vertical['name']}")

            num_batches = (CAMPAIGNS_PER_VERTICAL + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_num in range(num_batches):
                prompt = build_prompt(vertical, batch_num, BATCH_SIZE)
                batch = await generate_batch(client, prompt, vertical["id"])
                campaigns.extend(batch)
                print(f"  batch {batch_num + 1}/{num_batches}: +{len(batch)}")
                await asyncio.sleep(RATE_LIMIT_DELAY)

            save_checkpoint(campaigns, v_idx)

    # dedupe and reassign IDs
    seen = set()
    unique = [c for c in campaigns if c["campaign_id"] not in seen and not seen.add(c["campaign_id"])]
    for i, c in enumerate(unique):
        c["campaign_id"] = f"camp_{i + 1:05d}"

    with open(CAMPAIGNS_PATH, "w") as f:
        json.dump(unique, f, indent=2)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"done: {len(unique)} campaigns")


if __name__ == "__main__":
    asyncio.run(main())
