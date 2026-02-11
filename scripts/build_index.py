#!/usr/bin/env python3
import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "clip-ViT-B-32"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def build_text_index(campaigns, categories, model):
    """Build FAISS index + category embeddings using text embedding model."""
    # campaign embeddings
    print(f"encoding {len(campaigns)} campaigns...")
    campaign_texts = [
        f"{c['title']} {c['description']} {' '.join(c['categories'])} {' '.join(c['keywords'])} {c['advertiser']}"
        for c in campaigns
    ]
    campaign_embeddings = model.encode(
        campaign_texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # faiss index
    print("building text index...")
    index = faiss.IndexFlatIP(campaign_embeddings.shape[1])
    index.add(campaign_embeddings)

    # category embeddings
    category_data = []
    for vertical in categories["verticals"]:
        for cat in vertical["categories"]:
            text = f"{cat['name']} {' '.join(cat.get('keywords', []))}"
            category_data.append((cat["id"], text))

    category_ids = [c[0] for c in category_data]
    category_texts = [c[1] for c in category_data]
    category_embeddings = model.encode(
        category_texts,
        normalize_embeddings=True,
    ).astype(np.float32)

    # save
    faiss.write_index(index, str(DATA_DIR / "campaigns.index"))
    np.save(DATA_DIR / "campaign_embeddings.npy", campaign_embeddings)
    np.save(DATA_DIR / "category_embeddings.npy", category_embeddings)

    with open(DATA_DIR / "campaign_id_mapping.json", "w") as f:
        json.dump([c["campaign_id"] for c in campaigns], f)
    with open(DATA_DIR / "category_ids.json", "w") as f:
        json.dump(category_ids, f)
    with open(DATA_DIR / "campaigns_indexed.json", "w") as f:
        json.dump(campaigns, f)

    return index, campaign_embeddings


def build_clip_index(campaigns):
    """Build CLIP text-proxy index for image-based retrieval."""
    print(f"\nloading CLIP model ({CLIP_MODEL_NAME})...")
    clip_model = SentenceTransformer(CLIP_MODEL_NAME)

    # Use campaign text as proxy for image content
    print(f"encoding {len(campaigns)} campaign descriptions with CLIP...")
    caption_texts = [
        f"{c['title']} {c['description']}"
        for c in campaigns
    ]
    clip_embeddings = clip_model.encode(
        caption_texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Build FAISS index for CLIP embeddings
    print("building caption index...")
    clip_index = faiss.IndexFlatIP(clip_embeddings.shape[1])
    clip_index.add(clip_embeddings)

    # Save
    faiss.write_index(clip_index, str(DATA_DIR / "caption.index"))
    np.save(DATA_DIR / "caption_embeddings.npy", clip_embeddings)

    print(f"caption index: {clip_index.ntotal} campaigns, {clip_embeddings.shape[1]}-dim")
    return clip_index, clip_embeddings


def main():
    campaigns_path = DATA_DIR / "campaigns.json"
    if not campaigns_path.exists():
        print(f"no campaigns.json found - run generate_campaigns.py first")
        return

    with open(campaigns_path) as f:
        campaigns = json.load(f)
    with open(DATA_DIR / "categories.json") as f:
        categories = json.load(f)

    print(f"loading text model...")
    model = SentenceTransformer(MODEL_NAME)

    index, campaign_embeddings = build_text_index(campaigns, categories, model)

    # Build CLIP index if enabled or explicitly requested
    build_clip = os.getenv("BUILD_CLIP_INDEX", "true").lower() in ("1", "true")
    if build_clip:
        try:
            build_clip_index(campaigns)
        except Exception as e:
            print(f"warning: CLIP index build failed ({e}), skipping")

    # quick sanity check
    query = model.encode(["running shoes for marathon"], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query, 3)
    print(f"\ntest query 'running shoes for marathon':")
    for score, idx in zip(scores[0], indices[0]):
        print(f"  [{score:.3f}] {campaigns[idx]['title'][:60]}")

    print(f"\ndone: {index.ntotal} campaigns")


if __name__ == "__main__":
    main()
