#!/usr/bin/env python3
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    campaigns_path = DATA_DIR / "campaigns.json"
    if not campaigns_path.exists():
        print(f"no campaigns.json found - run generate_campaigns.py first")
        return

    with open(campaigns_path) as f:
        campaigns = json.load(f)
    with open(DATA_DIR / "categories.json") as f:
        categories = json.load(f)

    print(f"loading model...")
    model = SentenceTransformer(MODEL_NAME)

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
    print("building index...")
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

    # quick sanity check
    query = model.encode(["running shoes for marathon"], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query, 3)
    print(f"\ntest query 'running shoes for marathon':")
    for score, idx in zip(scores[0], indices[0]):
        print(f"  [{score:.3f}] {campaigns[idx]['title'][:60]}")

    print(f"\ndone: {index.ntotal} campaigns, {len(category_ids)} categories")


if __name__ == "__main__":
    main()
