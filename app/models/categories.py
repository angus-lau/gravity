import json
from pathlib import Path

import numpy as np

from app.models.embeddings import get_embedding_model

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MIN_SIMILARITY = 0.25
MAX_CATEGORIES = 10


class CategoryMatcher:
    _instance: "CategoryMatcher | None" = None

    def __init__(self):
        self.category_ids: list[str] = []
        self.category_names: dict[str, str] = {}
        self.category_embeddings: np.ndarray | None = None
        self._load()

    @classmethod
    def get_instance(cls) -> "CategoryMatcher":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        categories_path = DATA_DIR / "categories.json"
        embeddings_path = DATA_DIR / "category_embeddings.npy"
        ids_path = DATA_DIR / "category_ids.json"

        if not categories_path.exists():
            return

        with open(categories_path) as f:
            data = json.load(f)

        for vertical in data["verticals"]:
            for cat in vertical["categories"]:
                self.category_ids.append(cat["id"])
                self.category_names[cat["id"]] = cat["name"]

        if embeddings_path.exists() and ids_path.exists():
            self.category_embeddings = np.load(embeddings_path)
            with open(ids_path) as f:
                self.category_ids = json.load(f)
        else:
            self._build_embeddings(data)

    def _build_embeddings(self, data: dict):
        model = get_embedding_model()
        texts = []
        self.category_ids = []

        for vertical in data["verticals"]:
            for cat in vertical["categories"]:
                text = f"{cat['name']} {' '.join(cat.get('keywords', []))}"
                texts.append(text)
                self.category_ids.append(cat["id"])
                self.category_names[cat["id"]] = cat["name"]

        if texts:
            self.category_embeddings = model.encode(texts, normalize=True)

    def match(self, query_embedding: np.ndarray, top_k: int = 5) -> list[str]:
        if self.category_embeddings is None or len(self.category_ids) == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = np.dot(query_embedding, self.category_embeddings.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:MAX_CATEGORIES]

        results = []
        for idx in top_indices:
            if similarities[idx] >= MIN_SIMILARITY and len(results) < top_k:
                cat_id = self.category_ids[idx]
                results.append(self.category_names.get(cat_id, cat_id))

        return results if results else [self.category_names.get(self.category_ids[top_indices[0]], "general")]


def get_category_matcher() -> CategoryMatcher:
    return CategoryMatcher.get_instance()
