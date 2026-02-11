"""FAISS index over CLIP text-proxy embeddings for image-based retrieval."""

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class CaptionIndex:
    """FAISS index over CLIP text-proxy embeddings for campaign images."""

    _instance: "CaptionIndex | None" = None

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.campaign_ids: list[str] = []
        self._load()

    @classmethod
    def get_instance(cls) -> "CaptionIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        index_path = DATA_DIR / "caption.index"
        mapping_path = DATA_DIR / "campaign_id_mapping.json"

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

        if mapping_path.exists():
            with open(mapping_path) as f:
                self.campaign_ids = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.campaign_ids) > 0

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def search(self, query_embedding: np.ndarray, top_k: int = 500) -> list[tuple[str, float]]:
        if not self.is_loaded:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.campaign_ids):
                results.append((self.campaign_ids[idx], float(score)))

        return results


@lru_cache(maxsize=1)
def get_caption_index() -> CaptionIndex:
    return CaptionIndex.get_instance()
