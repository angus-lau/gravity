import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if len(t) > 1]


class BM25Index:
    """Lexical search using BM25Okapi over campaign text."""

    _instance: "BM25Index | None" = None

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.campaign_ids: list[str] = []
        self._load()

    @classmethod
    def get_instance(cls) -> "BM25Index":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        campaigns_path = DATA_DIR / "campaigns_indexed.json"
        if not campaigns_path.exists():
            campaigns_path = DATA_DIR / "campaigns.json"
        if not campaigns_path.exists():
            return

        with open(campaigns_path) as f:
            campaigns = json.load(f)

        corpus = []
        for c in campaigns:
            text = " ".join([
                c.get("title", ""),
                c.get("description", ""),
                " ".join(c.get("categories", [])),
                " ".join(c.get("keywords", [])),
                c.get("advertiser", ""),
            ])
            corpus.append(_tokenize(text))
            self.campaign_ids.append(c["campaign_id"])

        if corpus:
            self.index = BM25Okapi(corpus)

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.campaign_ids) > 0

    def search(self, query: str, top_k: int = 2000) -> list[tuple[str, float]]:
        if not self.is_loaded:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.index.get_scores(tokens)

        # Get top-k indices by score
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.campaign_ids[idx], score))

        return results


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    return BM25Index.get_instance()
