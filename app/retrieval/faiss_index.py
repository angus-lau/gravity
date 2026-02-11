import json
from pathlib import Path

import faiss
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class CampaignIndex:
    _instance: "CampaignIndex | None" = None

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.campaign_ids: list[str] = []
        self.campaigns: dict[str, dict] = {}
        self._load()

    @classmethod
    def get_instance(cls) -> "CampaignIndex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        index_path = DATA_DIR / "campaigns.index"
        mapping_path = DATA_DIR / "campaign_id_mapping.json"
        campaigns_path = DATA_DIR / "campaigns_indexed.json"

        if not index_path.exists():
            campaigns_path = DATA_DIR / "campaigns.json"

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

        if mapping_path.exists():
            with open(mapping_path) as f:
                self.campaign_ids = json.load(f)

        if campaigns_path.exists():
            with open(campaigns_path) as f:
                campaigns_list = json.load(f)
                self.campaigns = {c["campaign_id"]: c for c in campaigns_list}
            # Pre-compute targeting fields for faster reranking
            self._precompute_targeting()

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.campaign_ids) > 0

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def search(self, query_embedding: np.ndarray, top_k: int = 2000) -> list[tuple[str, float]]:
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

    def _precompute_targeting(self):
        """Pre-normalize campaign targeting fields at load time for faster reranking."""
        from app.retrieval.ranker import _normalize_location
        for campaign in self.campaigns.values():
            targeting = campaign.get("targeting", {})
            locs = targeting.get("locations", [])
            if locs:
                targeting["_locations_normalized"] = [_normalize_location(loc) for loc in locs]
            interests = targeting.get("interests", [])
            if interests:
                targeting["_interests_lowered"] = frozenset(i.lower() for i in interests)
            genders = targeting.get("genders")
            if isinstance(genders, list):
                targeting["_genders_lowered"] = [g.lower() for g in genders]
            elif isinstance(genders, str):
                targeting["_genders_lowered_str"] = genders.lower()

    def get_campaign(self, campaign_id: str) -> dict | None:
        return self.campaigns.get(campaign_id)


def get_campaign_index() -> CampaignIndex:
    return CampaignIndex.get_instance()
