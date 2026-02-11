from collections import OrderedDict
from functools import lru_cache

from app.models.commercial import get_commercial_classifier
from app.models.safety import get_blocklist_checker, get_safety_classifier


class EligibilityClassifier:
    _instance: "EligibilityClassifier | None" = None
    _cache: OrderedDict[str, float] = OrderedDict()
    _cache_max_size: int = 10000

    def __init__(self):
        self._blocklist = get_blocklist_checker()
        self._safety = get_safety_classifier()
        self._commercial = get_commercial_classifier()

    @classmethod
    def get_instance(cls) -> "EligibilityClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def score(self, query: str) -> float:
        normalized = query.lower().strip()
        if normalized in self._cache:
            return self._cache[normalized]

        if self._blocklist.is_blocked(query):
            self._cache_put(normalized, 0.0)
            return 0.0

        safety = self._safety.classify(query)
        if safety.is_blocked:
            self._cache_put(normalized, 0.0)
            return 0.0

        score = self._commercial.score(query, safety)
        self._cache_put(normalized, score)
        return score

    def _cache_put(self, key: str, value: float) -> None:
        self._cache[key] = value
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)


@lru_cache(maxsize=1)
def get_eligibility_classifier() -> EligibilityClassifier:
    return EligibilityClassifier.get_instance()
