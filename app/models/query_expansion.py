import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MAX_EXPANSION_TERMS = 4


class QueryExpander:
    _instance: "QueryExpander | None" = None

    def __init__(self):
        self._synonyms: dict[str, list[str]] = {}
        self._patterns: dict[str, re.Pattern] = {}
        self._expand_cache: dict[str, str] = {}
        self._expand_cache_max_size: int = 10000
        self._load()

    @classmethod
    def get_instance(cls) -> "QueryExpander":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        synonyms_path = DATA_DIR / "synonyms.json"
        if synonyms_path.exists():
            with open(synonyms_path) as f:
                data = json.load(f)
            for _domain, mappings in data.items():
                for term, expansions in mappings.items():
                    key = term.lower()
                    if key not in self._synonyms:
                        self._synonyms[key] = []
                    for exp in expansions:
                        if exp.lower() not in self._synonyms[key]:
                            self._synonyms[key].append(exp.lower())

        for term in self._synonyms:
            self._patterns[term] = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)

        # Pre-sort by length descending (longest-first match) once at load time
        self._sorted_terms = sorted(self._synonyms.keys(), key=len, reverse=True)

    def expand(self, query: str) -> str:
        cache_key = query.lower().strip()
        if cache_key in self._expand_cache:
            return self._expand_cache[cache_key]

        q_lower = query.lower()
        added: list[str] = []

        for term in self._sorted_terms:
            if len(added) >= MAX_EXPANSION_TERMS:
                break
            if self._patterns[term].search(query):
                for synonym in self._synonyms[term]:
                    if synonym not in q_lower and synonym not in added:
                        added.append(synonym)
                        if len(added) >= MAX_EXPANSION_TERMS:
                            break

        if not added:
            result = query
        else:
            result = f"{query} {' '.join(added)}"

        self._expand_cache[cache_key] = result
        if len(self._expand_cache) > self._expand_cache_max_size:
            oldest = next(iter(self._expand_cache))
            del self._expand_cache[oldest]
        return result


@lru_cache(maxsize=1)
def get_query_expander() -> QueryExpander:
    return QueryExpander.get_instance()
