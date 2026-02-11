import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MAX_EXPANSION_TERMS = 4
MAX_KEYWORD_TERMS = 5
# Queries shorter than this (in words) get KeyBERT keyword extraction
VAGUE_QUERY_THRESHOLD = 4


class QueryExpander:
    """Lightweight synonym expansion to improve recall before embedding."""

    _instance: "QueryExpander | None" = None

    def __init__(self):
        self._synonyms: dict[str, list[str]] = {}
        self._patterns: dict[str, re.Pattern] = {}
        self._keyword_model = None
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

    def _get_keyword_model(self):
        """Lazy-load KeyBERT using the existing embedding model."""
        if self._keyword_model is None:
            from keybert import KeyBERT
            from app.models.embeddings import get_embedding_model
            self._keyword_model = KeyBERT(model=get_embedding_model().model)
        return self._keyword_model

    def extract_keywords(self, query: str, top_n: int = MAX_KEYWORD_TERMS) -> list[str]:
        """Extract keywords from a vague query using KeyBERT.

        Uses the existing all-MiniLM-L6-v2 model — no extra downloads.
        Returns the most relevant keywords/keyphrases.
        """
        kw_model = self._get_keyword_model()
        keywords = kw_model.extract_keywords(
            query,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,       # Maximal Marginal Relevance for diversity
            diversity=0.5,
        )
        return [kw for kw, _score in keywords]

    def expand(self, query: str) -> str:
        """Expand query with synonyms and keyword extraction.

        1. Rule-based synonym expansion for known terms
        2. KeyBERT keyword extraction for vague/short queries
        """
        q_lower = query.lower()
        added: list[str] = []

        # Step 1: Rule-based synonym expansion
        sorted_terms = sorted(self._synonyms.keys(), key=len, reverse=True)
        for term in sorted_terms:
            if len(added) >= MAX_EXPANSION_TERMS:
                break
            if self._patterns[term].search(query):
                for synonym in self._synonyms[term]:
                    if synonym not in q_lower and synonym not in added:
                        added.append(synonym)
                        if len(added) >= MAX_EXPANSION_TERMS:
                            break

        # Step 2: KeyBERT keyword extraction for vague queries
        # Only if synonym expansion didn't find much and query is short/vague
        word_count = len(query.split())
        if len(added) < 2 and word_count <= VAGUE_QUERY_THRESHOLD:
            try:
                keywords = self.extract_keywords(query, top_n=MAX_KEYWORD_TERMS)
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in q_lower and kw_lower not in added:
                        added.append(kw_lower)
                        if len(added) >= MAX_EXPANSION_TERMS + MAX_KEYWORD_TERMS:
                            break
            except Exception:
                pass  # Graceful degradation — synonym expansion still works

        if not added:
            return query

        return f"{query} {' '.join(added)}"


@lru_cache(maxsize=1)
def get_query_expander() -> QueryExpander:
    return QueryExpander.get_instance()
