import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from app.models.embeddings import EmbeddingModel

TOXICITY_MODEL = "martin-ha/toxic-comment-model"

DANGEROUS_ANCHORS = [
    "instructions for making weapons bombs and explosives",
    "how to harm hurt or kill people violently",
    "self-harm suicide methods and ways to end life",
    "illegal drug manufacturing and distribution",
]

SAFE_ANCHORS = [
    "shopping for products and services online",
    "learning about history science and education",
    "everyday questions tasks and information",
]

# ============================================================================
# BLOCKLISTS
# ============================================================================

NSFW_TERMS = {
    "porn", "pornography", "xxx", "nsfw", "nude", "nudes", "naked",
    "sex video", "adult video", "onlyfans leak", "hentai",
}

VIOLENCE_TERMS = {
    "bomb", "pipe bomb", "explosive", "terrorist", "terrorism",
    "mass shooting", "school shooting", "kill people", "murder someone",
    "assassinate", "attack planning",
}

SELF_HARM_TERMS = {
    "suicide", "kill myself", "end my life", "self-harm", "self harm",
    "cut myself", "want to die", "how to die",
}

HATE_PATTERNS = [
    r"\bhate\b.*\b(all|every)\b.*\b(immigrants?|muslims?|jews?|blacks?|whites?|asians?|gays?|lesbians?|trans)\b",
    r"\b(kill|eliminate|exterminate)\b.*\b(immigrants?|muslims?|jews?|blacks?|whites?|asians?|gays?|lesbians?|trans)\b",
    r"\b(n[i1]gg|f[a4]gg|k[i1]ke|sp[i1]c|ch[i1]nk)\b",
]

TRAGEDY_TERMS = {
    "passed away", "just died", "funeral", "death of my", "lost my mom",
    "lost my dad", "lost my wife", "lost my husband", "lost my child",
    "cancer diagnosis", "terminal illness", "hospice",
}

_hate_patterns_compiled = [re.compile(p, re.IGNORECASE) for p in HATE_PATTERNS]

# ============================================================================
# LEETSPEAK / OBFUSCATION NORMALIZATION
# ============================================================================

# Character substitution map: common leetspeak + unicode lookalikes
_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "9": "g",
    "@": "a", "$": "s", "!": "i", "+": "t",
    # Common unicode lookalikes
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0443": "y",  # Cyrillic у
    "\u0445": "x",  # Cyrillic х
})

# Non-space separators between word chars: "b.o.m.b", "b-o-m-b", "s3lf-h4rm"
# Does NOT strip spaces (preserves "kill myself" as two words)
_PUNCTUATION_SEPARATOR = re.compile(r"(?<=\w)[.\-_*/|\\]+(?=\w)")

# Chains of single chars separated by spaces: "b o m b" → "bomb"
# Requires 3+ single chars to avoid false positives on "a b" or "to a"
_SPACED_CHARS = re.compile(r"(?<!\w)(\w(?:\s\w){2,})(?!\w)")

# Repeated characters: "boooomb" → "bomb"
_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")


def _normalize_obfuscation(text: str) -> str:
    """Normalize leetspeak, separators, and repeated chars for blocklist matching.

    Examples:
        "b0mb"        → "bomb"
        "su1c1de"     → "suicide"
        "p.o.r.n"     → "porn"
        "k!ll mys3lf" → "kill myself"  (spaces between words preserved)
        "t3rror1st"   → "terrorist"
        "naaaked"     → "naked"
        "b o m b"     → "bomb"
        "s3lf-h4rm"   → "self-harm" → "selfharm"
    """
    t = text.lower()
    # Collapse spaced single-char chains: "b o m b" → "bomb"
    t = _SPACED_CHARS.sub(lambda m: m.group().replace(" ", ""), t)
    # Strip non-space punctuation separators: "b.o.m.b" → "bomb", "self-harm" → "selfharm"
    t = _PUNCTUATION_SEPARATOR.sub("", t)
    # Apply leetspeak substitutions
    t = t.translate(_LEET_MAP)
    # Collapse repeated characters (3+ → 1): "boooomb" → "bomb"
    t = _REPEAT_PATTERN.sub(r"\1", t)
    return t


@dataclass
class SafetyResult:
    is_blocked: bool
    base_score: float  # 0.5 - danger_penalty - toxicity_penalty
    toxicity: float  # raw toxicity for downstream sentiment gating
    danger_diff: float  # danger_sim - safe_sim


class BlocklistChecker:
    """Microsecond-fast blocklist check with obfuscation normalization."""

    _instance: "BlocklistChecker | None" = None

    def __init__(self):
        # Pre-normalize all blocklist terms so "self-harm" also matches as "selfharm"
        self._all_term_sets = [NSFW_TERMS, VIOLENCE_TERMS, SELF_HARM_TERMS, TRAGEDY_TERMS]
        self._normalized_terms: list[set[str]] = []
        for term_set in self._all_term_sets:
            normalized = set()
            for term in term_set:
                normalized.add(term)
                n = _normalize_obfuscation(term)
                # Only add normalized form if it's meaningful (>1 char)
                # Prevents "xxx" → "x" matching in every word containing 'x'
                if n != term and len(n) > 1:
                    normalized.add(n)
            self._normalized_terms.append(normalized)

    @classmethod
    def get_instance(cls) -> "BlocklistChecker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_blocked(self, query: str) -> bool:
        q_lower = query.lower()
        q_normalized = _normalize_obfuscation(query)

        for term_set in self._normalized_terms:
            for term in term_set:
                if term in q_lower or term in q_normalized:
                    return True

        for pattern in _hate_patterns_compiled:
            if pattern.search(query) or pattern.search(q_normalized):
                return True

        return False


class SafetyClassifier:
    """ML-based safety scoring using toxicity model + danger embeddings."""

    _instance: "SafetyClassifier | None" = None
    _cache: OrderedDict[str, SafetyResult] = OrderedDict()
    _cache_max_size: int = 10000

    DANGER_BLOCK_THRESHOLD = 0.4
    TOXICITY_BLOCK_THRESHOLD = 0.7
    DANGER_PENALTY_WEIGHT = 0.3
    TOXICITY_PENALTY_WEIGHT = 0.4

    def __init__(self):
        self._embedding_model = EmbeddingModel.get_instance()
        # Stack anchor embeddings into matrices for vectorized cosine sim
        # Embeddings are already L2-normalized by sentence-transformers,
        # so cosine sim = dot product (skip norm computation)
        self._dangerous_mat = np.vstack([
            self._embedding_model.encode(a).flatten() for a in DANGEROUS_ANCHORS
        ])
        self._safe_mat = np.vstack([
            self._embedding_model.encode(a).flatten() for a in SAFE_ANCHORS
        ])

        from onnxruntime import SessionOptions
        session_options = SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1

        self._toxicity_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        self._toxicity_model = ORTModelForSequenceClassification.from_pretrained(
            TOXICITY_MODEL,
            export=True,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )

    @classmethod
    def get_instance(cls) -> "SafetyClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_danger_score(self, query: str, query_embedding: np.ndarray | None = None) -> tuple[float, float]:
        q_emb = query_embedding if query_embedding is not None else self._embedding_model.encode(query).flatten()
        # Vectorized dot product against all anchors at once
        # Embeddings are L2-normalized, so dot product = cosine similarity
        max_danger = float((self._dangerous_mat @ q_emb).max())
        max_safe = float((self._safe_mat @ q_emb).max())
        return max_danger, max_safe

    def _get_toxicity_score(self, query: str) -> float:
        inputs = self._toxicity_tokenizer(
            query[:512], return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._toxicity_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()

    def classify(self, query: str, query_embedding: np.ndarray | None = None) -> SafetyResult:
        """
        Classify query safety. Returns SafetyResult with base_score,
        raw toxicity, and danger_diff for downstream use.
        """
        normalized = query.lower().strip()
        if normalized in self._cache:
            return self._cache[normalized]

        danger_sim, safe_sim = self._get_danger_score(query, query_embedding)
        toxicity = self._get_toxicity_score(query)

        danger_diff = danger_sim - safe_sim

        if danger_diff > self.DANGER_BLOCK_THRESHOLD or toxicity > self.TOXICITY_BLOCK_THRESHOLD:
            result = SafetyResult(is_blocked=True, base_score=0.0, toxicity=toxicity, danger_diff=danger_diff)
        else:
            base_score = 0.5
            if danger_diff > 0:
                base_score -= danger_diff * self.DANGER_PENALTY_WEIGHT
            base_score -= toxicity * self.TOXICITY_PENALTY_WEIGHT
            result = SafetyResult(is_blocked=False, base_score=base_score, toxicity=toxicity, danger_diff=danger_diff)

        self._cache[normalized] = result
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
        return result


@lru_cache(maxsize=1)
def get_blocklist_checker() -> BlocklistChecker:
    return BlocklistChecker.get_instance()


@lru_cache(maxsize=1)
def get_safety_classifier() -> SafetyClassifier:
    return SafetyClassifier.get_instance()
