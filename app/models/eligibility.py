import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from app.models.embeddings import EmbeddingModel

# Models for ML-based eligibility scoring
TOXICITY_MODEL = "martin-ha/toxic-comment-model"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Semantic anchors for dangerous content detection via embeddings
# These represent categories - the embedding model generalizes to similar content
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
# BLOCKLISTS - Hard blocks for inappropriate content (return 0.0)
# ============================================================================

# NSFW/Adult content
NSFW_TERMS = {
    "porn", "pornography", "xxx", "nsfw", "nude", "nudes", "naked",
    "sex video", "adult video", "onlyfans leak", "hentai",
}

# Violence and weapons
VIOLENCE_TERMS = {
    "bomb", "pipe bomb", "explosive", "terrorist", "terrorism",
    "mass shooting", "school shooting", "kill people", "murder someone",
    "assassinate", "attack planning",
}

# Self-harm (supplement the embedding-based detection)
SELF_HARM_TERMS = {
    "suicide", "kill myself", "end my life", "self-harm", "self harm",
    "cut myself", "want to die", "how to die",
}

# Hate speech patterns
HATE_PATTERNS = [
    r"\bhate\b.*\b(all|every)\b.*\b(immigrants?|muslims?|jews?|blacks?|whites?|asians?|gays?|lesbians?|trans)\b",
    r"\b(kill|eliminate|exterminate)\b.*\b(immigrants?|muslims?|jews?|blacks?|whites?|asians?|gays?|lesbians?|trans)\b",
    r"\b(n[i1]gg|f[a4]gg|k[i1]ke|sp[i1]c|ch[i1]nk)\b",  # Slurs (partially obfuscated)
]

# Tragedy/grief - inappropriate for ads
TRAGEDY_TERMS = {
    "passed away", "just died", "funeral", "death of my", "lost my mom",
    "lost my dad", "lost my wife", "lost my husband", "lost my child",
    "cancer diagnosis", "terminal illness", "hospice",
}

# ============================================================================
# COMMERCIAL INTENT SIGNALS - Boost scores for purchase-ready queries
# ============================================================================

COMMERCIAL_SIGNALS = {
    # Strong purchase intent
    "buy", "purchase", "order", "shop", "deal", "deals", "discount",
    "coupon", "sale", "cheap", "affordable", "price", "cost",
    # Product research
    "best", "top", "top rated", "review", "reviews", "compare", "vs",
    "recommend", "recommendation", "recommended",
    # Specific shopping
    "where to buy", "where can i buy", "how much", "for sale",
}

# ============================================================================
# SENSITIVE TOPICS - Reduce scores but don't block
# ============================================================================

SENSITIVE_TERMS = {
    # Mental health
    "depression", "anxiety", "stressed", "mental health", "therapy",
    "counseling", "psychiatrist",
    # Financial hardship
    "unemployment", "bankrupt", "bankruptcy", "debt", "foreclosure",
    "eviction", "homeless",
    # Relationship issues
    "divorce", "separation", "breakup", "cheating", "affair",
    # Health concerns
    "disease", "illness", "symptoms", "diagnosis",
}


class EligibilityClassifier:
    _instance: "EligibilityClassifier | None" = None
    _cache: dict[str, float] = {}
    _cache_max_size: int = 10000

    # Thresholds for scoring
    DANGER_BLOCK_THRESHOLD = 0.4  # Block if danger_diff exceeds this
    TOXICITY_BLOCK_THRESHOLD = 0.7  # Block if toxicity exceeds this (lowered from 0.8)
    TOXICITY_HIGH_THRESHOLD = 0.4  # High toxicity threshold for strong penalty
    DANGER_PENALTY_WEIGHT = 0.3  # How much danger similarity reduces score
    TOXICITY_PENALTY_WEIGHT = 0.4  # How much toxicity reduces score
    SENTIMENT_BOOST_WEIGHT = 0.2  # How much positive sentiment boosts score
    COMMERCIAL_BOOST = 0.45  # Boost for commercial intent queries
    SENSITIVE_PENALTY = 0.15  # Penalty for sensitive topics

    # Compile hate speech patterns once
    _hate_patterns = [re.compile(p, re.IGNORECASE) for p in HATE_PATTERNS]

    def __init__(self):
        # Load embedding model for danger detection
        self._embedding_model = EmbeddingModel.get_instance()
        self._dangerous_embs = [
            self._embedding_model.encode(a).flatten() for a in DANGEROUS_ANCHORS
        ]
        self._safe_embs = [
            self._embedding_model.encode(a).flatten() for a in SAFE_ANCHORS
        ]

        # Load toxicity model (for hate speech, threats)
        # Limit threads per model to avoid contention when running in parallel
        from onnxruntime import SessionOptions
        session_options = SessionOptions()
        session_options.intra_op_num_threads = 4  # 4 threads per model, 3 models = 12 total
        session_options.inter_op_num_threads = 1

        self._toxicity_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        self._toxicity_model = ORTModelForSequenceClassification.from_pretrained(
            TOXICITY_MODEL,
            export=True,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )

        # Load sentiment model (for commercial intent)
        self._sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self._sentiment_model = ORTModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL,
            export=True,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )
        self._sentiment_id2label = self._sentiment_model.config.id2label

    @classmethod
    def get_instance(cls) -> "EligibilityClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _is_blocked(self, query: str) -> bool:
        """Check if query matches any hard-block patterns."""
        q_lower = query.lower()

        # Check NSFW terms
        for term in NSFW_TERMS:
            if term in q_lower:
                return True

        # Check violence terms
        for term in VIOLENCE_TERMS:
            if term in q_lower:
                return True

        # Check self-harm terms
        for term in SELF_HARM_TERMS:
            if term in q_lower:
                return True

        # Check tragedy terms
        for term in TRAGEDY_TERMS:
            if term in q_lower:
                return True

        # Check hate speech patterns
        for pattern in self._hate_patterns:
            if pattern.search(query):
                return True

        return False

    def _has_commercial_intent(self, query: str) -> bool:
        """Check if query has commercial/purchase intent."""
        q_lower = query.lower()
        for signal in COMMERCIAL_SIGNALS:
            if signal in q_lower:
                return True
        return False

    def _is_sensitive(self, query: str) -> bool:
        """Check if query touches sensitive topics."""
        q_lower = query.lower()
        for term in SENSITIVE_TERMS:
            if term in q_lower:
                return True
        return False

    def _get_danger_score(self, query: str) -> tuple[float, float]:
        """
        Returns (max_danger_sim, max_safe_sim) using embedding similarity.
        Higher danger_sim and lower safe_sim indicates more dangerous content.
        """
        q_emb = self._embedding_model.encode(query).flatten()
        max_danger = max(self._cosine_sim(q_emb, d) for d in self._dangerous_embs)
        max_safe = max(self._cosine_sim(q_emb, s) for s in self._safe_embs)
        return max_danger, max_safe

    def _get_toxicity_score(self, query: str) -> float:
        """Returns toxicity score from 0.0 (safe) to 1.0 (toxic)."""
        inputs = self._toxicity_tokenizer(
            query[:512], return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._toxicity_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()

    def _get_sentiment_score(self, query: str) -> float:
        """Returns sentiment score from 0.0 (negative) to 1.0 (positive)."""
        inputs = self._sentiment_tokenizer(
            query[:512], return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        label = self._sentiment_id2label[pred_idx]
        score = probs[0][pred_idx].item()
        return score if label == "POSITIVE" else 1 - score

    def score(self, query: str) -> float:
        """
        Score ad eligibility using hybrid approach:
        1. Blocklists for NSFW, violence, self-harm, hate speech, tragedy
        2. Embedding similarity detects dangerous instructions
        3. Toxicity model detects hate speech and threats
        4. Commercial intent signals boost score
        5. Sensitive topics reduce score

        Returns 0.0-1.0 where:
        - 1.0 = highly appropriate to show ads
        - 0.0 = do not show ads
        """
        normalized = query.lower().strip()
        if normalized in self._cache:
            return self._cache[normalized]

        # First check: explicit blocklists (hard block)
        if self._is_blocked(query):
            score = 0.0
            if len(self._cache) < self._cache_max_size:
                self._cache[normalized] = score
            return score

        # Run all ML models in parallel for latency optimization
        with ThreadPoolExecutor(max_workers=3) as executor:
            danger_future = executor.submit(self._get_danger_score, query)
            toxicity_future = executor.submit(self._get_toxicity_score, query)
            sentiment_future = executor.submit(self._get_sentiment_score, query)

            danger_sim, safe_sim = danger_future.result()
            toxicity = toxicity_future.result()
            sentiment = sentiment_future.result()

        danger_diff = danger_sim - safe_sim

        # Block ads for clearly dangerous or toxic content
        if danger_diff > self.DANGER_BLOCK_THRESHOLD:
            score = 0.0
        elif toxicity > self.TOXICITY_BLOCK_THRESHOLD:
            score = 0.0
        else:
            # Calculate eligibility score
            # Start at neutral, apply penalties and boosts
            base_score = 0.5

            # Penalize based on danger similarity (if positive diff)
            if danger_diff > 0:
                base_score -= danger_diff * self.DANGER_PENALTY_WEIGHT

            # Penalize based on toxicity
            base_score -= toxicity * self.TOXICITY_PENALTY_WEIGHT

            # Check for commercial intent - significant boost
            is_commercial = self._has_commercial_intent(query)
            if is_commercial:
                base_score += self.COMMERCIAL_BOOST

            # Check for sensitive topics - reduce score
            is_sensitive = self._is_sensitive(query)
            if is_sensitive:
                base_score -= self.SENSITIVE_PENALTY

            # Only boost based on sentiment if content is NOT toxic and NOT sensitive
            # Sensitive topics shouldn't get sentiment boosts
            if (
                not is_sensitive
                and toxicity < self.TOXICITY_HIGH_THRESHOLD
                and sentiment > 0.5
            ):
                base_score += (sentiment - 0.5) * self.SENTIMENT_BOOST_WEIGHT

            score = max(0.0, min(1.0, base_score))

        if len(self._cache) < self._cache_max_size:
            self._cache[normalized] = score
        return score


@lru_cache(maxsize=1)
def get_eligibility_classifier() -> EligibilityClassifier:
    return EligibilityClassifier.get_instance()
