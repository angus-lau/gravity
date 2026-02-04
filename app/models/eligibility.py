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


class EligibilityClassifier:
    _instance: "EligibilityClassifier | None" = None
    _cache: dict[str, float] = {}
    _cache_max_size: int = 10000

    # Thresholds for scoring
    DANGER_BLOCK_THRESHOLD = 0.4  # Block if danger_diff exceeds this
    TOXICITY_BLOCK_THRESHOLD = 0.8  # Block if toxicity exceeds this
    TOXICITY_HIGH_THRESHOLD = 0.5  # High toxicity threshold for strong penalty
    DANGER_PENALTY_WEIGHT = 0.4  # How much danger similarity reduces score
    TOXICITY_PENALTY_WEIGHT = 0.5  # How much toxicity reduces score
    SENTIMENT_BOOST_WEIGHT = 0.3  # How much positive sentiment boosts score (only applied if low toxicity)

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
        self._toxicity_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        self._toxicity_model = ORTModelForSequenceClassification.from_pretrained(
            TOXICITY_MODEL,
            export=True,
            provider="CPUExecutionProvider",
        )

        # Load sentiment model (for commercial intent)
        self._sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self._sentiment_model = ORTModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL,
            export=True,
            provider="CPUExecutionProvider",
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
        Score ad eligibility using hybrid ML approach:
        1. Embedding similarity detects dangerous instructions (bombs, weapons, etc.)
        2. Toxicity model detects hate speech and threats
        3. Sentiment model boosts commercial/positive queries

        Returns 0.0-1.0 where:
        - 1.0 = highly appropriate to show ads
        - 0.0 = do not show ads
        """
        normalized = query.lower().strip()
        if normalized in self._cache:
            return self._cache[normalized]

        # Get danger score via embedding similarity
        danger_sim, safe_sim = self._get_danger_score(query)
        danger_diff = danger_sim - safe_sim

        # Get toxicity and sentiment scores
        toxicity = self._get_toxicity_score(query)
        sentiment = self._get_sentiment_score(query)

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

            # Only boost based on sentiment if content is NOT toxic
            # This prevents toxic content from being boosted by misleading sentiment
            if toxicity < self.TOXICITY_HIGH_THRESHOLD and sentiment > 0.5:
                base_score += (sentiment - 0.5) * self.SENTIMENT_BOOST_WEIGHT

            score = max(0.0, min(1.0, base_score))

        if len(self._cache) < self._cache_max_size:
            self._cache[normalized] = score
        return score


@lru_cache(maxsize=1)
def get_eligibility_classifier() -> EligibilityClassifier:
    return EligibilityClassifier.get_instance()
