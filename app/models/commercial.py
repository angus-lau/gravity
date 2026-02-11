from collections import OrderedDict
from functools import lru_cache

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from app.models.safety import SafetyResult

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

COMMERCIAL_SIGNALS = {
    "buy", "purchase", "order", "shop", "deals", "discount",
    "coupon", "sale", "cheap", "affordable", "price", "cost",
    "best", "top", "top rated", "review", "reviews", "compare", "vs",
    "recommend", "recommendation", "recommended",
    "where to buy", "where can i buy", "how much", "for sale",
    "great deal", "good deal", "best deal",
}

COMMERCIAL_EXCLUSIONS = {"deal with", "deal breaker", "dealing with"}

SENSITIVE_TERMS = {
    "depression", "anxiety", "stressed", "mental health", "therapy",
    "counseling", "psychiatrist",
    "unemployment", "laid off", "bankrupt", "bankruptcy", "debt", "foreclosure",
    "eviction", "homeless", "fired from",
    "divorce", "separation", "breakup", "cheating", "affair",
    "disease", "illness", "symptoms", "diagnosis",
}


class CommercialIntentClassifier:
    _instance: "CommercialIntentClassifier | None" = None
    _cache: OrderedDict[str, float] = OrderedDict()
    _sentiment_cache: dict[str, float] = {}
    _cache_max_size: int = 10000

    COMMERCIAL_BOOST = 0.45
    SENSITIVE_PENALTY = 0.15
    SENTIMENT_BOOST_WEIGHT = 0.3
    TOXICITY_HIGH_THRESHOLD = 0.4
    AD_SAFE_BOOST = 0.15
    AD_SAFE_DANGER_THRESHOLD = 0.2

    def __init__(self):
        from onnxruntime import SessionOptions
        session_options = SessionOptions()
        session_options.intra_op_num_threads = 4
        session_options.inter_op_num_threads = 1

        self._sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        self._sentiment_model = ORTModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL,
            export=True,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )
        self._sentiment_id2label = self._sentiment_model.config.id2label

    @classmethod
    def get_instance(cls) -> "CommercialIntentClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _has_commercial_intent(self, q_lower: str) -> bool:
        for exclusion in COMMERCIAL_EXCLUSIONS:
            if exclusion in q_lower:
                return False
        for signal in COMMERCIAL_SIGNALS:
            if signal in q_lower:
                return True
        return False

    def _is_sensitive(self, q_lower: str) -> bool:
        for term in SENSITIVE_TERMS:
            if term in q_lower:
                return True
        return False

    def _get_sentiment_score(self, query: str) -> float:
        cache_key = query.lower().strip()
        if cache_key in self._sentiment_cache:
            return self._sentiment_cache[cache_key]
        inputs = self._sentiment_tokenizer(
            query[:512], return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        label = self._sentiment_id2label[pred_idx]
        score = probs[0][pred_idx].item()
        result = score if label == "POSITIVE" else 1 - score
        if len(self._sentiment_cache) < self._cache_max_size:
            self._sentiment_cache[cache_key] = result
        return result

    def score(self, query: str, safety: SafetyResult) -> float:
        q_lower = query.lower().strip()
        cache_key = f"{q_lower}:{safety.base_score:.6f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        base_score = safety.base_score

        if self._has_commercial_intent(q_lower):
            base_score += self.COMMERCIAL_BOOST

        is_sensitive = self._is_sensitive(q_lower)
        if is_sensitive:
            base_score -= self.SENSITIVE_PENALTY

        if not is_sensitive and safety.toxicity < self.TOXICITY_HIGH_THRESHOLD:
            if safety.danger_diff < self.AD_SAFE_DANGER_THRESHOLD:
                base_score += self.AD_SAFE_BOOST
            sentiment = self._get_sentiment_score(query)
            if sentiment > 0.5:
                base_score += (sentiment - 0.5) * self.SENTIMENT_BOOST_WEIGHT

        score = max(0.0, min(1.0, base_score))

        self._cache[cache_key] = score
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
        return score

    def score_with_precomputed_sentiment(
        self, query: str, safety: SafetyResult, sentiment: float
    ) -> float:
        q_lower = query.lower().strip()
        cache_key = f"{q_lower}:{safety.base_score:.6f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        base_score = safety.base_score

        if self._has_commercial_intent(q_lower):
            base_score += self.COMMERCIAL_BOOST

        is_sensitive = self._is_sensitive(q_lower)
        if is_sensitive:
            base_score -= self.SENSITIVE_PENALTY

        if not is_sensitive and safety.toxicity < self.TOXICITY_HIGH_THRESHOLD:
            if safety.danger_diff < self.AD_SAFE_DANGER_THRESHOLD:
                base_score += self.AD_SAFE_BOOST
            if sentiment > 0.5:
                base_score += (sentiment - 0.5) * self.SENTIMENT_BOOST_WEIGHT

        score = max(0.0, min(1.0, base_score))

        self._cache[cache_key] = score
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)
        return score


@lru_cache(maxsize=1)
def get_commercial_classifier() -> CommercialIntentClassifier:
    return CommercialIntentClassifier.get_instance()
