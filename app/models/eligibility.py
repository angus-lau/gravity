import os
import re
from functools import lru_cache

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_MODE = os.getenv("ELIGIBILITY_MODE", "rule-based")  # "rule-based" or "distilbert"

BLOCKLIST_CRITICAL = [
    r"\b(suicide|self[- ]?harm|end my life|kill myself|kill me)\b",
]

BLOCKLIST_SEVERE = [
    r"\b(bomb|explosives?|weapon|poison|kill|murder|hurt someone|harm someone|attack)\b",
    r"\bhow to (make|build).*(bomb|explosive|weapon)\b",
    r"\b(died|passed away|death|funeral|cancer|terminal|tragedy)\b",
    r"\b(burned down|destroyed|lost everything)\b",
    r"\b(porn|nsfw|explicit|xxx|adult content|pornographic?)\b",
]

COMMERCIAL_SIGNALS = [
    r"\b(buy|purchase|shop|order|deal|discount|sale|price|cheap|affordable)\b",
    r"\b(best|top|recommend|review|compare|vs|versus)\b",
    r"\b(need|want|looking for|searching for|find)\b",
    r"\b(where to|how to get|where can i|where do i)\b",
    r"\b(flights?|hotels?|restaurants?|stores?)\b",
    r"\b(iphone|samsung|galaxy|pixel|laptop|headphones?|shoes?|coffee)\b",
    r"\b(upgrade|organic|wireless|programming|hawaii)\b",
    r"\b(recommendations?|under \$)\b",
    r"\biphone \d+|samsung s\d+\b",
]

INFORMATIONAL_SIGNALS = [
    r"\b(why|how|what|when|where|who)\b.*\??\s*$",
    r"\b(history|explain|understand|learn)\b",
    r"\b(runners?|athletes?|performance|training)\b",
]


class EligibilityClassifier:
    _instance: "EligibilityClassifier | None" = None
    _cache: dict[str, float] = {}
    _cache_max_size: int = 10000

    def __init__(self, mode: str = DEFAULT_MODE):
        self.mode = mode
        self._blocklist_critical = [re.compile(p, re.IGNORECASE) for p in BLOCKLIST_CRITICAL]
        self._blocklist_severe = [re.compile(p, re.IGNORECASE) for p in BLOCKLIST_SEVERE]
        self._commercial_re = [re.compile(p, re.IGNORECASE) for p in COMMERCIAL_SIGNALS]
        self._informational_re = [re.compile(p, re.IGNORECASE) for p in INFORMATIONAL_SIGNALS]

        if mode == "distilbert":
            self._tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)
            self._model = ORTModelForSequenceClassification.from_pretrained(
                DISTILBERT_MODEL,
                export=True,
                provider="CPUExecutionProvider",
            )
            self._id2label = self._model.config.id2label

    @classmethod
    def get_instance(cls, mode: str = DEFAULT_MODE) -> "EligibilityClassifier":
        if cls._instance is None:
            cls._instance = cls(mode=mode)
        return cls._instance

    def _check_blocklist(self, text: str) -> float | None:
        for pattern in self._blocklist_critical:
            if pattern.search(text):
                return 0.01
        for pattern in self._blocklist_severe:
            if pattern.search(text):
                return 0.05
        return None

    def _count_commercial_signals(self, text: str) -> int:
        return sum(1 for p in self._commercial_re if p.search(text))

    def _count_informational_signals(self, text: str) -> int:
        return sum(1 for p in self._informational_re if p.search(text))

    def _get_sentiment_score(self, query: str) -> float:
        inputs = self._tokenizer(
            query[:512], return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        label = self._id2label[pred_idx]
        score = probs[0][pred_idx].item()
        return score if label == "POSITIVE" else 1 - score

    def score(self, query: str) -> float:
        normalized = query.lower().strip()
        if normalized in self._cache:
            return self._cache[normalized]

        blocklist_score = self._check_blocklist(query)
        if blocklist_score is not None:
            return blocklist_score

        commercial_count = self._count_commercial_signals(query)
        informational_count = self._count_informational_signals(query)
        commercial_boost = min(commercial_count * 0.12, 0.45)
        informational_boost = min(informational_count * 0.08, 0.2)

        if self.mode == "distilbert":
            sentiment_score = self._get_sentiment_score(query)
            base_score = 0.5 + (sentiment_score - 0.5) * 0.2 + commercial_boost + informational_boost
        else:
            base_score = 0.5 + commercial_boost + informational_boost

        score = max(0.0, min(1.0, base_score))

        if len(self._cache) < self._cache_max_size:
            self._cache[normalized] = score
        return score


@lru_cache(maxsize=1)
def get_eligibility_classifier() -> EligibilityClassifier:
    return EligibilityClassifier.get_instance()
