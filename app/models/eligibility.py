import os
import re
from functools import lru_cache

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_MODE = os.getenv("ELIGIBILITY_MODE", "rule-based")  # "rule-based" or "distilbert"

BLOCKLIST_CRITICAL = [
    r"\b(suicide|suicidal|self[- ]?harm|end my life|kill myself|kill me)\b",
    r"\bthoughts of (death|dying|suicide|self[- ]?harm)\b",
    r"\bhow to (make|build|create).*(bomb|explosive|weapon|gun)\b",
    r"\b(pipe bomb|homemade bomb|improvised explosive)\b",
]

BLOCKLIST_SEVERE = [
    # Violence
    r"\b(murder|assassinate|hurt someone|harm someone|attack someone)\b",
    r"\bkill (someone|people|person|him|her|them|a |the )\b",
    # Tragedy/Death
    r"\b(mom|dad|mother|father|parent|wife|husband|child|son|daughter|friend|pet).*(died|passed away|death|killed)\b",
    r"\b(died|passed away|death of|funeral for|lost my)\b",
    r"\b(cancer|terminal illness|terminally ill)\b",
    r"\b(tragedy|tragic accident|devastating loss)\b",
    r"\b(burned down|destroyed|lost everything)\b",
    # NSFW
    r"\b(porn|pornography|nsfw|explicit|xxx|adult content|hentai)\b",
    r"\bwatch.*(porn|xxx|adult)\b",
    # Hate speech
    r"\b(hate|kill|deport|ban) (all )?(immigrants?|muslims?|jews?|blacks?|whites?|gays?|mexicans?|asians?)\b",
    r"\b(racist|racism|bigot|nazi|white supremac|antisemit)\b",
    r"\b(slur|ethnic cleansing|genocide)\b",
    r"\b(terroris[mt]|jihadis[mt]|extremis[mt])\b",
]

COMMERCIAL_SIGNALS = [
    r"\b(buy|purchase|shop|order|deal|discount|sale|price|cheap|affordable)\b",
    r"\b(best|top|recommend|review|compare|vs|versus|rated)\b",
    r"\b(need|want|looking for|searching for|find|get)\b",
    r"\b(where to|how to get|where can i|where do i)\b",
    r"\b(flights?|hotels?|restaurants?|stores?|brands?)\b",
    r"\b(iphone|samsung|galaxy|pixel|laptop|headphones?|shoes?|sneakers?|coffee)\b",
    r"\b(upgrade|organic|wireless|programming|hawaii)\b",
    r"\b(recommendations?|under \$|\$\d+)\b",
    r"\biphone \d+|samsung s\d+\b",
    r"\bfor (flat feet|running|travel|work|home|kids|women|men)\b",
]

INFORMATIONAL_SIGNALS = [
    r"\b(why|how|what|when|where|who)\b.*\??\s*$",
    r"\b(history|explain|understand|learn)\b",
    r"\b(runners?|athletes?|performance|training)\b",
]

# Sensitive topics - not blocked, but lower ad eligibility
SENSITIVE_SIGNALS = [
    r"\b(unemployment|laid off|fired|job loss|lost my job)\b",
    r"\b(stressed|anxious|depressed|overwhelmed|struggling)\b",
    r"\b(divorce|separation|breakup|broke up)\b",
    r"\b(debt|bankruptcy|foreclosure|eviction)\b",
    r"\b(illness|sick|diagnosed|hospital|surgery)\b",
    r"\b(grieving|mourning|loss of)\b",
    r"\b(addiction|rehab|recovery|sober)\b",
    r"\b(abuse|assault|victim|trauma)\b",
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
        self._sensitive_re = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_SIGNALS]

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
                return 0.0  # Critical: never show ads
        for pattern in self._blocklist_severe:
            if pattern.search(text):
                return 0.0  # Severe: never show ads
        return None

    def _count_commercial_signals(self, text: str) -> int:
        return sum(1 for p in self._commercial_re if p.search(text))

    def _count_informational_signals(self, text: str) -> int:
        return sum(1 for p in self._informational_re if p.search(text))

    def _count_sensitive_signals(self, text: str) -> int:
        return sum(1 for p in self._sensitive_re if p.search(text))

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
        sensitive_count = self._count_sensitive_signals(query)

        commercial_boost = min(commercial_count * 0.15, 0.50)
        informational_boost = min(informational_count * 0.10, 0.25)
        sensitive_penalty = min(sensitive_count * 0.15, 0.35)

        if self.mode == "distilbert":
            sentiment_score = self._get_sentiment_score(query)
            base_score = 0.5 + (sentiment_score - 0.5) * 0.2 + commercial_boost + informational_boost - sensitive_penalty
        else:
            base_score = 0.5 + commercial_boost + informational_boost - sensitive_penalty

        score = max(0.0, min(1.0, base_score))

        if len(self._cache) < self._cache_max_size:
            self._cache[normalized] = score
        return score


@lru_cache(maxsize=1)
def get_eligibility_classifier() -> EligibilityClassifier:
    return EligibilityClassifier.get_instance()
