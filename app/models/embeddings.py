import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BACKEND = os.getenv("EMBEDDING_BACKEND", "onnx")


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None
    _cache: dict[str, "np.ndarray"] = {}
    _cache_max_size: int = 10000

    def __init__(self, backend: str = DEFAULT_BACKEND):
        self.backend = backend
        if backend == "onnx":
            self.model = SentenceTransformer(
                MODEL_NAME,
                backend="onnx",
                model_kwargs={
                    "file_name": "onnx/model_O4.onnx",
                    "provider": "CPUExecutionProvider",
                },
            )
        else:
            self.model = SentenceTransformer(MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get_instance(cls, backend: str = DEFAULT_BACKEND) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls(backend=backend)
        return cls._instance

    def encode(self, texts: list[str] | str, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, context: dict | None = None) -> np.ndarray:
        parts = [query]
        if context:
            if context.get("interests"):
                parts.append(" ".join(context["interests"]))
        combined = " ".join(parts)
        cache_key = combined.lower().strip()

        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.encode(combined)
        if len(self._cache) < self._cache_max_size:
            self._cache[cache_key] = embedding
        return embedding


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel.get_instance()
