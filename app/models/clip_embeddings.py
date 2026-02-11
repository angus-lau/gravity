"""CLIP model for text-to-image and image-to-image matching."""

import io
import os
from functools import lru_cache

import numpy as np


class CLIPModel:
    """CLIP ViT-B/32 for encoding text and images into shared embedding space."""

    _instance: "CLIPModel | None" = None
    MODEL_NAME = "clip-ViT-B-32"
    DIMENSION = 512

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dimension = self.DIMENSION

    @classmethod
    def get_instance(cls) -> "CLIPModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_text(self, text: str | list[str], normalize: bool = True) -> np.ndarray:
        """Encode text queries into CLIP embedding space."""
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def encode_image(self, image_bytes: bytes, normalize: bool = True) -> np.ndarray:
        """Encode an uploaded image into CLIP embedding space."""
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        embedding = self.model.encode(
            image,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32).reshape(1, -1)


def is_image_search_enabled() -> bool:
    return os.getenv("ENABLE_IMAGE_SEARCH", "").lower() in ("1", "true")


@lru_cache(maxsize=1)
def get_clip_model() -> CLIPModel:
    return CLIPModel.get_instance()
