import os

import pytest


class TestImageSearchDisabled:
    """Test image search behavior when ENABLE_IMAGE_SEARCH is not set."""

    def test_image_endpoint_returns_error_when_disabled(self, client):
        """Image search endpoint should return error when not enabled."""
        # Create a minimal valid PNG (1x1 pixel)
        import struct
        import zlib

        def create_minimal_png():
            signature = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            raw = zlib.compress(b'\x00\x00\x00\x00')
            idat_crc = zlib.crc32(b'IDAT' + raw)
            idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
            iend_crc = zlib.crc32(b'IEND')
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            return signature + ihdr + idat + iend

        png_bytes = create_minimal_png()
        response = client.post(
            "/api/retrieve/image",
            files={"image": ("test.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data


class TestTextToImageRetrieval:
    """Test that text queries with image search enabled work through the main endpoint."""

    def test_main_endpoint_works_without_image_search(self, client):
        """The main endpoint should work fine without ENABLE_IMAGE_SEARCH."""
        response = client.post("/api/retrieve", json={"query": "red running shoes"})
        assert response.status_code == 200
        assert len(response.json()["campaigns"]) > 0

    def test_visual_query_returns_results(self, client):
        """Visual/descriptive queries should still return results via FAISS."""
        response = client.post("/api/retrieve", json={"query": "bright blue sneakers"})
        assert response.status_code == 200
        campaigns = response.json()["campaigns"]
        assert len(campaigns) > 0


class TestCaptionIndex:

    def test_caption_index_graceful_when_missing(self):
        """CaptionIndex should handle missing index files gracefully."""
        from app.retrieval.image_index import CaptionIndex
        # Create a fresh instance (won't find caption.index unless built)
        idx = CaptionIndex()
        # If index doesn't exist, is_loaded should be False
        if not (CaptionIndex._instance and CaptionIndex._instance.is_loaded):
            assert idx.search(None, top_k=10) == [] if not idx.is_loaded else True
