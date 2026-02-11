import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile

from app.models.categories import get_category_matcher
from app.models.commercial import get_commercial_classifier
from app.models.eligibility import get_eligibility_classifier
from app.models.embeddings import get_embedding_model
from app.models.safety import get_blocklist_checker, get_safety_classifier
from app.retrieval.faiss_index import get_campaign_index
from app.retrieval.ranker import rerank
from app.schemas import (
    HealthResponse,
    ResponseMetadata,
    RetrievalRequest,
    RetrievalResponse,
    TimingMetadata,
)

_executor = ThreadPoolExecutor(max_workers=5)
_MODEL_VERSIONS = {
    "embedding": "all-MiniLM-L6-v2",
    "eligibility": "modular-tiered",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model = get_embedding_model()
    get_blocklist_checker()
    get_safety_classifier()
    get_commercial_classifier()
    eligibility_classifier = get_eligibility_classifier()
    category_matcher = get_category_matcher()
    get_campaign_index()

    warmup_query = "best running shoes for marathon training"
    _ = eligibility_classifier.score(warmup_query)
    warmup_emb = embedding_model.encode(warmup_query)
    _ = category_matcher.match(warmup_emb, top_k=5)

    index = get_campaign_index()
    warmup_candidates = index.search(warmup_emb, top_k=1000)
    from app.schemas import UserContext
    warmup_context = UserContext(gender="male", age=30, location="New York", interests=["fitness"])
    _ = rerank(warmup_candidates, warmup_context, top_k=1000)

    try:
        from app.retrieval.bm25_index import get_bm25_index
        bm25 = get_bm25_index()
        if bm25.is_loaded:
            _ = bm25.search(warmup_query, top_k=10)
    except ImportError:
        pass

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    index = get_campaign_index()
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        campaigns_indexed=index.size,
    )


@app.post("/api/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    start = time.perf_counter()
    timing = {}
    loop = asyncio.get_event_loop()

    t0 = time.perf_counter()
    blocklist = get_blocklist_checker()
    is_blocked = blocklist.is_blocked(request.query)
    timing["blocklist_ms"] = (time.perf_counter() - t0) * 1000

    if is_blocked:
        total_ms = (time.perf_counter() - start) * 1000
        return RetrievalResponse(
            ad_eligibility=0.0,
            extracted_categories=[],
            campaigns=[],
            latency_ms=round(total_ms, 2),
            metadata=ResponseMetadata(
                timing=TimingMetadata(
                    eligibility_ms=round(timing["blocklist_ms"], 2),
                    embedding_ms=0,
                    category_match_ms=0,
                    faiss_search_ms=0,
                    reranking_ms=0,
                    total_ms=round(total_ms, 2),
                    blocklist_ms=round(timing["blocklist_ms"], 2),
                ),
                model_versions=_MODEL_VERSIONS,
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    t0_raw = time.perf_counter()
    raw_embedding = get_embedding_model().encode(request.query).flatten()
    timing["raw_embedding_ms"] = (time.perf_counter() - t0_raw) * 1000

    query_normalized = request.query.lower().strip()
    safety_classifier = get_safety_classifier()
    cached_safety = safety_classifier._cache.get(query_normalized)
    embedding_model = get_embedding_model()

    if cached_safety is not None:
        t0 = time.perf_counter()
        safety_result = cached_safety
        timing["safety_ms"] = 0.0

        commercial_classifier = get_commercial_classifier()
        precomputed_sentiment = commercial_classifier._get_sentiment_score(request.query)
        timing["commercial_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ctx = request.context.model_dump() if request.context else None
        embedding = embedding_model.encode_query(request.query, ctx)
        timing["embedding_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        try:
            from app.retrieval.bm25_index import get_bm25_index
            bm25 = get_bm25_index()
            bm25_candidates = bm25.search(request.query, top_k=2000) if bm25.is_loaded else []
        except ImportError:
            bm25_candidates = []
        timing["bm25_search_ms"] = (time.perf_counter() - t0) * 1000
    else:
        def compute_safety():
            t0 = time.perf_counter()
            result = safety_classifier.classify(request.query, query_embedding=raw_embedding)
            return result, (time.perf_counter() - t0) * 1000

        def compute_commercial_sentiment():
            t0 = time.perf_counter()
            sentiment = get_commercial_classifier()._get_sentiment_score(request.query)
            return sentiment, (time.perf_counter() - t0) * 1000

        def compute_embedding():
            t0 = time.perf_counter()
            ctx = request.context.model_dump() if request.context else None
            emb = embedding_model.encode_query(request.query, ctx)
            return emb, (time.perf_counter() - t0) * 1000

        def compute_bm25():
            t0 = time.perf_counter()
            try:
                from app.retrieval.bm25_index import get_bm25_index
                bm25 = get_bm25_index()
                if bm25.is_loaded:
                    results = bm25.search(request.query, top_k=2000)
                    return results, (time.perf_counter() - t0) * 1000
            except ImportError:
                pass
            return [], (time.perf_counter() - t0) * 1000

        safety_future = loop.run_in_executor(_executor, compute_safety)
        sentiment_future = loop.run_in_executor(_executor, compute_commercial_sentiment)
        emb_future = loop.run_in_executor(_executor, compute_embedding)
        bm25_future = loop.run_in_executor(_executor, compute_bm25)

        (
            (safety_result, timing["safety_ms"]),
            (precomputed_sentiment, timing["commercial_ms"]),
            (embedding, timing["embedding_ms"]),
            (bm25_candidates, timing["bm25_search_ms"]),
        ) = await asyncio.gather(safety_future, sentiment_future, emb_future, bm25_future)

    if safety_result.is_blocked:
        timing["eligibility_ms"] = timing["safety_ms"]
        total_ms = (time.perf_counter() - start) * 1000
        return RetrievalResponse(
            ad_eligibility=0.0,
            extracted_categories=[],
            campaigns=[],
            latency_ms=round(total_ms, 2),
            metadata=ResponseMetadata(
                timing=TimingMetadata(
                    eligibility_ms=round(timing["eligibility_ms"], 2),
                    embedding_ms=round(timing["embedding_ms"], 2),
                    category_match_ms=0,
                    faiss_search_ms=0,
                    reranking_ms=0,
                    total_ms=round(total_ms, 2),
                    blocklist_ms=round(timing.get("blocklist_ms", 0), 2),
                    safety_ms=round(timing["safety_ms"], 2),
                ),
                model_versions=_MODEL_VERSIONS,
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    t0_combine = time.perf_counter()
    eligibility = get_commercial_classifier().score_with_precomputed_sentiment(
        request.query, safety_result, precomputed_sentiment
    )
    timing["commercial_combine_ms"] = (time.perf_counter() - t0_combine) * 1000

    timing["eligibility_ms"] = timing["blocklist_ms"] + timing["safety_ms"] + timing["commercial_ms"]

    ELIGIBILITY_THRESHOLD = 0.1
    if eligibility < ELIGIBILITY_THRESHOLD:
        total_ms = (time.perf_counter() - start) * 1000
        return RetrievalResponse(
            ad_eligibility=round(eligibility, 4),
            extracted_categories=[],
            campaigns=[],
            latency_ms=round(total_ms, 2),
            metadata=ResponseMetadata(
                timing=TimingMetadata(
                    eligibility_ms=round(timing["eligibility_ms"], 2),
                    embedding_ms=round(timing["embedding_ms"], 2),
                    category_match_ms=0,
                    faiss_search_ms=0,
                    reranking_ms=0,
                    total_ms=round(total_ms, 2),
                    blocklist_ms=round(timing.get("blocklist_ms", 0), 2),
                    safety_ms=round(timing.get("safety_ms", 0), 2),
                    commercial_ms=round(timing.get("commercial_ms", 0), 2),
                    bm25_search_ms=round(timing.get("bm25_search_ms", 0), 2),
                ),
                model_versions=_MODEL_VERSIONS,
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    def compute_categories():
        t0 = time.perf_counter()
        cats = get_category_matcher().match(embedding, top_k=5)
        return cats, (time.perf_counter() - t0) * 1000

    def compute_faiss():
        t0 = time.perf_counter()
        index = get_campaign_index()
        candidates = index.search(embedding, top_k=2000)
        return candidates, (time.perf_counter() - t0) * 1000

    def compute_image_search():
        t0 = time.perf_counter()
        if not os.getenv("ENABLE_IMAGE_SEARCH", "").lower() in ("1", "true"):
            return [], (time.perf_counter() - t0) * 1000
        try:
            from app.retrieval.image_index import get_caption_index
            from app.models.clip_embeddings import get_clip_model
            clip = get_clip_model()
            caption_index = get_caption_index()
            if caption_index.is_loaded:
                clip_emb = clip.encode_text(request.query)
                results = caption_index.search(clip_emb, top_k=500)
                return results, (time.perf_counter() - t0) * 1000
        except ImportError:
            pass
        return [], (time.perf_counter() - t0) * 1000

    categories_future = loop.run_in_executor(_executor, compute_categories)
    faiss_future = loop.run_in_executor(_executor, compute_faiss)
    image_future = loop.run_in_executor(_executor, compute_image_search)

    (
        (categories, timing["category_match_ms"]),
        (faiss_candidates, timing["faiss_search_ms"]),
        (image_candidates, timing["image_search_ms"]),
    ) = await asyncio.gather(categories_future, faiss_future, image_future)

    t0 = time.perf_counter()
    all_sources = [faiss_candidates]
    if bm25_candidates:
        all_sources.append(bm25_candidates)
    if image_candidates:
        all_sources.append(image_candidates)

    if len(all_sources) > 1:
        from app.retrieval.fusion import reciprocal_rank_fusion
        candidates = reciprocal_rank_fusion(all_sources, top_k=1000)
    else:
        candidates = faiss_candidates

    timing["fusion_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    campaigns = rerank(candidates, request.context, top_k=1000)
    timing["reranking_ms"] = (time.perf_counter() - t0) * 1000

    total_ms = (time.perf_counter() - start) * 1000
    timing["total_ms"] = total_ms

    return RetrievalResponse(
        ad_eligibility=round(eligibility, 4),
        extracted_categories=categories,
        campaigns=campaigns,
        latency_ms=round(total_ms, 2),
        metadata=ResponseMetadata(
            timing=TimingMetadata(**{k: round(v, 2) for k, v in timing.items()}),
            model_versions={
                "embedding": "all-MiniLM-L6-v2",
                "eligibility": "modular-tiered",
            },
            query_embedding_dim=384,
            candidates_before_rerank=len(candidates),
        ),
    )


@app.post("/api/retrieve/image")
async def retrieve_by_image(image: UploadFile = File(...), top_k: int = 100):
    from app.schemas import ImageRetrievalResponse

    if not os.getenv("ENABLE_IMAGE_SEARCH", "").lower() in ("1", "true"):
        return {"error": "Image search is not enabled. Set ENABLE_IMAGE_SEARCH=true."}

    start = time.perf_counter()

    image_bytes = await image.read()

    loop = asyncio.get_event_loop()

    def encode_and_search():
        from app.models.clip_embeddings import get_clip_model
        from app.retrieval.image_index import get_caption_index

        clip = get_clip_model()
        caption_index = get_caption_index()

        if not caption_index.is_loaded:
            return []

        clip_emb = clip.encode_image(image_bytes)
        return caption_index.search(clip_emb, top_k=top_k * 2)

    candidates = await loop.run_in_executor(_executor, encode_and_search)

    campaigns = rerank(candidates, context=None, top_k=top_k)

    total_ms = (time.perf_counter() - start) * 1000
    return ImageRetrievalResponse(
        campaigns=campaigns,
        latency_ms=round(total_ms, 2),
        metadata={"source": "clip-image-search", "candidates": len(candidates)},
    )
