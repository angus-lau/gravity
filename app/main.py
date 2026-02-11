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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models into memory
    embedding_model = get_embedding_model()
    get_blocklist_checker()
    get_safety_classifier()
    get_commercial_classifier()
    eligibility_classifier = get_eligibility_classifier()
    category_matcher = get_category_matcher()
    get_campaign_index()

    # Warmup
    warmup_query = "best running shoes for marathon training"
    _ = eligibility_classifier.score(warmup_query)
    warmup_emb = embedding_model.encode(warmup_query)
    _ = category_matcher.match(warmup_emb, top_k=5)

    index = get_campaign_index()
    warmup_candidates = index.search(warmup_emb, top_k=1000)
    from app.schemas import UserContext
    warmup_context = UserContext(gender="male", age=30, location="New York", interests=["fitness"])
    _ = rerank(warmup_candidates, warmup_context, top_k=1000)

    # Phase 3: warmup BM25 if available
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

    # ── Tier 0: Blocklist check (synchronous, ~0.01ms) ──
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
                model_versions={
                    "embedding": "all-MiniLM-L6-v2",
                    "eligibility": "modular-tiered",
                },
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    # ── Tier 1: Raw embedding → Safety + Expanded Embedding + BM25 (parallel) ──
    # Compute raw query embedding first (shared with safety classifier)
    t0_raw = time.perf_counter()
    raw_embedding = get_embedding_model().encode(request.query).flatten()
    timing["raw_embedding_ms"] = (time.perf_counter() - t0_raw) * 1000

    # Run safety, expanded embedding, AND BM25 in parallel
    # BM25 only needs the raw query string — no reason to wait for Tier 1 to finish
    def compute_safety():
        t0 = time.perf_counter()
        result = get_safety_classifier().classify(request.query, query_embedding=raw_embedding)
        return result, (time.perf_counter() - t0) * 1000

    def compute_embedding():
        t0 = time.perf_counter()
        ctx = request.context.model_dump() if request.context else None
        emb = get_embedding_model().encode_query(request.query, ctx)
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
    emb_future = loop.run_in_executor(_executor, compute_embedding)
    bm25_future = loop.run_in_executor(_executor, compute_bm25)

    (safety_result, timing["safety_ms"]), (embedding, timing["embedding_ms"]), (bm25_candidates, timing["bm25_search_ms"]) = await asyncio.gather(
        safety_future, emb_future, bm25_future
    )

    # Early stop if unsafe
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
                model_versions={
                    "embedding": "all-MiniLM-L6-v2",
                    "eligibility": "modular-tiered",
                },
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    # ── Tier 2: Commercial intent + FAISS + Categories + Image (parallel) ──
    # BM25 already completed above in Tier 1
    def compute_commercial():
        t0 = time.perf_counter()
        score = get_commercial_classifier().score(request.query, safety_result)
        return score, (time.perf_counter() - t0) * 1000

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

    # Run remaining tier 2 tasks in parallel
    commercial_future = loop.run_in_executor(_executor, compute_commercial)
    categories_future = loop.run_in_executor(_executor, compute_categories)
    faiss_future = loop.run_in_executor(_executor, compute_faiss)
    image_future = loop.run_in_executor(_executor, compute_image_search)

    (
        (eligibility, timing["commercial_ms"]),
        (categories, timing["category_match_ms"]),
        (faiss_candidates, timing["faiss_search_ms"]),
        (image_candidates, timing["image_search_ms"]),
    ) = await asyncio.gather(
        commercial_future, categories_future, faiss_future, image_future
    )

    timing["eligibility_ms"] = timing["blocklist_ms"] + timing["safety_ms"] + timing["commercial_ms"]

    # Short-circuit: don't retrieve campaigns for ineligible queries
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
                    category_match_ms=round(timing["category_match_ms"], 2),
                    faiss_search_ms=round(timing["faiss_search_ms"], 2),
                    reranking_ms=0,
                    total_ms=round(total_ms, 2),
                    blocklist_ms=round(timing.get("blocklist_ms", 0), 2),
                    safety_ms=round(timing.get("safety_ms", 0), 2),
                    commercial_ms=round(timing.get("commercial_ms", 0), 2),
                    bm25_search_ms=round(timing.get("bm25_search_ms", 0), 2),
                    image_search_ms=round(timing.get("image_search_ms", 0), 2),
                ),
                model_versions={
                    "embedding": "all-MiniLM-L6-v2",
                    "eligibility": "modular-tiered",
                },
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    # ── Tier 3: Fusion + Reranking ──
    t0 = time.perf_counter()

    # Fuse results from all retrieval sources
    all_sources = [faiss_candidates]
    if bm25_candidates:
        all_sources.append(bm25_candidates)
    if image_candidates:
        all_sources.append(image_candidates)

    if len(all_sources) > 1:
        from app.retrieval.fusion import reciprocal_rank_fusion
        candidates = reciprocal_rank_fusion(all_sources, top_k=2000)
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
    """Retrieve campaigns by uploading an image. Uses CLIP to match against campaign embeddings."""
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
