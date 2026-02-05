import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models.categories import get_category_matcher
from app.models.eligibility import get_eligibility_classifier
from app.models.embeddings import get_embedding_model
from app.retrieval.faiss_index import get_campaign_index
from app.retrieval.ranker import rerank
from app.schemas import (
    HealthResponse,
    ResponseMetadata,
    RetrievalRequest,
    RetrievalResponse,
    TimingMetadata,
)

_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are pre-downloaded in Docker image, this just loads them into memory
    embedding_model = get_embedding_model()
    eligibility_classifier = get_eligibility_classifier()
    category_matcher = get_category_matcher()
    get_campaign_index()

    # Warmup: run dummy inference to JIT-compile and warm caches
    # This ensures the first real request doesn't pay cold-start penalty
    warmup_query = "best running shoes for marathon training"
    _ = eligibility_classifier.score(warmup_query)
    warmup_emb = embedding_model.encode(warmup_query)
    _ = category_matcher.match(warmup_emb, top_k=5)

    # Warmup FAISS search and reranker
    index = get_campaign_index()
    warmup_candidates = index.search(warmup_emb, top_k=1000)
    from app.schemas import UserContext
    warmup_context = UserContext(gender="male", age=30, location="New York", interests=["fitness"])
    _ = rerank(warmup_candidates, warmup_context, top_k=1000)

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

    def compute_eligibility():
        t0 = time.perf_counter()
        score = get_eligibility_classifier().score(request.query)
        return score, (time.perf_counter() - t0) * 1000

    def compute_embedding():
        t0 = time.perf_counter()
        ctx = request.context.model_dump() if request.context else None
        emb = get_embedding_model().encode_query(request.query, ctx)
        return emb, (time.perf_counter() - t0) * 1000

    def compute_categories(embedding):
        t0 = time.perf_counter()
        cats = get_category_matcher().match(embedding, top_k=5)
        return cats, (time.perf_counter() - t0) * 1000

    elig_future = loop.run_in_executor(_executor, compute_eligibility)
    emb_future = loop.run_in_executor(_executor, compute_embedding)

    (eligibility, timing["eligibility_ms"]), (embedding, timing["embedding_ms"]) = await asyncio.gather(
        elig_future, emb_future
    )

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
                    category_match_ms=0,
                    faiss_search_ms=0,
                    reranking_ms=0,
                    total_ms=round(total_ms, 2),
                ),
                model_versions={
                    "embedding": "all-MiniLM-L6-v2",
                    "eligibility": "hybrid-ml",
                },
                query_embedding_dim=384,
                candidates_before_rerank=0,
            ),
        )

    categories, timing["category_match_ms"] = await loop.run_in_executor(
        _executor, compute_categories, embedding
    )

    t0 = time.perf_counter()
    index = get_campaign_index()
    candidates = index.search(embedding, top_k=2000)
    timing["faiss_search_ms"] = (time.perf_counter() - t0) * 1000

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
                "eligibility": "hybrid-ml",
            },
            query_embedding_dim=384,
            candidates_before_rerank=len(candidates),
        ),
    )
