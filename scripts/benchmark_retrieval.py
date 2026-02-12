#!/usr/bin/env python3
"""Benchmark retrieval pipeline: BM25 search, FAISS search, RRF fusion, and reranking."""

import statistics
import time

from app.models.embeddings import get_embedding_model
from app.retrieval.bm25_index import get_bm25_index
from app.retrieval.faiss_index import get_campaign_index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.ranker import rerank
from app.schemas import UserContext

QUERIES = [
    "running shoes for marathon training",
    "organic baby food subscription",
    "luxury watches for men",
    "home fitness equipment deals",
    "sustainable fashion brands",
    "electric vehicle charging stations",
    "gaming laptops under $1500",
    "pet insurance for dogs",
    "travel credit cards with rewards",
    "smart home security systems",
    "vegan protein powder",
    "online coding bootcamp",
    "wedding photography services",
    "noise cancelling headphones",
    "meal delivery kits",
]

CONTEXTS = [
    UserContext(location="New York", age=28, gender="female", interests=["fitness", "cooking"]),
    UserContext(location="Los Angeles", age=35, gender="male", interests=["technology", "gaming"]),
    UserContext(location="Chicago", age=45, gender="female", interests=["fashion", "travel"]),
    None,
]


def benchmark_bm25(index, queries: list[str], n_iterations: int = 200) -> dict:
    latencies = []
    for _ in range(10):
        index.search(queries[0])

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        index.search(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "min": min(latencies),
        "median": statistics.median(latencies),
        "mean": statistics.mean(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "max": max(latencies),
    }


def benchmark_faiss(index, embedding_model, queries: list[str], n_iterations: int = 200) -> dict:
    latencies_embed = []
    latencies_search = []
    latencies_total = []

    # Warmup
    for _ in range(5):
        emb = embedding_model.encode(queries[0])
        index.search(emb)

    for i in range(n_iterations):
        query = queries[i % len(queries)]

        start = time.perf_counter()
        emb = embedding_model.encode(query)
        embed_time = time.perf_counter() - start

        start = time.perf_counter()
        index.search(emb)
        search_time = time.perf_counter() - start

        latencies_embed.append(embed_time * 1000)
        latencies_search.append(search_time * 1000)
        latencies_total.append((embed_time + search_time) * 1000)

    latencies_embed.sort()
    latencies_search.sort()
    latencies_total.sort()

    return {
        "embed": {
            "median": statistics.median(latencies_embed),
            "p95": latencies_embed[int(len(latencies_embed) * 0.95)],
        },
        "search": {
            "median": statistics.median(latencies_search),
            "p95": latencies_search[int(len(latencies_search) * 0.95)],
        },
        "total": {
            "median": statistics.median(latencies_total),
            "p95": latencies_total[int(len(latencies_total) * 0.95)],
        },
    }


def benchmark_fusion(faiss_results_list, bm25_results_list, n_iterations: int = 500) -> dict:
    latencies = []

    for _ in range(10):
        reciprocal_rank_fusion([faiss_results_list[0], bm25_results_list[0]])

    for i in range(n_iterations):
        idx = i % len(faiss_results_list)
        start = time.perf_counter()
        reciprocal_rank_fusion([faiss_results_list[idx], bm25_results_list[idx]])
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(elapsed_us)

    latencies.sort()
    return {
        "unit": "us",
        "min": min(latencies),
        "median": statistics.median(latencies),
        "mean": statistics.mean(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "max": max(latencies),
    }


def benchmark_rerank(candidates_list, contexts: list, n_iterations: int = 200) -> dict:
    latencies = []

    for _ in range(5):
        rerank(candidates_list[0], contexts[0])

    for i in range(n_iterations):
        idx = i % len(candidates_list)
        ctx = contexts[i % len(contexts)]
        start = time.perf_counter()
        rerank(candidates_list[idx], ctx)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "min": min(latencies),
        "median": statistics.median(latencies),
        "mean": statistics.mean(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "max": max(latencies),
    }


def main():
    print("=" * 60)
    print("RETRIEVAL PIPELINE BENCHMARK")
    print("=" * 60)

    # Load components
    print("\nLoading embedding model...")
    start = time.perf_counter()
    embedding_model = get_embedding_model()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")

    print("\nLoading FAISS index...")
    start = time.perf_counter()
    faiss_index = get_campaign_index()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")
    print(f"  Campaigns: {faiss_index.index.ntotal}")

    print("\nLoading BM25 index...")
    start = time.perf_counter()
    bm25_index = get_bm25_index()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")
    print(f"  Campaigns: {len(bm25_index.campaign_ids)}")

    # BM25 benchmark
    print("\n" + "-" * 60)
    print("BM25 SEARCH (200 iterations)")
    print("-" * 60)

    bm25_results = benchmark_bm25(bm25_index, QUERIES)
    print(f"\n  median: {bm25_results['median']:.2f}ms")
    print(f"  p95:    {bm25_results['p95']:.2f}ms")
    print(f"  p99:    {bm25_results['p99']:.2f}ms")

    # FAISS benchmark
    print("\n" + "-" * 60)
    print("FAISS SEARCH (200 iterations)")
    print("-" * 60)

    faiss_results = benchmark_faiss(faiss_index, embedding_model, QUERIES)
    print(f"\n  Embedding:  {faiss_results['embed']['median']:.2f}ms median, {faiss_results['embed']['p95']:.2f}ms p95")
    print(f"  Search:     {faiss_results['search']['median']:.2f}ms median, {faiss_results['search']['p95']:.2f}ms p95")
    print(f"  Total:      {faiss_results['total']['median']:.2f}ms median, {faiss_results['total']['p95']:.2f}ms p95")

    # Pre-compute results for fusion and rerank benchmarks
    print("\n  Pre-computing results for fusion/rerank benchmarks...")
    faiss_results_list = []
    bm25_results_list = []
    for q in QUERIES:
        emb = embedding_model.encode(q)
        faiss_results_list.append(faiss_index.search(emb))
        bm25_results_list.append(bm25_index.search(q))

    # Fusion benchmark
    print("\n" + "-" * 60)
    print("RRF FUSION (500 iterations)")
    print("-" * 60)

    fusion_results = benchmark_fusion(faiss_results_list, bm25_results_list)
    avg_faiss = statistics.mean(len(r) for r in faiss_results_list)
    avg_bm25 = statistics.mean(len(r) for r in bm25_results_list)
    print(f"\n  Avg input sizes: FAISS={avg_faiss:.0f}, BM25={avg_bm25:.0f}")
    print(f"  median: {fusion_results['median']:.0f} us")
    print(f"  p95:    {fusion_results['p95']:.0f} us")
    print(f"  max:    {fusion_results['max']:.0f} us")

    # Rerank benchmark
    print("\n" + "-" * 60)
    print("RERANKING (200 iterations)")
    print("-" * 60)

    candidates_list = []
    for i in range(len(QUERIES)):
        fused = reciprocal_rank_fusion([faiss_results_list[i], bm25_results_list[i]])
        candidates_list.append(fused)

    avg_candidates = statistics.mean(len(c) for c in candidates_list)
    print(f"\n  Avg candidates per query: {avg_candidates:.0f}")

    rerank_results = benchmark_rerank(candidates_list, CONTEXTS)
    print(f"  median: {rerank_results['median']:.2f}ms")
    print(f"  p95:    {rerank_results['p95']:.2f}ms")
    print(f"  p99:    {rerank_results['p99']:.2f}ms")

    # Full pipeline (BM25 + FAISS + Fusion + Rerank)
    print("\n" + "-" * 60)
    print("FULL RETRIEVAL PIPELINE (100 iterations)")
    print("-" * 60)

    pipeline_latencies = []
    for _ in range(3):
        emb = embedding_model.encode(QUERIES[0])
        faiss_r = faiss_index.search(emb)
        bm25_r = bm25_index.search(QUERIES[0])
        fused = reciprocal_rank_fusion([faiss_r, bm25_r])
        rerank(fused, CONTEXTS[0])

    for i in range(100):
        query = QUERIES[i % len(QUERIES)]
        ctx = CONTEXTS[i % len(CONTEXTS)]

        start = time.perf_counter()
        emb = embedding_model.encode(query)
        faiss_r = faiss_index.search(emb)
        bm25_r = bm25_index.search(query)
        fused = reciprocal_rank_fusion([faiss_r, bm25_r])
        rerank(fused, ctx)
        elapsed_ms = (time.perf_counter() - start) * 1000
        pipeline_latencies.append(elapsed_ms)

    pipeline_latencies.sort()
    print(f"\n  median: {statistics.median(pipeline_latencies):.2f}ms")
    print(f"  p95:    {pipeline_latencies[int(len(pipeline_latencies)*0.95)]:.2f}ms")
    print(f"  p99:    {pipeline_latencies[int(len(pipeline_latencies)*0.99)]:.2f}ms")
    print(f"  max:    {max(pipeline_latencies):.2f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Component':<25} {'Median':>10} {'P95':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'BM25 search':<25} {bm25_results['median']:>9.2f}ms {bm25_results['p95']:>9.2f}ms")
    print(f"  {'FAISS embed+search':<25} {faiss_results['total']['median']:>9.2f}ms {faiss_results['total']['p95']:>9.2f}ms")
    print(f"  {'  - embedding only':<25} {faiss_results['embed']['median']:>9.2f}ms {faiss_results['embed']['p95']:>9.2f}ms")
    print(f"  {'  - search only':<25} {faiss_results['search']['median']:>9.2f}ms {faiss_results['search']['p95']:>9.2f}ms")
    print(f"  {'RRF fusion':<25} {fusion_results['median']/1000:>9.3f}ms {fusion_results['p95']/1000:>9.3f}ms")
    print(f"  {'Reranking':<25} {rerank_results['median']:>9.2f}ms {rerank_results['p95']:>9.2f}ms")
    print(f"  {'Full pipeline':<25} {statistics.median(pipeline_latencies):>9.2f}ms {pipeline_latencies[int(len(pipeline_latencies)*0.95)]:>9.2f}ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
