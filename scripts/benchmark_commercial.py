#!/usr/bin/env python3
"""Benchmark commercial intent classifier: keyword signals + sentiment model."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.models.commercial import get_commercial_classifier
from app.models.safety import SafetyResult

QUERIES_COMMERCIAL = [
    "where to buy running shoes",
    "best laptop deals under $1000",
    "cheap flights to Paris",
    "compare iPhone vs Samsung",
    "top rated coffee makers",
    "discount gym memberships",
    "affordable wedding venues",
    "buy organic dog food online",
    "best price on standing desk",
    "shop summer dresses on sale",
]

QUERIES_NONCOMMERCIAL = [
    "history of ancient Greece",
    "how does photosynthesis work",
    "who won the 2024 election",
    "climate change research papers",
    "quantum physics explained simply",
]

QUERIES_SENSITIVE = [
    "dealing with depression",
    "filing for bankruptcy",
    "anxiety treatment options",
    "coping with job loss",
    "divorce lawyer near me",
]

ALL_QUERIES = QUERIES_COMMERCIAL + QUERIES_NONCOMMERCIAL + QUERIES_SENSITIVE

SAFE_RESULT = SafetyResult(is_blocked=False, base_score=0.5, toxicity=0.1, danger_diff=-0.2)
BORDERLINE_RESULT = SafetyResult(is_blocked=False, base_score=0.3, toxicity=0.5, danger_diff=0.1)


def benchmark_scoring(classifier, queries: list[str], safety: SafetyResult, n_iterations: int = 100) -> dict:
    latencies = []
    for _ in range(5):
        classifier.score(queries[0], safety)

    classifier._cache.clear()

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        classifier._cache.clear()
        start = time.perf_counter()
        classifier.score(query, safety)
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


def benchmark_cached(classifier, queries: list[str], safety: SafetyResult, n_iterations: int = 1000) -> dict:
    latencies = []
    for q in queries:
        classifier.score(q, safety)

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        classifier.score(query, safety)
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


def benchmark_concurrent(classifier, queries: list[str], safety: SafetyResult, concurrency: int, n_requests: int = 100) -> dict:
    latencies = []
    classifier.score(queries[0], safety)

    def score_query(idx: int) -> float:
        query = queries[idx % len(queries)]
        classifier._cache.clear()
        start = time.perf_counter()
        classifier.score(query, safety)
        return (time.perf_counter() - start) * 1000

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(score_query, i) for i in range(n_requests)]
        for future in as_completed(futures):
            latencies.append(future.result())
    total_time = time.perf_counter() - start_time

    latencies.sort()
    return {
        "median": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "throughput": n_requests / total_time,
    }


def main():
    print("=" * 60)
    print("COMMERCIAL INTENT CLASSIFIER BENCHMARK")
    print("=" * 60)

    print("\nLoading commercial intent classifier...")
    start = time.perf_counter()
    classifier = get_commercial_classifier()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")

    # Uncached benchmarks by query type
    print("\n" + "-" * 60)
    print("UNCACHED SCORING (100 iterations)")
    print("-" * 60)

    print("\n  Commercial queries (keyword match → sentiment runs):")
    comm = benchmark_scoring(classifier, QUERIES_COMMERCIAL, SAFE_RESULT)
    print(f"    median: {comm['median']:.2f}ms, p95: {comm['p95']:.2f}ms")

    print("\n  Non-commercial queries (no keyword → sentiment runs):")
    noncomm = benchmark_scoring(classifier, QUERIES_NONCOMMERCIAL, SAFE_RESULT)
    print(f"    median: {noncomm['median']:.2f}ms, p95: {noncomm['p95']:.2f}ms")

    print("\n  Sensitive queries (sentiment SKIPPED):")
    sens = benchmark_scoring(classifier, QUERIES_SENSITIVE, SAFE_RESULT)
    print(f"    median: {sens['median']:.2f}ms, p95: {sens['p95']:.2f}ms")

    print("\n  High-toxicity context (sentiment SKIPPED):")
    toxic = benchmark_scoring(classifier, QUERIES_COMMERCIAL, BORDERLINE_RESULT)
    print(f"    median: {toxic['median']:.2f}ms, p95: {toxic['p95']:.2f}ms")

    # Cached benchmark
    print("\n" + "-" * 60)
    print("CACHED SCORING (1000 iterations)")
    print("-" * 60)
    cached = benchmark_cached(classifier, ALL_QUERIES, SAFE_RESULT)
    print(f"\n  Cache hit latency:")
    print(f"    median: {cached['median']:.1f} us, p95: {cached['p95']:.1f} us")

    # Concurrent benchmarks
    print("\n" + "-" * 60)
    print("CONCURRENT SCORING (100 requests)")
    print("-" * 60)

    for c in [2, 4]:
        conc = benchmark_concurrent(classifier, ALL_QUERIES, SAFE_RESULT, c)
        print(f"\n  Concurrency = {c}:")
        print(f"    median: {conc['median']:>6.2f}ms, p95: {conc['p95']:>6.2f}ms, throughput: {conc['throughput']:>5.1f} req/s")

    # Correctness check
    print("\n" + "-" * 60)
    print("SCORE DISTRIBUTION")
    print("-" * 60)
    classifier._cache.clear()

    print("\n  Commercial queries (safe context):")
    for q in QUERIES_COMMERCIAL[:5]:
        s = classifier.score(q, SAFE_RESULT)
        print(f"    {s:.3f}  {q}")

    print("\n  Non-commercial queries (safe context):")
    for q in QUERIES_NONCOMMERCIAL[:3]:
        s = classifier.score(q, SAFE_RESULT)
        print(f"    {s:.3f}  {q}")

    print("\n  Sensitive queries (safe context):")
    for q in QUERIES_SENSITIVE[:3]:
        s = classifier.score(q, SAFE_RESULT)
        print(f"    {s:.3f}  {q}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Commercial (with sentiment): {comm['median']:.2f}ms median")
    print(f"  Sensitive (no sentiment):    {sens['median']:.2f}ms median")
    print(f"  Cached:                      {cached['median']:.1f} us median")
    savings = (1 - sens['median'] / comm['median']) * 100 if comm['median'] > 0 else 0
    print(f"  Sentiment skip savings:      {savings:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
