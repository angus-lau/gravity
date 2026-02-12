#!/usr/bin/env python3
"""Benchmark safety classifier: blocklist check + ML-based safety scoring."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.models.safety import get_blocklist_checker, get_safety_classifier

QUERIES_SAFE = [
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
]

QUERIES_SENSITIVE = [
    "dealing with job loss and unemployment",
    "feeling stressed about money problems",
    "going through a difficult divorce",
    "coping with grief and loss",
    "anxiety treatment options",
]

QUERIES_BLOCKED = [
    "how to make a pipe bomb",
    "I want to kill someone",
    "suicide methods and self harm",
]

QUERIES_OBFUSCATED = [
    "b0mb making instructions",
    "su1c1de methods",
    "p.o.r.n videos",
    "k!ll mys3lf",
    "t3rror1st attack",
]

ALL_QUERIES = QUERIES_SAFE + QUERIES_SENSITIVE + QUERIES_BLOCKED + QUERIES_OBFUSCATED


def benchmark_blocklist(checker, queries: list[str], n_iterations: int = 1000) -> dict:
    latencies = []
    for _ in range(10):
        checker.is_blocked(queries[0])

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        checker.is_blocked(query)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(elapsed_us)

    latencies.sort()
    return {
        "unit": "us",
        "min": min(latencies),
        "median": statistics.median(latencies),
        "mean": statistics.mean(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "max": max(latencies),
    }


def benchmark_safety(classifier, queries: list[str], n_iterations: int = 100) -> dict:
    latencies = []
    for _ in range(5):
        classifier.classify(queries[0])

    # Clear cache to get true latencies
    classifier._cache.clear()

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        classifier._cache.clear()
        start = time.perf_counter()
        classifier.classify(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "unit": "ms",
        "min": min(latencies),
        "median": statistics.median(latencies),
        "mean": statistics.mean(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "max": max(latencies),
    }


def benchmark_safety_cached(classifier, queries: list[str], n_iterations: int = 1000) -> dict:
    latencies = []
    # Prime cache
    for q in queries:
        classifier.classify(q)

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        classifier.classify(query)
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


def benchmark_concurrent(classifier, queries: list[str], concurrency: int, n_requests: int = 100) -> dict:
    latencies = []
    classifier.classify(queries[0])

    def classify_query(idx: int) -> float:
        query = queries[idx % len(queries)]
        classifier._cache.clear()
        start = time.perf_counter()
        classifier.classify(query)
        return (time.perf_counter() - start) * 1000

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(classify_query, i) for i in range(n_requests)]
        for future in as_completed(futures):
            latencies.append(future.result())
    total_time = time.perf_counter() - start_time

    latencies.sort()
    return {
        "median": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "throughput": n_requests / total_time,
    }


def print_results(name: str, results: dict):
    unit = results.get("unit", "ms")
    fmt = ".3f" if unit == "us" else ".2f"
    print(f"\n  {name}:")
    print(f"    min:    {results['min']:{fmt}} {unit}")
    print(f"    median: {results['median']:{fmt}} {unit}")
    print(f"    mean:   {results['mean']:{fmt}} {unit}")
    print(f"    p95:    {results['p95']:{fmt}} {unit}")
    if "p99" in results:
        print(f"    p99:    {results['p99']:{fmt}} {unit}")
    print(f"    max:    {results['max']:{fmt}} {unit}")


def main():
    print("=" * 60)
    print("SAFETY CLASSIFIER BENCHMARK")
    print("=" * 60)

    # Blocklist
    print("\nLoading blocklist checker...")
    start = time.perf_counter()
    checker = get_blocklist_checker()
    print(f"  Load time: {(time.perf_counter() - start)*1000:.1f}ms")

    # Safety classifier
    print("\nLoading safety classifier (toxicity + danger embeddings)...")
    start = time.perf_counter()
    classifier = get_safety_classifier()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")

    # Blocklist benchmarks
    print("\n" + "-" * 60)
    print("BLOCKLIST CHECK (1000 iterations)")
    print("-" * 60)

    bl_safe = benchmark_blocklist(checker, QUERIES_SAFE)
    print_results("Safe queries", bl_safe)

    bl_blocked = benchmark_blocklist(checker, QUERIES_BLOCKED)
    print_results("Blocked queries", bl_blocked)

    bl_obfuscated = benchmark_blocklist(checker, QUERIES_OBFUSCATED)
    print_results("Obfuscated queries", bl_obfuscated)

    # Safety classifier benchmarks
    print("\n" + "-" * 60)
    print("SAFETY CLASSIFIER - UNCACHED (100 iterations)")
    print("-" * 60)

    sc_safe = benchmark_safety(classifier, QUERIES_SAFE)
    print_results("Safe queries", sc_safe)

    sc_sensitive = benchmark_safety(classifier, QUERIES_SENSITIVE)
    print_results("Sensitive queries", sc_sensitive)

    sc_blocked = benchmark_safety(classifier, QUERIES_BLOCKED)
    print_results("Blocked queries", sc_blocked)

    # Cached benchmark
    print("\n" + "-" * 60)
    print("SAFETY CLASSIFIER - CACHED (1000 iterations)")
    print("-" * 60)

    sc_cached = benchmark_safety_cached(classifier, ALL_QUERIES)
    print_results("Cache hit latency", sc_cached)

    # Concurrent benchmarks
    print("\n" + "-" * 60)
    print("SAFETY CLASSIFIER - CONCURRENT (100 requests)")
    print("-" * 60)

    for c in [2, 4]:
        conc = benchmark_concurrent(classifier, ALL_QUERIES, c)
        print(f"\n  Concurrency = {c}:")
        print(f"    median: {conc['median']:>6.2f}ms, p95: {conc['p95']:>6.2f}ms, throughput: {conc['throughput']:>5.1f} req/s")

    # Correctness check
    print("\n" + "-" * 60)
    print("CORRECTNESS CHECK")
    print("-" * 60)
    classifier._cache.clear()

    for q in QUERIES_SAFE[:3]:
        r = classifier.classify(q)
        print(f"  Safe:     {q[:40]:<40} blocked={r.is_blocked} base={r.base_score:.3f} tox={r.toxicity:.3f}")

    for q in QUERIES_BLOCKED:
        r = classifier.classify(q)
        print(f"  Blocked:  {q[:40]:<40} blocked={r.is_blocked} base={r.base_score:.3f} tox={r.toxicity:.3f}")

    for q in QUERIES_OBFUSCATED[:3]:
        blocked = checker.is_blocked(q)
        print(f"  Obfusc:   {q[:40]:<40} blocklist_hit={blocked}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Blocklist check:     {bl_safe['median']:.1f} us median (safe), {bl_obfuscated['median']:.1f} us (obfuscated)")
    print(f"  Safety uncached:     {sc_safe['median']:.2f} ms median (safe), {sc_blocked['median']:.2f} ms (blocked)")
    print(f"  Safety cached:       {sc_cached['median']:.1f} us median")
    print("=" * 60)


if __name__ == "__main__":
    main()
