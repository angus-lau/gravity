#!/usr/bin/env python3
"""Benchmark query expansion: synonym lookup."""

import statistics
import time

from app.models.query_expansion import get_query_expander

QUERIES_WITH_SYNONYMS = [
    "comfortable work shoes",
    "cheap running shoes",
    "best luxury watches",
    "lightweight hiking boots",
    "affordable gym equipment",
    "durable outdoor jacket",
    "cozy winter blanket",
    "portable bluetooth speaker",
    "premium coffee beans",
    "waterproof phone case",
]

QUERIES_NO_SYNONYMS = [
    "Nike Pegasus 40",
    "quantum entanglement research papers",
    "how does photosynthesis work",
    "MacBook Pro M3 specs",
    "xylophone maintenance tips",
]

QUERIES_LONG = [
    "best comfortable ergonomic running shoes for flat feet marathon training",
    "cheap affordable budget wireless noise cancelling headphones for working out",
    "lightweight waterproof hiking boots for winter mountain trails with ankle support",
]

ALL_QUERIES = QUERIES_WITH_SYNONYMS + QUERIES_NO_SYNONYMS + QUERIES_LONG


def benchmark_expansion(expander, queries: list[str], n_iterations: int = 500) -> dict:
    latencies = []
    for _ in range(10):
        expander.expand(queries[0])

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        expander.expand(query)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        latencies.append(elapsed_us)

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
    print("QUERY EXPANSION BENCHMARK")
    print("=" * 60)

    print("\nLoading query expander (synonyms)...")
    start = time.perf_counter()
    expander = get_query_expander()
    synonym_load = (time.perf_counter() - start) * 1000
    print(f"  Load time: {synonym_load:.1f}ms")
    print(f"  Synonym entries: {len(expander._synonyms)}")

    # Synonym expansion benchmarks
    print("\n" + "-" * 60)
    print("SYNONYM EXPANSION (500 iterations)")
    print("-" * 60)

    print("\n  Queries with synonym matches:")
    syn = benchmark_expansion(expander, QUERIES_WITH_SYNONYMS)
    print(f"    median: {syn['median']:.1f} us, p95: {syn['p95']:.1f} us, max: {syn['max']:.1f} us")

    print("\n  Queries without synonym matches:")
    nosyn = benchmark_expansion(expander, QUERIES_NO_SYNONYMS)
    print(f"    median: {nosyn['median']:.1f} us, p95: {nosyn['p95']:.1f} us, max: {nosyn['max']:.1f} us")

    print("\n  Long queries:")
    long_q = benchmark_expansion(expander, QUERIES_LONG)
    print(f"    median: {long_q['median']:.1f} us, p95: {long_q['p95']:.1f} us, max: {long_q['max']:.1f} us")

    # Expansion examples
    print("\n" + "-" * 60)
    print("EXPANSION EXAMPLES")
    print("-" * 60)

    for q in QUERIES_WITH_SYNONYMS[:5]:
        expanded = expander.expand(q)
        if expanded != q:
            added = expanded[len(q):].strip()
            print(f"  {q}")
            print(f"    + {added}")
        else:
            print(f"  {q}  (no expansion)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Synonym lookup:          {syn['median']:.0f} us median (with matches)")
    print(f"  Synonym lookup:          {nosyn['median']:.0f} us median (no matches)")
    print(f"  Long query expansion:    {long_q['median']:.0f} us median")
    print(f"  Synonym entries loaded:  {len(expander._synonyms)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
