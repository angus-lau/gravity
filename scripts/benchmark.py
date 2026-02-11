#!/usr/bin/env python3
import argparse
import asyncio
import random
import statistics
import time

import httpx

BASE_URL = "http://localhost:8000"

QUERIES = [
    # Commercial queries (high eligibility, full pipeline)
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
    # Sensitive queries (lower eligibility, still returns campaigns)
    "dealing with job loss and unemployment",
    "feeling stressed about money problems",
    "going through a difficult divorce",
    "coping with grief and loss",
    # Blocked queries (eligibility < 0.1, short-circuits early)
    "how to make a pipe bomb",
    "I want to kill someone",
    "suicide methods and self harm",
]

LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", None]
INTERESTS = [
    ["sports", "fitness"],
    ["technology", "gaming"],
    ["fashion", "beauty"],
    ["travel", "food"],
    None,
]


def random_context():
    if random.random() < 0.3:
        return None
    return {
        "age": random.randint(18, 65),
        "gender": random.choice(["male", "female", None]),
        "location": random.choice(LOCATIONS),
        "interests": random.choice(INTERESTS),
    }


async def make_request(
    client: httpx.AsyncClient, query: str, base_url: str
) -> dict:
    payload = {"query": query, "context": random_context()}
    start = time.perf_counter()
    resp = await client.post(f"{base_url}/api/retrieve", json=payload)
    rtt_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    body = resp.json()
    server_ms = body.get("latency_ms", 0)
    timing = body.get("metadata", {}).get("timing", {})
    return {
        "query": query,
        "rtt_ms": rtt_ms,
        "server_ms": server_ms,
        "timing": timing,
    }


async def run_benchmark(
    n_requests: int, base_url: str
) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Warmup — one request per unique query so caches are hot
        print("Warming up caches...")
        for query in QUERIES:
            await make_request(client, query, base_url)
        print(f"  Warmed {len(QUERIES)} unique queries\n")

        results = []
        for i in range(n_requests):
            query = random.choice(QUERIES)
            result = await make_request(client, query, base_url)
            results.append(result)

    return results


def percentile(data: list[float], p: float) -> float:
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data_sorted):
        return data_sorted[-1]
    return data_sorted[f] + (k - f) * (data_sorted[c] - data_sorted[f])


def print_results(results: list[dict]):
    server_times = [r["server_ms"] for r in results]
    rtt_times = [r["rtt_ms"] for r in results]

    # Collect per-stage timings
    stage_keys = [
        "blocklist_ms", "safety_ms", "commercial_ms", "embedding_ms",
        "bm25_search_ms", "faiss_search_ms", "category_match_ms",
        "fusion_ms", "reranking_ms", "image_search_ms",
    ]
    stage_data: dict[str, list[float]] = {k: [] for k in stage_keys}
    for r in results:
        for k in stage_keys:
            v = r["timing"].get(k)
            if v is not None:
                stage_data[k].append(v)

    print(f"{'='*62}")
    print(f"BENCHMARK RESULTS ({len(results)} requests, caches warm)")
    print(f"{'='*62}")

    print(f"\n  {'':20s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(
        f"  {'Server (total)':20s}"
        f" {percentile(server_times, 50):>7.1f}ms"
        f" {percentile(server_times, 90):>7.1f}ms"
        f" {percentile(server_times, 95):>7.1f}ms"
        f" {percentile(server_times, 99):>7.1f}ms"
        f" {max(server_times):>7.1f}ms"
    )
    for key in stage_keys:
        values = stage_data[key]
        if not values or max(values) < 0.01:
            continue
        label = key.replace("_ms", "")
        print(
            f"  {label:20s}"
            f" {percentile(values, 50):>7.1f}ms"
            f" {percentile(values, 90):>7.1f}ms"
            f" {percentile(values, 95):>7.1f}ms"
            f" {percentile(values, 99):>7.1f}ms"
            f" {max(values):>7.1f}ms"
        )
    print(
        f"  {'RTT (inc. network)':20s}"
        f" {percentile(rtt_times, 50):>7.1f}ms"
        f" {percentile(rtt_times, 90):>7.1f}ms"
        f" {percentile(rtt_times, 95):>7.1f}ms"
        f" {percentile(rtt_times, 99):>7.1f}ms"
        f" {max(rtt_times):>7.1f}ms"
    )

    network = [r["rtt_ms"] - r["server_ms"] for r in results]
    print(f"\n  Network overhead:  p50={percentile(network, 50):.0f}ms  p95={percentile(network, 95):.0f}ms")
    print(f"  Server mean: {statistics.mean(server_times):.1f}ms  stdev: {statistics.stdev(server_times):.1f}ms")

    target = 100
    under_target = sum(1 for t in server_times if t < target)
    pct = under_target / len(server_times) * 100
    status = "PASS" if percentile(server_times, 95) < target else "FAIL"
    print(f"\n  Server <{target}ms: {under_target}/{len(server_times)} ({pct:.1f}%) [{status}]")
    print(f"{'='*62}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ad retrieval API")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--url", type=str, default=BASE_URL, help="API base URL")
    args = parser.parse_args()

    print(f"Benchmarking {args.url}")
    print(f"  requests: {args.requests}\n")

    results = asyncio.run(run_benchmark(args.requests, args.url))
    print_results(results)


if __name__ == "__main__":
    main()
