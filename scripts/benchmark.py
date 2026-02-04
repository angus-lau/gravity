#!/usr/bin/env python3
import argparse
import asyncio
import random
import statistics
import time

import httpx

BASE_URL = "http://localhost:8000"

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


async def make_request(client: httpx.AsyncClient, query: str, base_url: str) -> float:
    payload = {"query": query, "context": random_context()}
    start = time.perf_counter()
    resp = await client.post(f"{base_url}/api/retrieve", json=payload)
    elapsed = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    return elapsed


async def run_benchmark(n_requests: int, concurrency: int, base_url: str) -> list[float]:
    latencies = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client: httpx.AsyncClient, query: str):
        async with semaphore:
            return await make_request(client, query, base_url)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # warmup
        await make_request(client, QUERIES[0], base_url)

        tasks = [
            bounded_request(client, random.choice(QUERIES)) for _ in range(n_requests)
        ]
        latencies = await asyncio.gather(*tasks)

    return list(latencies)


def percentile(data: list[float], p: float) -> float:
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def print_results(latencies: list[float]):
    latencies.sort()
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS ({len(latencies)} requests)")
    print(f"{'='*50}")
    print(f"  min:    {min(latencies):>8.2f} ms")
    print(f"  p50:    {percentile(latencies, 50):>8.2f} ms")
    print(f"  p90:    {percentile(latencies, 90):>8.2f} ms")
    print(f"  p95:    {percentile(latencies, 95):>8.2f} ms")
    print(f"  p99:    {percentile(latencies, 99):>8.2f} ms")
    print(f"  max:    {max(latencies):>8.2f} ms")
    print(f"  mean:   {statistics.mean(latencies):>8.2f} ms")
    print(f"  stdev:  {statistics.stdev(latencies):>8.2f} ms")
    print(f"{'='*50}")

    target = 100
    under_target = sum(1 for l in latencies if l < target)
    pct = under_target / len(latencies) * 100
    status = "PASS" if percentile(latencies, 95) < target else "FAIL"
    print(f"  <{target}ms: {under_target}/{len(latencies)} ({pct:.1f}%) [{status}]")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ad retrieval API")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--url", type=str, default=BASE_URL, help="API base URL")
    args = parser.parse_args()

    print(f"Benchmarking {args.url}")
    print(f"  requests: {args.requests}")
    print(f"  concurrency: {args.concurrency}")

    latencies = asyncio.run(run_benchmark(args.requests, args.concurrency, args.url))
    print_results(latencies)


if __name__ == "__main__":
    main()
