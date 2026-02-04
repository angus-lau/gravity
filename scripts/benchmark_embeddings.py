#!/usr/bin/env python3
"""Benchmark embedding model latency: PyTorch vs ONNX."""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

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
    "best coffee makers for home",
    "wireless earbuds comparison",
    "hiking boots waterproof",
    "yoga mat non-slip",
    "standing desk adjustable",
]


def benchmark_model(model: SentenceTransformer, queries: list[str], n_iterations: int = 100) -> dict:
    """Run benchmark for a single model."""
    latencies = []

    # Warmup
    for _ in range(5):
        model.encode(queries[0], normalize_embeddings=True)

    # Single query benchmark
    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        model.encode(query, normalize_embeddings=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()

    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies),
        "p90": latencies[int(len(latencies) * 0.90)],
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
    }


def benchmark_batch(model: SentenceTransformer, queries: list[str], batch_size: int = 10, n_iterations: int = 50) -> dict:
    """Run batch benchmark."""
    latencies = []

    # Warmup
    model.encode(queries[:batch_size], normalize_embeddings=True)

    for _ in range(n_iterations):
        batch = queries[:batch_size]
        start = time.perf_counter()
        model.encode(batch, normalize_embeddings=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()

    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
    }


def benchmark_concurrent(
    model: SentenceTransformer,
    queries: list[str],
    concurrency: int,
    n_requests: int = 100,
) -> dict:
    """Run concurrent benchmark with thread pool."""
    latencies = []

    # Warmup
    model.encode(queries[0], normalize_embeddings=True)

    def encode_query(idx: int) -> float:
        query = queries[idx % len(queries)]
        start = time.perf_counter()
        model.encode(query, normalize_embeddings=True)
        return (time.perf_counter() - start) * 1000

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(encode_query, i) for i in range(n_requests)]
        for future in as_completed(futures):
            latencies.append(future.result())
    total_time = time.perf_counter() - start_time

    latencies.sort()
    throughput = n_requests / total_time

    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "throughput": throughput,
        "total_time": total_time,
    }


def print_results(name: str, results: dict):
    """Print benchmark results."""
    print(f"\n  {name}:")
    print(f"    min:    {results['min']:>7.2f} ms")
    print(f"    median: {results['median']:>7.2f} ms")
    print(f"    mean:   {results['mean']:>7.2f} ms")
    print(f"    p90:    {results.get('p90', 'N/A'):>7.2f} ms" if 'p90' in results else "")
    print(f"    p95:    {results['p95']:>7.2f} ms")
    if 'p99' in results:
        print(f"    p99:    {results['p99']:>7.2f} ms")
    print(f"    max:    {results['max']:>7.2f} ms")
    if 'throughput' in results:
        print(f"    throughput: {results['throughput']:>5.1f} req/s")


def main():
    print("=" * 60)
    print("EMBEDDING MODEL LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")

    # Load PyTorch model
    print("\nLoading PyTorch model...")
    start = time.perf_counter()
    pytorch_model = SentenceTransformer(MODEL_NAME)
    pytorch_load_time = time.perf_counter() - start
    print(f"  Load time: {pytorch_load_time:.2f}s")

    # Load ONNX model (optimized O4, CPU-only provider)
    print("\nLoading ONNX model (O4 optimized, CPU provider)...")
    start = time.perf_counter()
    onnx_model = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={
            "file_name": "onnx/model_O4.onnx",
            "provider": "CPUExecutionProvider",
        },
    )
    onnx_load_time = time.perf_counter() - start
    print(f"  Load time: {onnx_load_time:.2f}s")

    # Verify outputs match
    print("\nVerifying output consistency...")
    test_query = "test query for verification"
    pytorch_out = pytorch_model.encode(test_query, normalize_embeddings=True)
    onnx_out = onnx_model.encode(test_query, normalize_embeddings=True)
    cosine_sim = np.dot(pytorch_out, onnx_out)
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    # Run benchmarks
    print("\n" + "-" * 60)
    print("SINGLE QUERY LATENCY (100 iterations)")
    print("-" * 60)

    print("\nBenchmarking PyTorch...")
    pytorch_single = benchmark_model(pytorch_model, QUERIES)
    print_results("PyTorch", pytorch_single)

    print("\nBenchmarking ONNX...")
    onnx_single = benchmark_model(onnx_model, QUERIES)
    print_results("ONNX", onnx_single)

    speedup = pytorch_single["mean"] / onnx_single["mean"]
    print(f"\n  Speedup: {speedup:.2f}x faster with ONNX")

    # Batch benchmark
    print("\n" + "-" * 60)
    print("BATCH LATENCY (batch_size=10, 50 iterations)")
    print("-" * 60)

    print("\nBenchmarking PyTorch batch...")
    pytorch_batch = benchmark_batch(pytorch_model, QUERIES)
    print_results("PyTorch", pytorch_batch)

    print("\nBenchmarking ONNX batch...")
    onnx_batch = benchmark_batch(onnx_model, QUERIES)
    print_results("ONNX", onnx_batch)

    batch_speedup = pytorch_batch["mean"] / onnx_batch["mean"]
    print(f"\n  Speedup: {batch_speedup:.2f}x faster with ONNX")

    # Concurrent benchmarks
    concurrency_levels = [2, 4, 8]
    concurrent_results = {"pytorch": {}, "onnx": {}}

    print("\n" + "-" * 60)
    print("CONCURRENT REQUESTS (100 requests per level)")
    print("-" * 60)

    for c in concurrency_levels:
        print(f"\n  Concurrency = {c}:")

        pytorch_conc = benchmark_concurrent(pytorch_model, QUERIES, c)
        concurrent_results["pytorch"][c] = pytorch_conc
        print(f"    PyTorch: {pytorch_conc['median']:>6.1f}ms median, {pytorch_conc['p95']:>6.1f}ms p95, {pytorch_conc['throughput']:>5.1f} req/s")

        onnx_conc = benchmark_concurrent(onnx_model, QUERIES, c)
        concurrent_results["onnx"][c] = onnx_conc
        print(f"    ONNX:    {onnx_conc['median']:>6.1f}ms median, {onnx_conc['p95']:>6.1f}ms p95, {onnx_conc['throughput']:>5.1f} req/s")

        tp_speedup = onnx_conc['throughput'] / pytorch_conc['throughput']
        print(f"    -> ONNX throughput: {tp_speedup:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Single query speedup: {speedup:.2f}x")
    print(f"  Batch speedup:        {batch_speedup:.2f}x")
    print(f"  Output consistency:   {cosine_sim:.6f} cosine similarity")
    print("\n  Concurrent throughput (req/s):")
    print(f"    {'Concurrency':<12} {'PyTorch':>10} {'ONNX':>10} {'Speedup':>10}")
    for c in concurrency_levels:
        pt = concurrent_results["pytorch"][c]["throughput"]
        ox = concurrent_results["onnx"][c]["throughput"]
        print(f"    {c:<12} {pt:>10.1f} {ox:>10.1f} {ox/pt:>10.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
