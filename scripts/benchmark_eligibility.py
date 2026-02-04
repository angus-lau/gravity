#!/usr/bin/env python3
"""Benchmark eligibility classifier: rule-based vs DistilBERT."""

import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch

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

COMMERCIAL_SIGNALS = [
    r"\b(buy|purchase|shop|order|deal|discount|sale|price|cheap|affordable)\b",
    r"\b(best|top|recommend|review|compare|vs|versus)\b",
    r"\b(need|want|looking for|searching for|find)\b",
]


class RuleBasedClassifier:
    def __init__(self):
        self._commercial_re = [re.compile(p, re.IGNORECASE) for p in COMMERCIAL_SIGNALS]

    def score(self, query: str) -> float:
        commercial_count = sum(1 for p in self._commercial_re if p.search(query))
        return min(0.5 + commercial_count * 0.12, 1.0)


class DistilBertClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )

    def score(self, query: str) -> float:
        result = self.classifier(query[:512])[0]
        return result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]


class DistilBertOnnxClassifier:
    def __init__(self):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            provider="CPUExecutionProvider",
        )
        self.id2label = self.model.config.id2label

    def score(self, query: str) -> float:
        inputs = self.tokenizer(query[:512], return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        label = self.id2label[pred_idx]
        score = probs[0][pred_idx].item()
        return score if label == "POSITIVE" else 1 - score


def benchmark(classifier, queries: list[str], n_iterations: int = 100) -> dict:
    latencies = []

    # Warmup
    for _ in range(5):
        classifier.score(queries[0])

    for i in range(n_iterations):
        query = queries[i % len(queries)]
        start = time.perf_counter()
        classifier.score(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies.sort()
    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
    }


def benchmark_concurrent(classifier, queries: list[str], concurrency: int, n_requests: int = 100) -> dict:
    latencies = []

    # Warmup
    classifier.score(queries[0])

    def score_query(idx: int) -> float:
        query = queries[idx % len(queries)]
        start = time.perf_counter()
        classifier.score(query)
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
    print("ELIGIBILITY CLASSIFIER BENCHMARK")
    print("=" * 60)

    print("\nLoading rule-based classifier...")
    start = time.perf_counter()
    rule_clf = RuleBasedClassifier()
    print(f"  Load time: {(time.perf_counter() - start)*1000:.1f}ms")

    print("\nLoading DistilBERT (PyTorch) classifier...")
    start = time.perf_counter()
    bert_clf = DistilBertClassifier()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")

    print("\nLoading DistilBERT (ONNX) classifier...")
    start = time.perf_counter()
    onnx_clf = DistilBertOnnxClassifier()
    print(f"  Load time: {time.perf_counter() - start:.2f}s")

    # Verify outputs match
    test_query = "I love this product"
    bert_score = bert_clf.score(test_query)
    onnx_score = onnx_clf.score(test_query)
    print(f"\nOutput consistency: PyTorch={bert_score:.4f}, ONNX={onnx_score:.4f}")

    # Single query benchmark
    print("\n" + "-" * 60)
    print("SINGLE QUERY LATENCY (100 iterations)")
    print("-" * 60)

    print("\nRule-based:")
    rule_results = benchmark(rule_clf, QUERIES)
    print(f"  median: {rule_results['median']:.3f}ms")
    print(f"  p95:    {rule_results['p95']:.3f}ms")

    print("\nDistilBERT (PyTorch):")
    bert_results = benchmark(bert_clf, QUERIES)
    print(f"  median: {bert_results['median']:.2f}ms")
    print(f"  p95:    {bert_results['p95']:.2f}ms")

    print("\nDistilBERT (ONNX):")
    onnx_results = benchmark(onnx_clf, QUERIES)
    print(f"  median: {onnx_results['median']:.2f}ms")
    print(f"  p95:    {onnx_results['p95']:.2f}ms")

    speedup = bert_results["median"] / onnx_results["median"]
    print(f"\n  ONNX is {speedup:.2f}x faster than PyTorch")

    # Concurrent benchmark
    print("\n" + "-" * 60)
    print("CONCURRENT REQUESTS (100 requests)")
    print("-" * 60)

    concurrent_results = {"rule": {}, "bert": {}, "onnx": {}}
    for c in [2, 4, 8]:
        print(f"\n  Concurrency = {c}:")
        rule_conc = benchmark_concurrent(rule_clf, QUERIES, c)
        bert_conc = benchmark_concurrent(bert_clf, QUERIES, c)
        onnx_conc = benchmark_concurrent(onnx_clf, QUERIES, c)
        concurrent_results["rule"][c] = rule_conc
        concurrent_results["bert"][c] = bert_conc
        concurrent_results["onnx"][c] = onnx_conc
        print(f"    Rule-based:       {rule_conc['median']:>6.2f}ms median, {rule_conc['throughput']:>8.0f} req/s")
        print(f"    DistilBERT (PT):  {bert_conc['median']:>6.2f}ms median, {bert_conc['throughput']:>8.1f} req/s")
        print(f"    DistilBERT (ONNX):{onnx_conc['median']:>6.2f}ms median, {onnx_conc['throughput']:>8.1f} req/s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Classifier':<20} {'Median':>10} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    print(f"  {'Rule-based':<20} {rule_results['median']:>9.3f}ms {'-':>10}")
    print(f"  {'DistilBERT (PyTorch)':<20} {bert_results['median']:>9.2f}ms {'1.00x':>10}")
    print(f"  {'DistilBERT (ONNX)':<20} {onnx_results['median']:>9.2f}ms {speedup:>9.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
