# Gravity Take Home


Created an ad retrieval system that takes in a user query and matches it to relevant advertising campaigns via vector similarity and context aware ranking.

## Features

- **Semantic Search**: Use sentence embeddings to match relevant campaigns to query. (all-MiniLM-L6-v2)
- **Ad Eligibility Scoring**: Filters inappropriate queries (violence, tragedy, NSFW) and boosts commercial intent
- **Category Extraction**: Identifies relevant product/service categories from queries
- **Context-Aware Ranking**: Boosts campaigns based on user location, age, interests, and gender
- **Low Latency**: ONNX-optimized inference, parallel execution, query caching

### Architecture
![alt text](<Context Diagram (Current)-2.svg>)

## Quick Start

### Prerequisites

- Python 3.11+
- ~2GB RAM for model loading

### Installation

```bash
# Clone and setup
git clone https://github.com/angus-lau/gravity.git
cd gravity

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Data & Index

There are two ways to generate synthetic campaign data:

**Option 1: Template randomizer**
```bash
python scripts/generate_test_data.py
```
predefined templates with randomized advertisers, titles, and targeting.

**Option 2: LLM-generated**
```bash
export OPENROUTER_API_KEY=your_key_here
python scripts/generate_campaigns.py
```

Selected Claude Haiku via OpenRouter since it's a lightweight model, and can generate more diverse, authentic campaign data. Requires API key.

Then build the FAISS index:
```bash
python scripts/build_index.py
```

### Run Server

```bash
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "models_loaded": true,
  "campaigns_indexed": 10000
}
```

### Retrieve Campaigns

```bash
curl -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best running shoes for marathon training",
    "context": {
      "gender": "male",
      "age": 28,
      "location": "Boston, MA",
      "interests": ["running", "fitness"]
    }
  }'
```

### Response Schema

```json
{
  "ad_eligibility": 0.94,
  "extracted_categories": ["Running Shoes", "Athletic Apparel", "Fitness Trackers"],
  "campaigns": [
    {
      "campaign_id": "camp_00123",
      "relevance_score": 0.92,
      "advertiser": "Nike",
      "title": "Nike - Running Shoes Sale",
      "categories": ["Running Shoes"]
    }
  ],
  "latency_ms": 45.23,
  "metadata": {
    "timing": {
      "eligibility_ms": 0.15,
      "embedding_ms": 12.34,
      "category_match_ms": 1.23,
      "faiss_search_ms": 18.45,
      "reranking_ms": 8.67,
      "total_ms": 45.23
    },
    "model_versions": {
      "embedding": "all-MiniLM-L6-v2",
      "eligibility": "rule-based"
    },
    "query_embedding_dim": 384,
    "candidates_before_rerank": 2000
  }
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_eligibility.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

## Configuration

| Environment Variable | Options | Default | Description |
|---------------------|---------|---------|-------------|
| `EMBEDDING_BACKEND` | `onnx`, `pytorch` | `onnx` | Embedding model backend |
| `ELIGIBILITY_MODE` | `rule-based`, `distilbert` | `rule-based` | Eligibility classifier mode |

## Project Structure

```
gravity/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── schemas.py           # Pydantic models
│   ├── models/
│   │   ├── embeddings.py    # Query embedding (ONNX)
│   │   ├── eligibility.py   # Ad eligibility scoring
│   │   └── categories.py    # Category extraction
│   └── retrieval/
│       ├── faiss_index.py   # Vector similarity search
│       └── ranker.py        # Context-aware re-ranking
├── data/
│   ├── campaigns.json       # Campaign metadata
│   ├── campaigns.index      # FAISS index
│   └── categories.json      # Category taxonomy
├── tests/                   # Pytest test suite
├── scripts/
│   ├── generate_test_data.py
│   ├── build_index.py
│   └── benchmark_*.py
├── Dockerfile
└── railway.toml
```

## Design Decisions & Trade-offs

| Decision | Trade-off | Rationale |
|----------|-----------|-----------|
| **ONNX over PyTorch** | +500MB Docker image | Reduced single query latency from ~30ms to ~8ms.  |
| **Rule-based eligibility vs DistilBert** | Less detail | <1ms latency, sufficient for blocklist + signals |
| **IndexFlatIP (exact search)** | O(n) complexity | Won't miss any campaigns since only 10k |
| **2000→1000 FAISS + Reranker** | May drop edge cases | FAISS only considers semantic similarity but no context hence why we need the reranker. |
| **Pre-computed embeddings** | Stale if campaigns update | Campaigns are static; rebuild on changes |

## Scalability Considerations

**10x campaigns (100K)**:
- Switch from `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSW` for sub-linear search

**100x QPS**:
- Horizontal scaling via container replicas (stateless design)
- Redis for shared query/embedding cache across instances

## Benchmarks

Run latency benchmarks:

```bash
# API latency benchmark (100 requests, 10 concurrent)
python scripts/benchmark.py -n 100 -c 10

# Model-specific benchmarks
python scripts/benchmark_embeddings.py  # PyTorch vs ONNX
python scripts/benchmark_eligibility.py # Rule-based vs DistilBERT
```

Typical latency breakdown (local, M1 Mac):
- Embedding: ~8ms (ONNX)
- Eligibility: <1ms (rule-based)
- FAISS search: ~15ms
- Reranking: ~10ms
- **Total: ~35-50ms**
