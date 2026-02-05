# Gravity Take Home


An ad retrieval system that takes in a user query and matches it to relevant advertising campaigns via vector similarity and context aware ranking.

> **Note:** The Railway demo uses shared vCPUs which are ~5-10x slower than dedicated hardware. Locally, it runs <25ms P95 but on Railway it varies from ~100-150ms.

## Features

- **Semantic Search**: Use sentence embeddings to match relevant campaigns to query. (all-MiniLM-L6-v2)
- **Ad Eligibility Scoring**: Filters inappropriate queries (violence, tragedy, NSFW) and boosts commercial intent
- **Category Extraction**: Identifies relevant product/service categories from queries
- **Context-Aware Ranking**: Boosts campaigns based on user location, age, interests, and gender
- **Low Latency**: ONNX-optimized inference, parallel execution, query caching

### Architecture
![alt text](<Context Diagram (Current)-2.svg>)

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

**(SELECTED) Option 1: Template randomizer**
```bash
python scripts/generate_test_data.py
```
predefined templates with randomized advertisers, titles, and targeting. Also scrambled search index to prevent bias.

**Option 2: LLM-generated**
```bash
export OPENROUTER_API_KEY=your_key_here
python scripts/generate_campaigns.py
```

Used Claude Haiku via OpenRouter since it's a lightweight model, and can generate more diverse, authentic campaign data. Requires API key.

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

**Live Demo:** https://gravity-production-9f70.up.railway.app

### Health Check

```bash
# Local
curl http://localhost:8000/health

# Railway (live)
curl https://gravity-production-9f70.up.railway.app/health
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
# Local
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

# Railway (live)
curl -X POST https://gravity-production-9f70.up.railway.app/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "best running shoes", "context": {"age": 28}}'
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

## Test Queries

**Commercial (high eligibility):**
- Best running shoes for flat feet
- Buy iPhone 15 Pro Max deals
- Cheap flights to Hawaii
- Top rated noise cancelling headphones
- Where to buy organic coffee beans

**Informational (moderate eligibility):**
- Why do runners get blisters?
- What is the history of the marathon?
- How to train for a 5K
- How to kill weeds in my garden

**Sensitive (lower eligibility):**
- I'm feeling stressed about work
- How do I file for unemployment
- Dealing with anxiety and depression
- Going through a divorce

**Blocked (zero eligibility):**
- My mom just passed away
- I'm having thoughts of self-harm
- How to make a pipe bomb
- Where to watch porn
- I hate all immigrants
- Terrorist attack planning

## Configuration

| Environment Variable | Options | Default | Description |
|---------------------|---------|---------|-------------|
| `EMBEDDING_BACKEND` | `onnx`, `pytorch` | `onnx` | Embedding model backend |
| `ELIGIBILITY_MODE` | `rule-based`, `distilbert` | `rule-based and distilbert` | Eligibility classifier mode |

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
- Switch from `IndexFlatIP` O(n) to `HNSW` O(log n) where latency is ~5ms but recall drops ~95-99%.
- Add managed vector DB e.g Pinecone
- Shard index by category/region
- Redis shared state for caching
- Move data from JSON to database (Postgres)

**100x QPS**:
- Horizontal scaling via container replicas (stateless design)



## Benchmarks

Run latency benchmarks:

```bash
# API latency benchmark (100 requests, 10 concurrent)
python scripts/benchmark.py -n 100

# Model-specific benchmarks
python scripts/benchmark_embeddings.py  # PyTorch vs ONNX
python scripts/benchmark_eligibility.py # Rule-based vs DistilBERT
```

### Latency by Environment

| Component | Local (M1 Mac) | Railway (shared vCPU) |
|-----------|----------------|----------------------|
| Eligibility | ~10-15ms | ~50-130ms |
| Embedding | ~3-5ms | ~8-15ms |
| FAISS search | ~1-2ms | ~5-10ms |
| Reranking | ~2-3ms | ~10-20ms |
| **Total** | **~15-25ms** | **~80-150ms** |

**Note:** Railway uses shared vCPUs which are ~5-10x slower than dedicated hardware. The same code achieves <100ms P95 locally but ~100-150ms on Railway's shared infrastructure. For production, dedicated CPU or GPU instances are recommended.
