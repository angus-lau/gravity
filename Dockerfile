FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ALL models during build (cached in image, not downloaded at runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2', backend='onnx', model_kwargs={'file_name': 'onnx/model_O4.onnx'}); \
    from transformers import AutoTokenizer; \
    from optimum.onnxruntime import ORTModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('martin-ha/toxic-comment-model'); \
    ORTModelForSequenceClassification.from_pretrained('martin-ha/toxic-comment-model', export=True, provider='CPUExecutionProvider'); \
    AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); \
    ORTModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', export=True, provider='CPUExecutionProvider')"

# Pre-download CLIP model for image search (optional but cached)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('clip-ViT-B-32')" || true

COPY app/ app/
COPY data/ data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
