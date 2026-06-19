FROM python:3.11-slim

WORKDIR /app

# System deps for onnxruntime + fastembed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download ML models at build time so container starts fast
# NLI model (~90MB) and embedding model (~130MB) baked into image
RUN python3 -c "
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/nli-deberta-v3-small')
print('NLI model downloaded')
" && python3 -c "
from fastembed import TextEmbedding
list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))
print('Embedding model downloaded')
"

# Expose Cloud Run's expected port
EXPOSE 8080

# LLM backend via env vars (override at deploy time for Groq)
ENV LLM_BASE_URL=http://localhost:1234/v1
ENV LLM_API_KEY=lm-studio
ENV LLM_MODEL=qwen2.5-7b-instruct

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
