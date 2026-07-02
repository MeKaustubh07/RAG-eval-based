# Multi-stage build keeps the final image lean: install deps in a builder,
# copy only what's needed into the runtime image.

FROM python:3.13-slim AS builder

WORKDIR /app

# CPU-only torch: the default torch wheel drags in ~2GB of CUDA libs we don't
# use. The cpu index gives a much smaller image.
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.13-slim AS runtime

WORKDIR /app

# System libs sentence-transformers/faiss need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY app.py ./
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# Pre-download the embedding + cross-encoder models into the image so the
# container doesn't hit HuggingFace on first request. Cache lives in the image.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

# data/processed (the index) is mounted at runtime — see docker-compose.yml.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
