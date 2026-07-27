# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Lean, unbuffered Python. Models cache to a known, image-baked location so the
# non-root runtime user can read them and startup needs no network.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# curl -> container healthcheck; build-essential -> any sdist deps (e.g. hnswlib
# pulled in by chromadb) that lack a prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch FIRST. The default PyPI torch wheel bundles CUDA and
# is ~2 GB+; this gateway only does CPU inference, so the CPU wheel is correct
# and roughly 10x smaller. sentence-transformers then reuses it instead of
# pulling the CUDA build. Deps are installed before app code for layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Pre-download the bi-encoder + cross-encoder at BUILD time so container startup
# is fast and needs no network (see README for the image-size vs startup-time
# tradeoff). Names default to the app's config; override with --build-arg.
# Exporting them as ENV also makes the runtime use exactly the models baked in.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/quora-distilroberta-base
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    RERANKER_MODEL=${RERANKER_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer(os.environ['EMBEDDING_MODEL']); CrossEncoder(os.environ['RERANKER_MODEL']); print('pre-downloaded', os.environ['EMBEDDING_MODEL'], '+', os.environ['RERANKER_MODEL'])"

# Now that the configured models are cached, forbid runtime network fetches so
# startup is guaranteed offline and fast. (Set AFTER the download above, or that
# step would fail.) Overriding a model at runtime means unsetting this.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# App code only (the .dockerignore keeps .env, tests, and eval/ out of the
# image). Then drop to a non-root user that owns the code and the model cache.
COPY app ./app
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /opt/models
USER appuser

EXPOSE 8000

# Weights are baked in, so startup only loads them into RAM (a few seconds) --
# no download. The start-period covers the torch import + reranker load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
