# ⚡ AI Compute Optimizer

An **LLM cost-reduction API gateway**. It sits in front of your language models
and cuts token spend two ways:

1. **Semantic caching** — semantically similar prompts are served from a
   ChromaDB vector cache instead of hitting a model at all.
2. **Complexity-based routing** — simple prompts go to a cheap **local** model
   (Ollama); only genuinely complex prompts reach a paid **remote** model
   (Gemini).

Every request is timed and priced, and the gateway reports a running estimate of
the money saved versus an "always call the frontier model" baseline.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   POST /query           │                FastAPI gateway               │
  ────────────────────►  │                                              │
   { "prompt": "..." }   │   embeddings.py   embed(prompt)              │
                         │        │                                     │
                         │        ▼                                     │
                         │   cache.py  ── lookup (cosine ≥ threshold?) ─┼──► ChromaDB
                         │        │                                     │    (vector store)
                         │   hit  │  miss                               │
                         │   ◄────┘    │                                │
                         │             ▼                                │
                         │   router.py  simple? ──► Ollama (local)  ────┼──► llama3.2
                         │              complex? ─► Gemini (remote)  ───┼──► gemini-1.5-flash
                         │                    │                         │
                         │                    ▼                         │
                         │   cache.py  store(answer)                    │
                         │   metrics.py  record(latency, cost, savings) │
   { "answer", ... }     │                                              │
  ◄────────────────────  │                                              │
                         └──────────────────────────────────────────────┘
```

**Request flow (`/query`):** embed the prompt → look it up in the semantic cache
(cosine similarity ≥ `SIMILARITY_THRESHOLD` is a hit) → on a hit, return the
cached answer at zero token cost → on a miss, the router picks the local or
remote model by complexity, generates an answer, stores it in the cache, and
returns it. Latency and estimated cost are recorded for every request.

### Module map

| File | Responsibility |
|------|----------------|
| [app/main.py](app/main.py) | FastAPI app, lifespan wiring, `/health` `/query` `/metrics` |
| [app/config.py](app/config.py) | Settings from environment variables (no hardcoded keys) |
| [app/models.py](app/models.py) | Pydantic request/response schemas |
| [app/embeddings.py](app/embeddings.py) | sentence-transformers prompt embeddings (lazy load) |
| [app/cache.py](app/cache.py) | ChromaDB semantic cache, cosine + configurable threshold |
| [app/router.py](app/router.py) | Complexity heuristic → `local` / `remote` |
| [app/llm.py](app/llm.py) | Ollama (httpx) and Gemini (google-generativeai) backends |
| [app/metrics.py](app/metrics.py) | Latency + estimated token-cost + savings tracking |
| [app/services.py](app/services.py) | DI container so tests can inject fakes |

---

## Metrics

The gateway exposes a live snapshot at **`GET /metrics`**:

| Metric | Meaning |
|--------|---------|
| `total_requests` | Requests handled |
| `cache_hits` / `cache_misses` | Served from cache vs. forwarded to a model |
| `hit_rate` | `cache_hits / total_requests` |
| `local_routes` / `remote_routes` | Miss traffic split by the router |
| `avg_latency_ms` | Mean end-to-end latency |
| `avg_hit_latency_ms` / `avg_miss_latency_ms` | Latency split by cache outcome |
| `total_input_tokens` / `total_output_tokens` | Prompt vs. completion tokens seen |
| `avg_input_tokens` / `avg_output_tokens` | Per-query averages (drive the projection) |
| `total_cost_usd` | Estimated actual spend |
| `total_baseline_cost_usd` | What "always remote" would have cost |
| `total_savings_usd` | `baseline − actual` — the headline number |
| `projection` | Volume-based monthly/annual savings forecast (see below) |

**Cost model.** Tokens are estimated at ~4 characters/token. Real LLM pricing
bills **input (prompt)** and **output (completion)** tokens at different rates —
output is ~6x input — so each is counted and priced separately, per 1M tokens,
via `REMOTE_INPUT_COST_PER_1M` / `REMOTE_OUTPUT_COST_PER_1M` (and the `LOCAL_*`
equivalents). Defaults reflect Gemini 3.5 Flash list pricing ($1.50 input /
$9.00 output per 1M, verified July 2026). Cache hits cost `$0` and save the full
remote price; local routes are priced at the local rate (default free). Per-request
savings is the remote baseline cost minus the actual cost, so both caching and
local routing contribute.

**Savings projection.** `snapshot().projection` (exposed under `projection` in
`/metrics`) forecasts the cache's dollar value at production scale. Given
`PROJECTED_MONTHLY_QUERIES`, it applies the *measured* hit rate and average
input/output tokens per query against a no-cache baseline:

```
avg_remote_cost_per_query = remote_price(avg_input_tokens, avg_output_tokens)
projected_monthly_savings = monthly_queries × hit_rate × avg_remote_cost_per_query
projected_annual_savings  = projected_monthly_savings × 12
```

---

## Running it

### Prerequisites
- Python 3.11+
- (Optional) [Ollama](https://ollama.com) running locally with a model pulled:
  `ollama pull llama3.2`
- (Optional) A Gemini API key for remote routing:
  <https://aistudio.google.com/app/apikey>

### 1. Local (bare metal)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit values / add GEMINI_API_KEY
uvicorn app.main:app --reload
```

The API is now at <http://localhost:8000> (docs at `/docs`).

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

### 2. Docker Compose (app + ChromaDB)

```bash
export GEMINI_API_KEY=your-key-here   # optional
docker compose up --build
```

This starts the gateway (`:8000`) and a ChromaDB server (`:8001`). Ollama is
expected on the Docker **host** and is reached via `host.docker.internal`.

### 3. Benchmark

With the gateway running:

```bash
python benchmark.py --url http://localhost:8000 --rounds 2
```

Round 1 populates the cache (misses); round 2 should be mostly hits. The script
prints per-phase latency (avg/p95), the cold/warm speedup, and total estimated
savings versus all-remote.

### 4. Dashboard

```bash
streamlit run dashboard.py
```

Shows hit rate, hit-vs-miss latency, routing split, and cumulative cost saved.
Set `GATEWAY_URL` if the gateway is not on `localhost:8000`.

### 5. Tests

```bash
pytest
```

Tests cover **health**, **cache hit**, **cache miss**, and **routing**. They use
in-memory fakes injected through FastAPI's dependency overrides, so no models are
downloaded and no network calls are made — the same suite runs in CI
([.github/workflows/ci.yml](.github/workflows/ci.yml)) on every push.

---

## Configuration

All settings come from environment variables (see [.env.example](.env.example)).
Key knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIMILARITY_THRESHOLD` | `0.85` | Min cosine similarity for a cache hit |
| `COMPLEXITY_WORD_THRESHOLD` | `40` | Word count above which a prompt is "complex" |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `localhost:11434` / `llama3.2` | Local backend |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-1.5-flash` | Remote backend |
| `REMOTE_INPUT_COST_PER_1M` / `REMOTE_OUTPUT_COST_PER_1M` | `1.50` / `9.00` | Remote price per 1M input/output tokens |
| `LOCAL_INPUT_COST_PER_1M` / `LOCAL_OUTPUT_COST_PER_1M` | `0.0` / `0.0` | Local price per 1M input/output tokens |
| `PROJECTED_MONTHLY_QUERIES` | `100000` | Volume assumed for the savings projection |

**Secrets are never hardcoded.** `.env` is git-ignored; only `.env.example`
(with blank keys) is committed.

---

## Project layout

```
app/            # gateway package (see module map above)
tests/          # pytest suite with injected fakes
benchmark.py    # hit-vs-miss latency + savings benchmark
dashboard.py    # Streamlit metrics dashboard
Dockerfile
docker-compose.yml
.github/workflows/ci.yml
```
