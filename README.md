# RAG LangChain — Production-Ready Template

> A production-grade RAG pipeline using **LangChain + Qdrant + Ollama + Langfuse**.  
> 100% open-source, reproducible on any OS via Docker, deployable to Kubernetes.

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| **API Framework** | FastAPI | Async-native, auto OpenAPI docs, fast |
| **RAG Framework** | LangChain | LCEL pipe chain, rich ecosystem |
| **LLM** | Ollama (Llama 3) | Local, free, no API key |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Best open-source quality/speed ratio |
| **Vector DB** | Qdrant | Production-grade, filtering, gRPC |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder, major quality boost |
| **Cache** | Redis | LLM response caching, session store |
| **Observability** | Langfuse (self-hosted) | Full LLM tracing, open-source |
| **Container** | Docker + multi-stage | Non-root, minimal runtime image |
| **Orchestration** | Kubernetes | 3 namespaces, HPA, PDB, blue/green |
| **CI/CD** | GitHub Actions | 8-stage pipeline with auto-rollback |

---

## Quick Start (Local)

```bash
# 1. Clone and enter the project
git clone https://github.com/your-org/rag-langchain
cd rag-langchain

# 2. Install Python dev dependencies
make install-dev

# 3. Start the full Docker stack
make dev

# 4. Wait ~2 minutes for Ollama to pull llama3, then test
make query-sample
```

**Services after `make dev`:**
- API + Swagger UI: http://localhost:8000/docs
- Qdrant Dashboard: http://localhost:6333/dashboard
- Langfuse UI: http://localhost:3000 *(create account on first visit)*
- Ollama: http://localhost:11434

---

## Project Structure

```
rag-langchain/
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── config.py            # Pydantic Settings (env-based config)
│   ├── dependencies.py      # FastAPI DI: singletons + Depends()
│   ├── routers/
│   │   ├── query.py         # POST /api/v1/query + GET /api/v1/query/stream
│   │   ├── ingest.py        # POST /api/v1/ingest/file + /text
│   │   └── health.py        # GET /health (liveness) + /ready (readiness)
│   ├── services/
│   │   ├── ingestion.py     # PDF → chunks → embed → Qdrant
│   │   ├── retrieval.py     # Hybrid BM25+dense → RRF → rerank
│   │   └── chain.py         # LCEL chain → Ollama → citations
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── observability/
│       └── langfuse.py      # Langfuse tracing wrapper
├── tests/
│   ├── conftest.py          # Shared fixtures + mocks
│   └── unit/
│       └── test_all.py      # Unit tests (80%+ coverage gate)
├── k8s/
│   ├── namespace.yaml       # rag-dev / rag-qa / rag-prod namespaces
│   ├── configmap.yaml       # Non-sensitive config per environment
│   ├── secret.yaml          # Secret template (use External Secrets in prod)
│   ├── serviceaccount.yaml  # Least-privilege RBAC
│   ├── deployment.yaml      # RollingUpdate, probes, PodAntiAffinity
│   ├── service.yaml         # ClusterIP services for all components
│   ├── ingress.yaml         # NGINX + cert-manager TLS
│   ├── hpa.yaml             # CPU+memory autoscaling (3→10 replicas)
│   ├── pdb.yaml             # PodDisruptionBudget (min 2 always running)
│   └── qdrant-statefulset.yaml  # Qdrant 3-node cluster + Redis
├── .github/workflows/
│   └── cicd.yml             # 8-stage GitHub Actions pipeline
├── Dockerfile               # Multi-stage, non-root, reproducible
├── docker-compose.yml       # Full local dev stack
├── pyproject.toml           # Dependencies + tool config (ruff, mypy, pytest)
├── Makefile                 # Developer workflow commands
└── .env.example             # Config template
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Ask a question, get answer + citations |
| `GET` | `/api/v1/query/stream` | Stream answer token-by-token (SSE) |
| `POST` | `/api/v1/ingest/file` | Upload a PDF or text file |
| `POST` | `/api/v1/ingest/text` | Ingest raw text directly |
| `GET` | `/health` | Liveness probe (K8s) |
| `GET` | `/ready` | Readiness probe (K8s) |
| `GET` | `/docs` | Swagger UI (dev/qa only) |

---

## Retrieval Pipeline

```
User Question
    │
    ▼
Query Transform (optional: HyDE / multi-query)
    │
    ├──► Dense Vector Search (Qdrant ANN)          ─┐
    │                                               │ Reciprocal Rank Fusion (RRF)
    └──► BM25 Keyword Search                       ─┘
                    │
                    ▼
            Top-20 fused candidates
                    │
                    ▼
         BAAI/bge-reranker-base (cross-encoder)
                    │
                    ▼
            Top-4 chunks → Prompt
                    │
                    ▼
              Ollama (Llama 3)
                    │
                    ▼
         Answer + Source Citations
```

---

## Observability (Langfuse)

Every query is automatically traced. The `ObservabilityClient` injects a `LangfuseCallbackHandler` into each chain invocation. Langfuse captures:

- Full prompt (system + context + question)
- Full LLM response
- Per-step latency (retrieval, reranking, generation)
- Token counts
- Session continuity (grouped by `session_id`)
- Custom scores (attach RAGAS scores via `obs.score(trace_id, "faithfulness", 0.92)`)

Access traces at http://localhost:3000 (local) or your self-hosted Langfuse instance.

**Langfuse is entirely optional** — if `LANGFUSE_ENABLED=false` or keys are missing, all observability calls become no-ops. The API works without it.

---

## Kubernetes Deployment

### Deploy to dev
```bash
make k8s-apply-dev
```

### Deploy to QA
```bash
make k8s-apply-qa
# or via CI/CD — auto-triggered on merge to main
```

### Deploy to production
```bash
# Via CI/CD (recommended): merge to main → auto QA → manual approval → prod
# Manual emergency deploy:
make k8s-apply-prod
```

### Rollback
```bash
# Rollback QA to previous revision
make k8s-rollback-qa

# Rollback production to previous revision
make k8s-rollback-prod

# Rollback prod to a specific revision
kubectl rollout undo deployment/rag-api -n rag-prod --to-revision=5

# View rollout history
make k8s-history
```

### Deployment Strategy
The production Deployment uses `RollingUpdate` with:
- `maxUnavailable: 0` — zero downtime, old pods live until new ones are Ready
- `maxSurge: 1` — one extra pod spun up during updates
- `minAvailable: 2` via PodDisruptionBudget — never fewer than 2 running
- Auto-rollback in CI/CD if post-deploy smoke tests fail

---

## CI/CD Pipeline (GitHub Actions)

```
Push → Code Quality → Unit Tests
                          │
                     (main only)
                          │
                    Build Image → CVE Scan (Trivy)
                                       │
                              Auto-deploy to QA
                                       │
                           Integration Tests (vs QA)
                                       │
                          ⏸ Manual Approval Gate
                                       │
                   Blue/Green Deploy → Production
                                       │
                          Post-deploy smoke tests
                              │              │
                           PASS            FAIL
                              │              │
                         Tag stable    Auto-rollback
```

**Manual rollback** via GitHub Actions UI:
- Go to Actions → CI/CD → Run workflow → action: `rollback-prod`

---

## Configuration

All config is via environment variables. See `.env.example` for the full list.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `dev` | `dev` / `qa` / `prod` |
| `OLLAMA_MODEL` | `llama3` | Any Ollama-supported model |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RERANKER_ENABLED` | `true` | Toggle cross-encoder reranking |
| `LANGFUSE_ENABLED` | `true` | Toggle observability |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `RETRIEVAL_TOP_K` | `20` | Candidates before reranking |
| `RETRIEVAL_FINAL_K` | `4` | Chunks sent to LLM |

---

## Running Tests

```bash
# Unit tests (fast, no Docker needed)
make test-unit

# Integration tests (requires make dev)
make test-integration

# All quality checks
make quality

# Coverage report
open htmlcov/index.html
```

---

## Security Notes

- Container runs as **non-root user** (UID 1001)
- `readOnlyRootFilesystem` where possible
- All capabilities dropped (`drop: ALL`)
- Secrets managed via **External Secrets Operator** in production — never in Git
- Trivy CVE scan blocks CI if CRITICAL/HIGH vulnerabilities found
- API docs (`/docs`) disabled in production
- Rate limiting enforced at Ingress layer (20 RPS per IP)

---

## License

MIT — use freely, modify, and deploy.
