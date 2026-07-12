# Deployment Runbook

This runbook describes how to verify the production-style Enterprise RAG stack before a demo, interview,
or deployment review.

## 1. Prepare Environment

Copy the example environment file and replace placeholder values locally.

```bash
cp .env.example .env
```

Required for protected API smoke tests:

```text
ENTERPRISE_RAG_API_KEY
ENTERPRISE_RAG_API_KEYS
```

Optional for real provider smoke tests:

```text
OPENAI_API_KEY
QDRANT_URL
REDIS_URL
```

Do not commit real secrets.

## 2. Start Production-Style Stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up api worker qdrant redis prometheus grafana
```

Local URLs:

```text
API:        http://localhost:8000
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```

Grafana local demo login defaults to `admin` / `admin`. Set `GRAFANA_ADMIN_USER` and
`GRAFANA_ADMIN_PASSWORD` for anything beyond a local demo.

## 3. Run Smoke Test

The smoke test validates the deployed API surface:

- `/health`
- `/readiness`
- `/query`
- `/metrics`

```bash
uv run python scripts/smoke_test.py \
  --base-url http://localhost:8000 \
  --api-key "$ENTERPRISE_RAG_API_KEY" \
  --tenant acme \
  --query "What does AUTH-429 affect?" \
  --output-json data/reports/smoke_test.json
```

Require real OpenAI credentials before running:

```bash
uv run python scripts/smoke_test.py \
  --base-url http://localhost:8000 \
  --api-key "$ENTERPRISE_RAG_API_KEY" \
  --tenant acme \
  --require-openai \
  --output-json data/reports/smoke_test.json
```

If required credentials are missing, the script writes a `SKIPPED` report instead of failing.

## 4. Run Load Test

```bash
uv run python scripts/load_test.py \
  --url http://localhost:8000/query \
  --api-key "$ENTERPRISE_RAG_API_KEY" \
  --tenant acme \
  --query "What does AUTH-429 affect?" \
  --requests 50 \
  --concurrency 5 \
  --p95-threshold-ms 3000 \
  --avg-cost-threshold-usd 0.01 \
  --total-cost-threshold-usd 1.00 \
  --output-json data/reports/load_test.json
```

## 5. Check Monitoring

Prometheus should scrape:

```text
enterprise_rag_query_requests_total
enterprise_rag_query_latency_ms_count
enterprise_rag_provider_latency_ms_sum
enterprise_rag_query_estimated_cost_usd_sum
```

Grafana should show the provisioned dashboard:

```text
Enterprise RAG Operations
```

## 6. Common Failures

401 from API:

- `ENTERPRISE_RAG_API_KEY` is missing or does not match `ENTERPRISE_RAG_API_KEYS`.

400 missing tenant:

- Add `--tenant acme` or send `X-Tenant-ID`.

No citations or weak answer:

- Run ingestion before the smoke test.
- Check `/readiness`.
- Confirm query matches indexed test data.

Prometheus target down:

- Check `monitoring/prometheus.yml`.
- Confirm the API container is reachable as `api:8000` inside Compose.
- Confirm `ENTERPRISE_RAG_API_KEY` is available to the Prometheus container.

Grafana dashboard missing:

- Check provisioning mounts in `docker-compose.prod.yml`.
- Restart Grafana after changing dashboard JSON.
