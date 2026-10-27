# Project instructions — Drug Label RAG

Practice project run as if it were in production: a RAG + agent API over public drug labels
(openFDA), with an eval suite, deployed to AWS. `CLAUDE.md` is a symlink to this file.

## What it is

Ask questions about drug labels ("Can I take ibuprofen with warfarin?") and get answers that
**cite the label sections** they come from. Information only, never medical advice.

## Stack

| Layer | Choice |
|---|---|
| API | Python 3.13, FastAPI, uv, ruff, pytest (`api/`) |
| Web | React + TypeScript, Vite, bun, oxlint (`web/`) |
| Data | Postgres 17 + pgvector |
| LLM | Claude via AWS Bedrock; embeddings Titan v2 on Bedrock |
| Infra | Terraform on AWS: ECS Fargate, RDS, ECR, S3 + CloudFront |
| CI/CD | GitHub Actions (OIDC to AWS) |

## Constraints

1. Every answer cites its sources (drug + label section). No source, no answer.
2. Never present output as medical advice; the API adds a disclaimer.
3. The eval suite gates merges: retrieval recall and answer faithfulness must not regress.
4. No secrets in the repo: local `.env`, Secrets Manager in AWS.

## Roadmap

- [x] 0. Scaffold: FastAPI + React, Postgres/pgvector in compose, CI
- [x] 1. Ingestion: openFDA labels → section chunks → Titan embeddings → pgvector
- [ ] 2. RAG: `POST /ask`, hybrid retrieval (vector + full-text), cited answers, streaming
  (code done; generation blocked on the Bedrock Marketplace agreement, see Bedrock notes;
  streaming missing)
- [ ] 3. Agent: tool-use loop (`search_labels`, `get_section`, `compare_drugs`)
- [ ] 4. Eval: golden set, recall@k, LLM-judge faithfulness/correctness, CI gate
  (golden set + runner done; retrieval baseline 30/34 = 88%; full run pending; before trusting
  it: oracle/null checks on the judge; then a CI job on PRs touching `api/`)
- [ ] 4b. Compare an open model on Bedrock against Claude on the same golden set
  (quality vs cost per question); add a provider switch only for that comparison
- [ ] 5. Prod-readiness: structured logs, audit log, metrics, rate limit, guardrails
- [ ] 6. Infra: Terraform for AWS (IAM roles + GitHub OIDC, never root credentials)
- [ ] 7. CD: plan on PR, build/push/deploy on main

## Commands

```bash
docker compose up -d --build        # db + api on :8000
cd api && uv run pytest              # tests
cd api && uv run ruff check . && uv run ruff format .
cd web && bun dev                    # web on :5173
cd api && uv run python -m api.ingest [--embed]   # load data/drugs.txt from openFDA
cd api && uv run python -m api.evaluate [--retrieval-only] [--limit N]   # eval, see below
```

## Data notes

- One label per drug in `data/drugs.txt`: the newest single-ingredient label. openFDA name
  search also matches combinations, so `openfda.fetch_label` skips "X and Y", "X/Y", "X, Y".
- Only the sections in `openfda.SECTIONS` are indexed; packaging and ingredient lists are noise.
- Re-running ingest only rewrites chunks for labels whose version changed; `--embed` fills
  chunks with a null vector, so both steps are idempotent.
- Schema lives in `api/src/api/migrations/` (numbered SQL, applied on connect under an
  advisory lock).

## RAG notes

- Retrieval (`retrieval.py`): pgvector cosine + Postgres full-text, merged with reciprocal rank
  fusion in one SQL query. Pass vectors with `::vector`: psycopg sends lists as `float8[]`.
- Generation (`rag.py`): passages go to Claude as `search_result` blocks with citations on,
  so citations come from the API, not from parsing `[n]` out of the text.
- Model `anthropic.claude-opus-5-5` on Bedrock (Mantle client), effort `medium`. Bedrock has no
  server-side `fallbacks`: `BetaRefusalFallbackMiddleware` retries a refusal on Opus 4.8.
- Tests need Postgres (`docker compose up -d db`); CI runs it as a service.

## Eval notes

- `data/eval/golden.jsonl`: 34 questions with expected drug + label sections + key facts, each
  checked against the stored label text, plus 3 out-of-corpus negatives (must abstain).
- Metrics: `retrieval_hit` (expected drug and section in the top 8), `fact_recall`, `faithful`
  (no claim unsupported by the passages), `cited`, `abstain_ok`. Judge: Haiku 5.5 with
  structured output (`messages.parse`), never the generator model. Runs land in `eval/runs/`.
- Thresholds in `evaluate.THRESHOLDS` are a provisional baseline: raise them once a few runs show
  the noise floor. Full run costs about EUR 2 (Opus generation); `--retrieval-only` only embeds.
- Gold is changed only when the label text proves it too narrow (q18, q26 accept
  `warnings_and_cautions`: the same facts are in 5.1/5.2), never to make the system pass.
- Known retrieval misses, left for the agent: "What is Praluent used for?" (full-text `use`
  matches the "... use" section titles) and "blood thinner pills" (lay term, no drug named).

## Bedrock notes

- Region eu-central-1. Claude there has no in-region endpoint, only Global/EU profiles
  (`global.anthropic.…`, `eu.anthropic.…`): if the plain `anthropic.` ID still 404s once access
  works, switch `llm.MODEL` to the `eu.` profile to keep data in the EU.
- A 404 "model does not exist" for every Claude model means the account has no Marketplace
  agreement (`aws bedrock get-foundation-model-availability` shows
  `agreementAvailability: NOT_AVAILABLE`): submit the Anthropic use-case form in the console.
- Local dev reads `aws login` credentials, which needs `botocore[crt]` (dev dependency).

## Static analysis

SonarQube local instance (`~/projects/sonarqube`), key `drug-label-rag`:
`cd api && uv run pytest --cov=api --cov-report=xml && cd .. && bunx @sonar/scan`.
Scan before a release and after big changes.
