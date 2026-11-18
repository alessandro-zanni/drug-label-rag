import ipaddress
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from api import agent, db, rag, retrieval
from api.embeddings import embed

RATE_LIMIT = 10  # POST requests per client IP per minute
AUDIT_RETENTION = "30 days"


class JsonFormatter(logging.Formatter):
    """One JSON object per line; `extra={"fields": {...}}` adds top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        line = {"level": record.levelname, "message": record.getMessage()}
        return json.dumps(line | getattr(record, "fields", {}))


log = logging.getLogger("api")
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = db.pool()
    yield
    app.state.pool.close()


app = FastAPI(title="Drug Label RAG", lifespan=lifespan)

# ponytail: fixed window counted per process, so each ECS task allows RATE_LIMIT on its own;
# move to a WAF rate-based rule on the ALB in phase 6.
hits: dict[str, int] = {}
window = 0


def allow(ip: str) -> bool:
    global window
    if (now := int(time.time() // 60)) != window:
        window = now
        hits.clear()
    hits[ip] = hits.get(ip, 0) + 1
    return hits[ip] <= RATE_LIMIT


def metrics_line(path: str, latency_ms: int, refused: bool | None) -> dict:
    """CloudWatch Embedded Metric Format: on ECS the log line itself becomes metrics."""
    values = {"Latency": latency_ms, "Requests": 1}
    if refused is not None:
        values["Refused"] = int(refused)
    units = {"Latency": "Milliseconds"}
    return values | {
        "path": path,
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "DrugLabelRag",
                    "Dimensions": [["path"]],
                    "Metrics": [{"Name": m, "Unit": units.get(m, "Count")} for m in values],
                }
            ],
        },
    }


@app.middleware("http")
async def observe(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex
    start = time.monotonic()
    ip = request.client.host if request.client else "unknown"
    if request.method == "POST" and not allow(ip):
        retry = str(60 - int(time.time()) % 60)
        response = JSONResponse(
            {"detail": "Too many requests"}, status_code=429, headers={"Retry-After": retry}
        )
    else:
        response = await call_next(request)
    # Streaming responses: latency is time to headers; the audit row has the full duration.
    latency_ms = int((time.monotonic() - start) * 1000)
    response.headers["X-Request-ID"] = request.state.request_id
    route = request.scope.get("route")
    path = route.path if route else "unmatched"  # raw paths would explode metric cardinality
    fields = {
        "request_id": request.state.request_id,
        "method": request.method,
        "status": response.status_code,
    } | metrics_line(path, latency_ms, getattr(request.state, "refused", None))
    log.info("request", extra={"fields": fields})
    return response


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def client_net(request: Request):
    """Client IP truncated to /24 (IPv4) or /48 (IPv6); None when it is not an IP."""
    try:
        ip = ipaddress.ip_address(request.client.host)
    except (AttributeError, ValueError):
        return None
    return ipaddress.ip_network(f"{ip}/{24 if ip.version == 4 else 48}", strict=False)


def audit(request: Request, endpoint: str, question: str, sources, refused, start) -> None:
    with request.app.state.pool.connection() as conn:
        conn.execute(
            "insert into audit_log"
            " (request_id, endpoint, question, sources, refused, latency_ms, client_net)"
            " values (%s, %s, %s, %s, %s, %s, %s)",
            (
                request.state.request_id,
                endpoint,
                question,
                Jsonb(sources),
                refused,
                int((time.monotonic() - start) * 1000),
                client_net(request),
            ),
        )
        # ponytail: purge on every insert (cheap with the created_at index); move to pg_cron
        # on RDS if traffic grows.
        conn.execute(
            "delete from audit_log where created_at < now() - %s::interval", (AUDIT_RETENTION,)
        )


def audited(endpoint: str, body: Question, request: Request, answer: rag.Answer, start):
    request.state.refused = answer.refused
    titles = [s.title for s in answer.sources]
    audit(request, endpoint, body.question, titles, answer.refused, start)
    return answer


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def passages(question: str, request: Request) -> list[retrieval.Passage]:
    with request.app.state.pool.connection() as conn:
        return retrieval.search(conn, question, embed(question))


@app.post("/ask")
def ask(body: Question, request: Request) -> rag.Answer:
    start = time.monotonic()
    answer = rag.answer(body.question, passages(body.question, request))
    return audited("/ask", body, request, answer, start)


@app.post("/ask/stream")
def ask_stream(body: Question, request: Request) -> StreamingResponse:
    """Server-sent events: one JSON object per `data:` line (see rag.to_events)."""
    start = time.monotonic()
    found = passages(body.question, request)

    def events():
        titles, refused = [], False
        for e in rag.stream_answer(body.question, found):
            if e["type"] == "source":
                titles.append(e["title"])
            refused = refused or e["type"] == "refused"
            yield f"data: {json.dumps(e)}\n\n"
        audit(request, "/ask/stream", body.question, titles, refused, start)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/agent")
def ask_agent(body: Question, request: Request) -> rag.Answer:
    start = time.monotonic()
    answer = agent.answer(body.question, request.app.state.pool)
    return audited("/agent", body, request, answer, start)
