import json
import logging
from types import SimpleNamespace as NS

import psycopg
import pytest
from fastapi.testclient import TestClient

from api import agent, db, main, rag
from api.rag import DISCLAIMER, Answer, Source


@pytest.fixture
def client(monkeypatch):
    """App with its real pool; retrieval and models are stubbed. Audit rows it writes are
    committed, so they are deleted afterwards by request id."""
    main.hits.clear()
    monkeypatch.setattr(main, "passages", lambda q, r: ["passage"])
    monkeypatch.setattr(rag, "answer", lambda q, p: Answer("A [1]", [Source(1, "T", "u", "q")]))
    monkeypatch.setattr(agent, "answer", lambda q, pool: Answer(rag.NOT_FOUND, refused=True))
    ids = []
    with TestClient(main.app) as c:
        yield c, ids
    with psycopg.connect(db.DATABASE_URL) as conn:
        conn.execute("delete from audit_log where request_id = any(%s)", (ids,))


def audit_row(request_id: str):
    with psycopg.connect(db.DATABASE_URL) as conn:
        return conn.execute(
            "select endpoint, question, sources, refused from audit_log where request_id = %s",
            (request_id,),
        ).fetchone()


def test_ask_returns_disclaimer_and_writes_audit_row(client):
    c, ids = client
    r = c.post("/ask", json={"question": "Ibuprofen and warfarin?"})
    ids.append(r.headers["X-Request-ID"])
    assert r.json()["disclaimer"] == DISCLAIMER
    assert audit_row(ids[-1]) == ("/ask", "Ibuprofen and warfarin?", ["T"], False)


def test_agent_audits_refusals(client):
    c, ids = client
    r = c.post("/agent", json={"question": "Max dose?"})
    ids.append(r.headers["X-Request-ID"])
    assert audit_row(ids[-1]) == ("/agent", "Max dose?", [], True)


def test_stream_ends_with_disclaimer_and_audits_at_the_end(client, monkeypatch):
    c, ids = client
    events = [{"type": "source", "title": "T"}, {"type": "disclaimer", "text": DISCLAIMER}]
    monkeypatch.setattr(rag, "stream_answer", lambda q, p: iter(events))
    r = c.post("/ask/stream", json={"question": "Streamed?"})
    ids.append(r.headers["X-Request-ID"])
    assert r.text.strip().endswith(json.dumps(events[-1]))
    assert audit_row(ids[-1]) == ("/ask/stream", "Streamed?", ["T"], False)


def test_audit_purges_rows_past_retention(client):
    c, ids = client
    ids.append("old-row")
    with psycopg.connect(db.DATABASE_URL) as conn:
        conn.execute(
            "insert into audit_log (created_at, request_id, endpoint, question, sources,"
            " refused, latency_ms) values (now() - interval '31 days', 'old-row', '/ask', 'q',"
            " '[]', false, 1)"
        )
    ids.append(c.post("/ask", json={"question": "New one"}).headers["X-Request-ID"])
    assert audit_row("old-row") is None


def test_rate_limit_answers_429_with_retry_after(client, monkeypatch):
    c, ids = client
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    codes = []
    for _ in range(3):
        r = c.post("/ask", json={"question": "Again?"})
        ids.append(r.headers["X-Request-ID"])
        codes.append(r.status_code)
    assert codes == [200, 200, 429]
    assert 0 < int(r.headers["Retry-After"]) <= 60
    assert c.get("/health").status_code == 200  # GETs are not limited


def test_log_line_is_json_with_cloudwatch_metrics():
    record = logging.LogRecord("api", logging.INFO, "", 0, "request", None, None)
    record.fields = {"request_id": "x"} | main.metrics_line("/ask", 12, refused=True)
    line = json.loads(main.JsonFormatter().format(record))
    assert line["Latency"] == 12 and line["Refused"] == 1 and line["path"] == "/ask"
    metrics = line["_aws"]["CloudWatchMetrics"][0]["Metrics"]
    assert {m["Name"] for m in metrics} == {"Latency", "Requests", "Refused"}


def test_client_net_truncates_ip():
    request = NS(client=NS(host="203.0.113.77"))
    assert str(main.client_net(request)) == "203.0.113.0/24"
    assert main.client_net(NS(client=NS(host="testclient"))) is None
