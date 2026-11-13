import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api import db, rag, retrieval
from api.embeddings import embed


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = db.pool()
    yield
    app.state.pool.close()


app = FastAPI(title="Drug Label RAG", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def passages(question: str, request: Request) -> list[retrieval.Passage]:
    with request.app.state.pool.connection() as conn:
        return retrieval.search(conn, question, embed(question))


@app.post("/ask")
def ask(body: Question, request: Request) -> rag.Answer:
    return rag.answer(body.question, passages(body.question, request))


@app.post("/ask/stream")
def ask_stream(body: Question, request: Request) -> StreamingResponse:
    """Server-sent events: one JSON object per `data:` line (see rag.to_events)."""
    events = rag.stream_answer(body.question, passages(body.question, request))
    return StreamingResponse(
        (f"data: {json.dumps(e)}\n\n" for e in events), media_type="text/event-stream"
    )
