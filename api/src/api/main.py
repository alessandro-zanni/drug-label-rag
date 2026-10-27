from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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


@app.post("/ask")
def ask(body: Question, request: Request) -> rag.Answer:
    with request.app.state.pool.connection() as conn:
        passages = retrieval.search(conn, body.question, embed(body.question))
    return rag.answer(body.question, passages)
