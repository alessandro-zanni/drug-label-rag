from fastapi import FastAPI

app = FastAPI(title="Drug Label RAG")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
