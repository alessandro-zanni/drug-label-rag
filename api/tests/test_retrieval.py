from api.embeddings import DIMENSIONS
from api.retrieval import search


def one_hot(i: int) -> list[float]:
    v = [0.0] * DIMENSIONS
    v[i] = 1.0
    return v


def test_hybrid_search_merges_vector_and_full_text(conn):
    conn.execute(
        "insert into labels (set_id, label_id, generic_name, effective_date)"
        " values ('t', 't', 'ZZQTESTDRUG', '2026-01-01')"
    )
    rows = [
        (0, "Zzqtestdrug — Warnings\nzzqtestdrug text only", one_hot(0)),
        (1, "Other — Warnings\nvector only", one_hot(1)),
        (2, "Zzqtestdrug — Dosage\nzzqtestdrug text and vector", one_hot(1)),
    ]
    for ordinal, content, vector in rows:
        conn.execute(
            "insert into chunks (set_id, section, ordinal, content, embedding)"
            " values ('t', 'warnings', %s, %s, %s)",
            (ordinal, content, vector),
        )

    # candidates=2: the vector side keeps the two exact matches, the text side the two
    # chunks naming the drug, so only one chunk is found by both.
    results = search(conn, "zzqtestdrug", one_hot(1), limit=3, candidates=2)

    assert results[0].text == "zzqtestdrug text and vector"
    assert {p.text for p in results[1:]} == {"vector only", "zzqtestdrug text only"}
    assert results[0].title == "Zzqtestdrug — Dosage"
    assert results[0].source.endswith("setid=t")
