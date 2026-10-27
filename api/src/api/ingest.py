"""Load drug labels from openFDA into Postgres, then embed chunks that lack a vector.

uv run python -m api.ingest            # fetch + chunk every drug in data/drugs.txt
uv run python -m api.ingest --embed    # same, then embed missing chunks (needs AWS)
"""

import argparse
from datetime import datetime
from pathlib import Path

import httpx

from api.chunking import chunk_label
from api.db import connect
from api.embeddings import embed_many
from api.openfda import fetch_label

DRUGS = Path(__file__).parents[3] / "data" / "drugs.txt"
EMBED_BATCH = 64


def store(conn, label: dict) -> int:
    """Upsert a label; rewrite its chunks only when openFDA published a new version."""
    openfda = label.get("openfda", {})
    current = conn.execute(
        "select label_id from labels where set_id = %s", (label["set_id"],)
    ).fetchone()
    if current and current[0] == label["id"]:
        return 0

    conn.execute(
        """insert into labels (set_id, label_id, generic_name, brand_name, manufacturer,
                               product_type, effective_date)
           values (%s, %s, %s, %s, %s, %s, %s)
           on conflict (set_id) do update set label_id = excluded.label_id,
             effective_date = excluded.effective_date, fetched_at = now()""",
        (
            label["set_id"],
            label["id"],
            openfda.get("generic_name", ["Unknown"])[0],
            openfda.get("brand_name", [None])[0],
            openfda.get("manufacturer_name", [None])[0],
            openfda.get("product_type", [None])[0],
            datetime.strptime(label["effective_time"], "%Y%m%d").date(),
        ),
    )
    conn.execute("delete from chunks where set_id = %s", (label["set_id"],))
    chunks = chunk_label(label)
    with conn.cursor() as cur:
        cur.executemany(
            "insert into chunks (set_id, section, ordinal, content) values (%s, %s, %s, %s)",
            [(label["set_id"], c.section, c.ordinal, c.content) for c in chunks],
        )
    conn.commit()
    return len(chunks)


def embed_missing(conn) -> int:
    total = 0
    while rows := conn.execute(
        "select id, content from chunks where embedding is null limit %s", (EMBED_BATCH,)
    ).fetchall():
        vectors = embed_many([content for _, content in rows])
        with conn.cursor() as cur:
            cur.executemany(
                "update chunks set embedding = %s where id = %s",
                [(v, id_) for (id_, _), v in zip(rows, vectors, strict=True)],
            )
        conn.commit()
        total += len(rows)
        print(f"embedded {total}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true", help="embed chunks without a vector")
    args = parser.parse_args()

    drugs = [line.strip() for line in DRUGS.read_text().splitlines() if line.strip()]
    with connect() as conn, httpx.Client(timeout=30) as client:
        for drug in drugs:
            label = fetch_label(client, drug)
            if label is None:
                print(f"{drug}: not found")
                continue
            print(f"{drug}: {store(conn, label)} chunks")
        if args.embed:
            embed_missing(conn)


if __name__ == "__main__":
    main()
