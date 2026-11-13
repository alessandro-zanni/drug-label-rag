from dataclasses import dataclass

import psycopg

# Hybrid search: vector similarity catches paraphrases ("blood thinner"), full-text catches
# exact drug names and terms. The two rankings are merged with reciprocal rank fusion.
HYBRID_SQL = """
with drug as (
  select set_id from labels
  where %(drug)s::text is null or generic_name ilike %(drug)s or brand_name ilike %(drug)s
), vec as (
  select id, row_number() over (order by embedding <=> %(vector)s::vector) as rank
  from chunks
  where embedding is not null and set_id in (select set_id from drug)
  order by embedding <=> %(vector)s::vector
  limit %(candidates)s
), fts as (
  select id, row_number() over (order by ts_rank_cd(tsv, query) desc) as rank
  from chunks, websearch_to_tsquery('english', %(text)s) as query
  where tsv @@ query and set_id in (select set_id from drug)
  order by ts_rank_cd(tsv, query) desc
  limit %(candidates)s
)
select c.id, l.set_id, l.generic_name, l.brand_name, c.section, c.content,
       coalesce(1.0 / (%(rrf_k)s + vec.rank), 0) + coalesce(1.0 / (%(rrf_k)s + fts.rank), 0)
         as score
from vec
full outer join fts using (id)
join chunks c using (id)
join labels l using (set_id)
order by score desc
limit %(limit)s
"""


@dataclass
class Passage:
    id: int
    set_id: str
    generic_name: str
    brand_name: str | None
    section: str
    content: str
    score: float

    @property
    def title(self) -> str:
        # First line of content is the "Drug (Brand) — Section" header written at ingest.
        return self.content.split("\n", 1)[0]

    @property
    def text(self) -> str:
        return self.content.split("\n", 1)[-1]

    @property
    def source(self) -> str:
        return f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={self.set_id}"


def search(
    conn: psycopg.Connection,
    text: str,
    vector: list[float],
    limit: int = 8,
    candidates: int = 40,
    rrf_k: int = 60,
    drug: str | None = None,
) -> list[Passage]:
    # ponytail: the drug filter runs after the HNSW scan, so with a filter the vector side can
    # return fewer than `candidates`; full-text still finds the drug. Turn on pgvector's
    # hnsw.iterative_scan if filtered recall turns out low.
    rows = conn.execute(
        HYBRID_SQL,
        {
            "text": text,
            "vector": vector,
            "limit": limit,
            "candidates": candidates,
            "rrf_k": rrf_k,
            "drug": drug,
        },
    ).fetchall()
    return [Passage(*row) for row in rows]
