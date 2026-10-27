from dataclasses import dataclass

import psycopg

# Hybrid search: vector similarity catches paraphrases ("blood thinner"), full-text catches
# exact drug names and terms. The two rankings are merged with reciprocal rank fusion.
HYBRID_SQL = """
with vec as (
  select id, row_number() over (order by embedding <=> %(vector)s::vector) as rank
  from chunks
  where embedding is not null
  order by embedding <=> %(vector)s::vector
  limit %(candidates)s
), fts as (
  select id, row_number() over (order by ts_rank_cd(tsv, query) desc) as rank
  from chunks, websearch_to_tsquery('english', %(text)s) as query
  where tsv @@ query
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
) -> list[Passage]:
    rows = conn.execute(
        HYBRID_SQL,
        {
            "text": text,
            "vector": vector,
            "limit": limit,
            "candidates": candidates,
            "rrf_k": rrf_k,
        },
    ).fetchall()
    return [Passage(*row) for row in rows]
