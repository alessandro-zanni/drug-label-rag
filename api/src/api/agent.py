from typing import Literal

from anthropic import BetaFallbackState, beta_tool
from anthropic.lib.tools import ToolError

from api import llm, openfda, retrieval
from api.embeddings import embed
from api.rag import NOT_FOUND, Answer, Source, search_result
from api.rag import SYSTEM as RAG_SYSTEM
from api.retrieval import Passage

MAX_ITERATIONS = 6

SYSTEM = (
    RAG_SYSTEM
    + """
- The search results come from your tools. Search before answering; use get_section to read \
a whole section and compare_drugs to put the same section of several drugs side by side."""
)

Section = Literal[openfda.SECTIONS]

SECTION_SQL = """
select c.id, l.set_id, l.generic_name, l.brand_name, c.section, c.content, 1.0
from chunks c join labels l using (set_id)
where (l.generic_name ilike %(drug)s or l.brand_name ilike %(drug)s) and c.section = %(section)s
order by c.set_id, c.ordinal
"""


def section_passages(conn, drug: str, section: str) -> list[Passage]:
    rows = conn.execute(SECTION_SQL, {"drug": drug, "section": section}).fetchall()
    return [Passage(*row) for row in rows]


def tools(pool) -> list:
    """Tools bound to a pool; each call borrows a connection only while it queries."""

    @beta_tool
    def search_labels(query: str, drug: str | None = None) -> list[dict]:
        """Hybrid search over FDA drug label sections; returns the best matching passages.

        Args:
            query: What to look for, in plain words.
            drug: Generic or brand name to restrict the search to one drug's label.
        """
        vector = embed(query)
        with pool.connection() as conn:
            passages = retrieval.search(conn, query, vector, drug=drug)
        if not passages:
            raise ToolError(f"No label passages found for drug {drug!r}." if drug else "No match.")
        return [search_result(p) for p in passages]

    @beta_tool
    def get_section(drug: str, section: Section) -> list[dict]:
        """Full text of one section of a drug's label.

        Args:
            drug: Generic or brand name.
            section: Label section.
        """
        with pool.connection() as conn:
            passages = section_passages(conn, drug, section)
        if not passages:
            raise ToolError(f"No {section} section in the label for {drug!r}, or no such drug.")
        return [search_result(p) for p in passages]

    @beta_tool
    def compare_drugs(drugs: list[str], section: Section) -> list[dict]:
        """The same label section for 2 to 4 drugs, to compare them.

        Args:
            drugs: 2 to 4 generic or brand names.
            section: Label section.
        """
        if not 2 <= len(drugs) <= 4:
            raise ToolError("Pass 2 to 4 drugs.")
        content, missing = [], []
        with pool.connection() as conn:
            for drug in drugs:
                passages = section_passages(conn, drug, section)
                content += [search_result(p) for p in passages]
                if not passages:
                    missing.append(drug)
        if not content:
            raise ToolError(f"No {section} section found for any of {drugs}.")
        if missing:
            content.append({"type": "text", "text": f"No {section} section for: {missing}."})
        return content

    return [search_labels, get_section, compare_drugs]


def to_answer(response) -> Answer:
    """Like rag.to_answer, but citations span several tool results: a search_result_index
    counts results across the whole conversation, so number sources by (url, title) instead."""
    if response.stop_reason == "refusal":
        return Answer(NOT_FOUND, refused=True)
    if response.stop_reason == "tool_use":  # MAX_ITERATIONS hit before a final answer
        return Answer(NOT_FOUND)

    numbers: dict[tuple[str, str], Source] = {}
    parts = []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for c in block.citations or []:
            key = (c.source, c.title)
            if key not in numbers:
                numbers[key] = Source(len(numbers) + 1, c.title, c.source, c.cited_text)
            parts.append(f" [{numbers[key].number}]")
    return Answer("".join(parts).strip(), list(numbers.values()))


def answer(question: str, pool) -> Answer:
    with BetaFallbackState():
        runner = llm.client().beta.messages.tool_runner(
            model=llm.MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            system=SYSTEM,
            tools=tools(pool),
            messages=[{"role": "user", "content": question}],
            max_iterations=MAX_ITERATIONS,
        )
        return to_answer(runner.until_done())
