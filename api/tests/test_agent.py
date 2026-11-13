from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
from anthropic.lib.tools import ToolError

from api import agent
from api.rag import NOT_FOUND


@pytest.fixture
def tools(conn):
    for set_id, name in [("a", "ZZQALPHA"), ("b", "ZZQBETA")]:
        conn.execute(
            "insert into labels (set_id, label_id, generic_name, effective_date)"
            " values (%s, %s, %s, '2026-01-01')",
            (set_id, set_id, name),
        )
        for ordinal in range(2):
            conn.execute(
                "insert into chunks (set_id, section, ordinal, content)"
                " values (%s, 'warnings', %s, %s)",
                (set_id, ordinal, f"{name} — Warnings\n{name} part {ordinal}"),
            )
    search_labels, get_section, compare_drugs = agent.tools(
        NS(connection=lambda: nullcontext(conn))
    )
    return NS(get_section=get_section, compare_drugs=compare_drugs)


def test_get_section_returns_every_chunk_as_a_citable_search_result(tools):
    results = tools.get_section.call({"drug": "zzqalpha", "section": "warnings"})
    assert [r["content"][0]["text"] for r in results] == ["ZZQALPHA part 0", "ZZQALPHA part 1"]
    assert results[0]["citations"] == {"enabled": True}
    assert results[0]["source"].endswith("setid=a")


def test_unknown_drug_or_section_is_a_tool_error(tools):
    with pytest.raises(ToolError):
        tools.get_section.call({"drug": "nope", "section": "warnings"})
    with pytest.raises(ToolError):
        tools.get_section.call({"drug": "zzqalpha", "section": "overdosage"})


def test_compare_drugs_flags_missing_drugs(tools):
    content = tools.compare_drugs.call({"drugs": ["zzqalpha", "nope"], "section": "warnings"})
    assert [b["type"] for b in content] == ["search_result", "search_result", "text"]
    assert "nope" in content[-1]["text"]


def cite(source: str, title: str, quote: str = "q"):
    return NS(source=source, title=title, cited_text=quote)


def test_sources_are_numbered_by_url_and_title_across_tool_results():
    response = NS(
        stop_reason="end_turn",
        content=[
            NS(type="text", text="A bleeds.", citations=[cite("u/a", "A — Warnings", "bleed")]),
            NS(type="text", text=" B too.", citations=[cite("u/b", "B — Warnings")]),
            # Another chunk of A's warnings: same source, same number.
            NS(type="text", text=" A again.", citations=[cite("u/a", "A — Warnings", "x")]),
        ],
    )
    result = agent.to_answer(response)
    assert result.text == "A bleeds. [1] B too. [2] A again. [1]"
    assert [(s.number, s.url, s.quote) for s in result.sources] == [
        (1, "u/a", "bleed"),
        (2, "u/b", "q"),
    ]


def test_iteration_cap_without_final_answer_is_not_found():
    response = NS(
        stop_reason="tool_use", content=[NS(type="text", text="Let me check", citations=None)]
    )
    assert agent.to_answer(response).text == NOT_FOUND
