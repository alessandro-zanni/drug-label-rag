from types import SimpleNamespace as NS

from api.rag import NOT_FOUND, answer, to_answer
from api.retrieval import Passage


def passage(i: int) -> Passage:
    return Passage(i, f"set{i}", "X", None, "warnings", f"Drug {i} — Warnings\ntext {i}", 1.0)


def cite(i: int, quote: str = "q"):
    return NS(search_result_index=i, cited_text=quote)


def test_citations_become_numbered_sources_in_first_use_order():
    response = NS(
        stop_reason="end_turn",
        content=[
            NS(type="thinking"),
            NS(type="text", text="Bleeding risk.", citations=[cite(1, "bleeding")]),
            NS(type="text", text=" Also ulcers.", citations=[cite(0), cite(1)]),
            NS(type="text", text=" Ask a doctor.", citations=None),
        ],
    )
    result = to_answer(response, [passage(0), passage(1)])
    assert result.text == "Bleeding risk. [1] Also ulcers. [2] [1] Ask a doctor."
    assert [(s.number, s.title, s.quote) for s in result.sources] == [
        (1, "Drug 1 — Warnings", "bleeding"),
        (2, "Drug 0 — Warnings", "q"),
    ]


def test_refusal_returns_no_sources():
    result = to_answer(NS(stop_reason="refusal", content=[]), [passage(0)])
    assert result.refused and result.sources == []


def test_no_passages_skips_the_model():
    assert answer("anything", []).text == NOT_FOUND
