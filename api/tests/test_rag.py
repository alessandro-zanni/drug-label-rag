from types import SimpleNamespace as NS

from api.rag import NOT_FOUND, answer, stream_answer, to_answer, to_events
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


class FakeStream(list):
    def __init__(self, events, stop_reason="end_turn"):
        super().__init__(events)
        self.stop_reason = stop_reason

    def get_final_message(self):
        return NS(stop_reason=self.stop_reason)


def delta(**kw):
    return NS(type="content_block_delta", delta=NS(**kw))


def test_stream_emits_text_then_sources_and_markers_per_block():
    start, stop = NS(type="content_block_start"), NS(type="content_block_stop")
    stream = FakeStream(
        [
            start,
            delta(type="citations_delta", citation=cite(1, "bleeding")),
            delta(type="text_delta", text="Bleeding"),
            delta(type="text_delta", text=" risk."),
            stop,
            start,
            delta(type="citations_delta", citation=cite(1)),
            delta(type="text_delta", text=" Again."),
            stop,
        ]
    )
    events = list(to_events(stream, [passage(0), passage(1)]))
    assert "".join(e["text"] for e in events if e["type"] == "text") == (
        "Bleeding risk. [1] Again. [1]"
    )
    sources = [e for e in events if e["type"] == "source"]
    assert [(s["number"], s["title"], s["quote"]) for s in sources] == [
        (1, "Drug 1 — Warnings", "bleeding")
    ]


def test_stream_refusal_ends_with_refused_event():
    events = list(to_events(FakeStream([], stop_reason="refusal"), [passage(0)]))
    assert events == [{"type": "refused", "text": NOT_FOUND}]


def test_stream_without_passages_skips_the_model():
    assert list(stream_answer("anything", [])) == [{"type": "text", "text": NOT_FOUND}]
