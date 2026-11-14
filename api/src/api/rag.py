from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from anthropic import BetaFallbackState

from api import llm
from api.retrieval import Passage

SYSTEM = """You answer questions about medicines using only the FDA drug label excerpts \
provided as search results.

- Base every statement on the excerpts. If they do not answer the question, say that the \
labels you have do not cover it; do not fill gaps from general knowledge.
- Do not explain terms, abbreviations or mechanisms the excerpts do not explain, and do not \
name sources or resources the excerpts do not mention (no "the FDA label says", no websites). \
When the labels do not cover a question, say so and stop.
- You report what the labels say; you do not give medical advice. When a question is about \
someone's own treatment, close by recommending they check with a doctor or pharmacist.
- Answer entirely in the language of the question: translate quoted label text too, never \
mix languages. Be concise: a short paragraph or a few bullets."""

NOT_FOUND = "I couldn't find anything about this in the drug labels I have."


@dataclass
class Source:
    number: int
    title: str
    url: str
    quote: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    refused: bool = False


def search_result(p: Passage) -> dict:
    return {
        "type": "search_result",
        "source": p.source,
        "title": p.title,
        "content": [{"type": "text", "text": p.text}],
        "citations": {"enabled": True},
    }


def number(numbers: dict[int, Source], citation, passages: list[Passage]) -> Source | None:
    """Source for a citation, numbered by first use; returns it only the first time."""
    i = citation.search_result_index
    if i in numbers:
        return None
    p = passages[i]
    numbers[i] = Source(len(numbers) + 1, p.title, p.source, citation.cited_text)
    return numbers[i]


def to_answer(response, passages: list[Passage]) -> Answer:
    """Join text blocks, turning each citation into a [n] marker numbered by first use."""
    if response.stop_reason == "refusal":
        return Answer(NOT_FOUND, refused=True)

    numbers: dict[int, Source] = {}
    parts = []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for c in block.citations or []:
            number(numbers, c, passages)
            parts.append(f" [{numbers[c.search_result_index].number}]")
    return Answer("".join(parts).strip(), list(numbers.values()))


def request(question: str, passages: list[Passage]) -> dict:
    return {
        "model": llm.MODEL,
        "max_tokens": 16000,
        "output_config": {"effort": "medium"},
        "system": SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": [
                    *[search_result(p) for p in passages],
                    {"type": "text", "text": question},
                ],
            }
        ],
    }


def generate(question: str, passages: list[Passage]):
    """Raw model response; callers that only need the answer use answer()."""
    with BetaFallbackState():
        return llm.client().beta.messages.create(**request(question, passages))


def answer(question: str, passages: list[Passage]) -> Answer:
    if not passages:
        return Answer(NOT_FOUND)
    return to_answer(generate(question, passages), passages)


def to_events(stream, passages: list[Passage]) -> Iterator[dict]:
    """Stream events as dicts: text deltas, a source the first time it is cited, and a final
    refused event that tells the client to replace whatever text it showed."""
    numbers: dict[int, Source] = {}
    citations = []
    for event in stream:
        if event.type == "content_block_start":
            citations = []
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                yield {"type": "text", "text": event.delta.text}
            elif event.delta.type == "citations_delta":
                citations.append(event.delta.citation)
        elif event.type == "content_block_stop":
            # Markers go after the cited block, as in to_answer().
            for c in citations:
                if source := number(numbers, c, passages):
                    yield {"type": "source", **asdict(source)}
                yield {"type": "text", "text": f" [{numbers[c.search_result_index].number}]"}
    if stream.get_final_message().stop_reason == "refusal":
        yield {"type": "refused", "text": NOT_FOUND}


def stream_answer(question: str, passages: list[Passage]) -> Iterator[dict]:
    if not passages:
        yield {"type": "text", "text": NOT_FOUND}
        return
    with BetaFallbackState(), llm.client().beta.messages.stream(**request(question, passages)) as s:
        yield from to_events(s, passages)
