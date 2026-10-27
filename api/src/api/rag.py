from dataclasses import dataclass, field

from anthropic import BetaFallbackState

from api import llm
from api.retrieval import Passage

SYSTEM = """You answer questions about medicines using only the FDA drug label excerpts \
provided as search results.

- Base every statement on the excerpts. If they do not answer the question, say that the \
labels you have do not cover it; do not fill gaps from general knowledge.
- You report what the labels say; you do not give medical advice. When a question is about \
someone's own treatment, close by recommending they check with a doctor or pharmacist.
- Answer in the language of the question. Be concise: a short paragraph or a few bullets."""

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
            i = c.search_result_index
            if i not in numbers:
                p = passages[i]
                numbers[i] = Source(len(numbers) + 1, p.title, p.source, c.cited_text)
            parts.append(f" [{numbers[i].number}]")
    return Answer("".join(parts).strip(), list(numbers.values()))


def answer(question: str, passages: list[Passage]) -> Answer:
    if not passages:
        return Answer(NOT_FOUND)
    with BetaFallbackState():
        response = llm.client().beta.messages.create(
            model=llm.MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            system=SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        *[search_result(p) for p in passages],
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
    return to_answer(response, passages)
