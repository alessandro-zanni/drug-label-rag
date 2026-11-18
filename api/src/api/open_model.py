"""Open-weight model on Bedrock, for the eval comparison only (phase 4b); the API stays on Claude.

No native citations here: passages are numbered in the prompt and the model writes [n] markers,
which are parsed back into sources. Claude gets citations from the API instead (rag.py).
"""

import os
import re
from functools import cache

import boto3

from api import rag
from api.retrieval import Passage

# Strongest open-weight model served in-region (eu-central-1), so data stays in Frankfurt.
MODEL = "qwen.qwen3-235b-a22b-2507-v1:0"

SYSTEM = (
    rag.SYSTEM
    + """
- Cite every statement with the number of the excerpt it comes from, like [2], right after the \
sentence. Cite only excerpt numbers you were given."""
)


@cache
def _client():
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "eu-central-1"))


def generate(question: str, passages: list[Passage]) -> dict:
    numbered = "\n\n".join(f"[{i + 1}] {p.title}\n{p.text}" for i, p in enumerate(passages))
    prompt = f"<excerpts>\n{numbered}\n</excerpts>\n\n{question}"
    return _client().converse(
        modelId=MODEL,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 8000},
    )


def to_answer(response: dict, passages: list[Passage]) -> rag.Answer:
    """Renumber [n] markers by first use, as rag.to_answer does; drop out-of-range ones."""
    blocks = response["output"]["message"]["content"]
    text = "".join(b.get("text", "") for b in blocks).strip()  # skips reasoning blocks
    numbers: dict[int, rag.Source] = {}

    def renumber(m: re.Match) -> str:
        i = int(m.group(1)) - 1
        if not 0 <= i < len(passages):
            return ""
        if i not in numbers:
            p = passages[i]
            numbers[i] = rag.Source(len(numbers) + 1, p.title, p.source, "")
        return f" [{numbers[i].number}]"

    text = re.sub(r"\s*\[(\d+)\]", renumber, text)
    return rag.Answer(text, list(numbers.values()))
