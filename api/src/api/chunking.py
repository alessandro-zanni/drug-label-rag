import re
import textwrap
from dataclasses import dataclass

from api.openfda import SECTIONS

MAX_CHARS = 1500
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    section: str
    ordinal: int
    content: str


def split(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Pack sentences into windows of at most max_chars, repeating the last ~overlap chars."""
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        # Flattened tables have no sentence breaks: hard-wrap them.
        sentences += textwrap.wrap(s, max_chars) if len(s) > max_chars else [s]

    chunks, current = [], []
    for s in sentences:
        if current and len(" ".join([*current, s])) > max_chars:
            chunks.append(" ".join(current))
            tail = []
            for prev in reversed(current):
                if len(" ".join([prev, *tail])) > overlap:
                    break
                tail.insert(0, prev)
            current = tail
        current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


def label_name(label: dict) -> str:
    openfda = label.get("openfda", {})
    generic = openfda.get("generic_name", ["Unknown"])[0].title()
    brand = openfda.get("brand_name", [None])[0]
    return f"{generic} ({brand})" if brand and brand.lower() != generic.lower() else generic


def chunk_label(label: dict) -> list[Chunk]:
    name = label_name(label)
    chunks = []
    for section in SECTIONS:
        text = " ".join(label.get(section, []))
        if not text.strip():
            continue
        title = section.replace("_", " ").capitalize()
        for i, part in enumerate(split(text)):
            chunks.append(Chunk(section, i, f"{name} — {title}\n{part}"))
    return chunks
