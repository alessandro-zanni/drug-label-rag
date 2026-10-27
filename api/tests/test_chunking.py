from api.chunking import chunk_label, split


def test_split_keeps_short_text_whole():
    assert split("One. Two.") == ["One. Two."]


def test_split_respects_max_and_overlaps():
    text = " ".join(f"Sentence number {i}." for i in range(100))
    chunks = split(text, max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # each chunk starts with a sentence from the tail of the previous one
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split(". ")[0] + "." in prev[-50:]


def test_split_hard_wraps_sentences_without_breaks():
    chunks = split("x " * 1000, max_chars=300)
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_label_headers_and_skips_unknown_sections():
    label = {
        "openfda": {"generic_name": ["IBUPROFEN"], "brand_name": ["Advil"]},
        "warnings": ["Warnings Allergy alert."],
        "package_label_principal_display_panel": ["ignored"],
    }
    [chunk] = chunk_label(label)
    assert chunk.section == "warnings"
    assert chunk.content == "Ibuprofen (Advil) — Warnings\nWarnings Allergy alert."
