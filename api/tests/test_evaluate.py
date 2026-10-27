from api.evaluate import hit, summarize
from api.retrieval import Passage


def p(drug: str, section: str) -> Passage:
    return Passage(1, "s", drug, None, section, "T\nx", 1.0)


def test_hit_needs_drug_and_section_in_the_same_passage():
    case = {"drugs": ["WARFARIN"], "sections": ["drug_interactions"]}
    assert hit(case, [p("ASPIRIN", "warnings"), p("WARFARIN", "drug_interactions")])
    assert not hit(case, [p("WARFARIN", "warnings"), p("ASPIRIN", "drug_interactions")])


def test_summarize_skips_errored_and_truncated_rows():
    rows = [
        {"status": "ok", "grade": {"faithful": 1.0}},
        {"status": "ok", "grade": {"faithful": 0.0}},
        {"status": "error", "grade": {"faithful": 0.0}},
        {"status": "truncated", "grade": {}},
    ]
    assert summarize(rows) == {"faithful": 0.5}
