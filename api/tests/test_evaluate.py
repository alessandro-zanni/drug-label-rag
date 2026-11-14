from api.evaluate import FactCheck, Verdict, grade_checks, hit, strip_citations, summarize
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


def verdict(covered: list[bool], unsupported=(), abstained=False) -> Verdict:
    facts = [FactCheck(fact=str(i), covered=c) for i, c in enumerate(covered)]
    return Verdict(
        facts=facts, unsupported_claims=list(unsupported), abstained=abstained, explanation=""
    )


def test_grade_checks_scores_a_good_and_a_blind_judge():
    good = (verdict([True, True]), verdict([False, False], abstained=True), verdict([False], ["x"]))
    blind = (verdict([True, False]), verdict([True, True]), verdict([False]))
    assert grade_checks([good]) == {
        "oracle_recall": 1.0,
        "null_recall": 0.0,
        "null_abstained": 1.0,
        "fabricated_flagged": 1.0,
    }
    assert grade_checks([blind]) == {
        "oracle_recall": 0.5,
        "null_recall": 1.0,
        "null_abstained": 0.0,
        "fabricated_flagged": 0.0,
    }


def test_strip_citations_removes_markers_but_not_other_brackets():
    assert strip_citations("Avoid it [1]. Also [2][3] see x[a].") == "Avoid it. Also see x[a]."
