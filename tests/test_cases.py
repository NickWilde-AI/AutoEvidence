from autoevidence.evaluation.cases import build_e2e_cases


def test_e2e_cases_cover_required_types():
    cases = build_e2e_cases()
    names = {c["case_name"] for c in cases}
    assert "normal" in names or "hard_negative" in names
    assert "quality_event" in names or "safety_to_human" in names
    assert "insufficient_evidence" in names
    for c in cases:
        assert "gold" not in c["model_output"]
        assert "routing" in c
        assert "evidence_pack" in c
