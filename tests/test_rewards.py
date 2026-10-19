from autoevidence.rewards.cited_reward import cited_reward, score_group
from autoevidence.rewards.judge import RuleJudge, attack_samples
from autoevidence.schemas.enums import Disposal, FaultType
from autoevidence.schemas.models import AnalysisConclusion, AnalysisRecord, ClauseHit
from autoevidence.standards.loader import load_clause_library


def _rec(event, ft, dp, clauses):
    return AnalysisRecord(
        evidence="对话第1轮；信号窗口sw_01",
        clause_check=[ClauseHit(clause_id=c, hit=True, why="x") for c in clauses],
        conclusion=AnalysisConclusion(quality_event=event, fault_type=ft, disposal=dp),
    )


def test_fabricate_and_false_block():
    lib = load_clause_library()
    gold = {"quality_event": False, "fault_type": "无", "disposal": "归档"}
    bad = _rec(True, FaultType.BATTERY, Disposal.TECH, ["Q01-99"])
    r = cited_reward(bad, gold, lib, judge=RuleJudge())
    assert r["r_penalty"] < 0
    assert any("fabricate" in x or x == "false_block" for x in r["reasons"]) or r["r_penalty"] <= -0.4


def test_dodge_critical_via_tech_penalized():
    lib = load_clause_library()
    gold = {"quality_event": True, "fault_type": "制动与转向", "disposal": "紧急复核"}
    dodge = _rec(True, FaultType.BRAKE, Disposal.TECH, ["Q02-03", "Q05-01"])
    r = cited_reward(dodge, gold, lib, judge=RuleJudge())
    assert "miss_critical" in r["reasons"]
    assert r["r_penalty"] <= -0.7


def test_identical_rewards_skipped():
    lib = load_clause_library()
    gold = {"quality_event": False, "fault_type": "无", "disposal": "归档"}
    rec = _rec(False, FaultType.NONE, Disposal.ARCHIVE, ["Q06-02", "Q05-04"])
    g = score_group([rec, rec, rec], gold, lib)
    assert g["skipped"] is True


def test_attack_samples_defined():
    assert len(attack_samples()) >= 4
