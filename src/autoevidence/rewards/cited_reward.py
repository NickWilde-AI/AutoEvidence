"""引证奖励：分项得分、惩罚与组级汇总。"""

from __future__ import annotations

from typing import Any

from autoevidence.rewards.config import DEFAULT_REWARD, RewardConfig
from autoevidence.rewards.judge import JudgeCache, JudgeClient, RuleJudge, citation_support, dynamic_checks
from autoevidence.schemas.models import AnalysisRecord
from autoevidence.standards.loader import ClauseLibrary


def cited_reward(
    record: AnalysisRecord | dict,
    gold: dict,
    lib: ClauseLibrary,
    judge: JudgeClient | None = None,
    cfg: RewardConfig | None = None,
    cache: JudgeCache | None = None,
    shared_checks: list[dict] | None = None,
) -> dict[str, Any]:
    cfg = cfg or DEFAULT_REWARD
    judge = judge or RuleJudge()
    if isinstance(record, dict):
        record = AnalysisRecord.model_validate(record)

    c = record.conclusion
    r: dict[str, Any] = {
        "r_verdict": 0.0,
        "r_fault_type": 0.0,
        "r_disposal": 0.0,
        "r_citation": 0.0,
        "r_penalty": 0.0,
        "r_judge": 0.0,
        "checks": [],
        "judge_version": getattr(judge, "version", "unknown"),
        "format_ok": True,
        "reasons": [],
    }

    # 格式/缺失
    if not record.clause_check:
        r["format_ok"] = False
        r["reasons"].append("missing_clause_check")
        r["r_penalty"] += -0.3

    g_event = gold["quality_event"]
    g_ft = gold["fault_type"]
    g_dp = gold["disposal"]
    if hasattr(g_ft, "value"):
        g_ft = g_ft.value
    if hasattr(g_dp, "value"):
        g_dp = g_dp.value

    pred_ft = c.fault_type.value
    pred_dp = c.disposal.value

    if c.quality_event == g_event:
        r["r_verdict"] = cfg.verdict
        if g_event and pred_ft == g_ft:
            r["r_fault_type"] = cfg.fault_type
    elif c.quality_event and not g_event:
        r["r_penalty"] += cfg.false_block
        r["reasons"].append("false_block")

    if pred_dp == g_dp:
        r["r_disposal"] = cfg.disposal
    elif g_dp == "紧急复核" and pred_dp in ("归档", "技术排查"):
        # 堵住「判成技术排查躲避安全漏放扣分」
        r["r_penalty"] += cfg.miss_critical
        r["reasons"].append("miss_critical")
        if pred_dp == "技术排查":
            r["r_penalty"] += 0.0  # miss_critical 已覆盖 dodge；显式记录
            r["reasons"].append("dodge_critical_via_tech")

    cited = [x.clause_id for x in record.clause_check]
    illegal = [cid for cid in cited if cid not in lib.clauses]
    if illegal:
        r["r_penalty"] += cfg.fabricate
        r["reasons"].append(f"fabricate:{','.join(illegal)}")
    else:
        support, support_meta = citation_support(record, lib, judge, cache=cache)
        if support_meta.get("failed"):
            r["reasons"].append("judge_citation_failed")
        elif support > 0:
            r["r_citation"] = cfg.citation

    judge_score, checks, judge_meta = dynamic_checks(
        record, {"quality_event": g_event, "fault_type": g_ft, "disposal": g_dp}, lib, judge, shared_checks
    )
    if judge_meta.get("failed"):
        r["r_judge"] = 0.0
        r["reasons"].append("judge_checks_failed")
    else:
        r["r_judge"] = min(judge_score, cfg.judge_max)
    r["checks"] = checks
    r["judge_meta"] = judge_meta

    r["total"] = (
        r["r_verdict"]
        + r["r_fault_type"]
        + r["r_disposal"]
        + r["r_citation"]
        + r["r_penalty"]
        + r["r_judge"]
    )
    return r


def score_group(
    records: list[AnalysisRecord | dict],
    gold: dict,
    lib: ClauseLibrary,
    judge: JudgeClient | None = None,
    cfg: RewardConfig | None = None,
) -> dict[str, Any]:
    """同组共用核查项；奖励全同则 skipped。"""
    judge = judge or RuleJudge()
    # 先用第一条拟定 checks
    first = records[0] if not isinstance(records[0], dict) else AnalysisRecord.model_validate(records[0])
    _, shared, meta = dynamic_checks(
        first,
        {
            "quality_event": gold["quality_event"],
            "fault_type": gold["fault_type"].value if hasattr(gold["fault_type"], "value") else gold["fault_type"],
            "disposal": gold["disposal"].value if hasattr(gold["disposal"], "value") else gold["disposal"],
        },
        lib,
        judge,
    )
    if meta.get("failed"):
        shared = [
            {"name": "fallback_event_match", "weight": 1.0, "score": 0.0},
        ]
    rewards = [
        cited_reward(rec, gold, lib, judge=judge, cfg=cfg, shared_checks=shared) for rec in records
    ]
    totals = [x["total"] for x in rewards]
    skipped = len(set(round(t, 6) for t in totals)) <= 1
    return {
        "rewards": rewards,
        "totals": totals,
        "skipped": skipped,
        "shared_checks": shared,
        "judge_version": getattr(judge, "version", "unknown"),
        "valid_ratio": sum(1 for x in rewards if x.get("format_ok", True)) / max(len(rewards), 1),
    }
