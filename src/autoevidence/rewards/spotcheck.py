"""奖励人工抽查：高奖励是否确实优于低奖励（机制方差 ≠ 质量优劣）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoevidence.rewards.cited_reward import cited_reward
from autoevidence.rewards.judge import RuleJudge
from autoevidence.schemas.enums import Disposal, FaultType
from autoevidence.schemas.models import AnalysisConclusion, AnalysisRecord, ClauseHit
from autoevidence.standards.loader import load_clause_library


def _rec(event: bool, ft: FaultType, dp: Disposal, clauses: list[str], evidence: str) -> AnalysisRecord:
    return AnalysisRecord(
        evidence=evidence,
        clause_check=[ClauseHit(clause_id=c, hit=True, why="抽查样例") for c in clauses],
        conclusion=AnalysisConclusion(quality_event=event, fault_type=ft, disposal=dp),
    )


def build_spotcheck_pairs() -> list[dict[str, Any]]:
    """构造若干 (高奖励候选, 低奖励候选) 供人工判断相对优劣。"""
    lib = load_clause_library()
    judge = RuleJudge()
    gold = {"quality_event": False, "fault_type": "无", "disposal": "归档"}

    better = _rec(
        False,
        FaultType.NONE,
        Disposal.ARCHIVE,
        ["Q06-02", "Q05-04"],
        "对话第1轮低温续航下降；信号窗口sw_01健康度98%",
    )
    worse = _rec(
        True,
        FaultType.BATTERY,
        Disposal.TECH,
        ["Q01-99"],
        "对话第1轮；编造条款",
    )
    rb = cited_reward(better, gold, lib, judge=judge)
    rw = cited_reward(worse, gold, lib, judge=judge)

    gold_crit = {
        "quality_event": True,
        "fault_type": "制动与转向",
        "disposal": "紧急复核",
    }
    good_crit = _rec(
        True,
        FaultType.BRAKE,
        Disposal.CRITICAL,
        ["Q02-03", "Q05-01"],
        "对话第3轮踏板发软；信号窗口sw_01建压超时",
    )
    dodge = _rec(
        True,
        FaultType.BRAKE,
        Disposal.TECH,
        ["Q02-03"],
        "对话第3轮；仅技术排查",
    )
    rg = cited_reward(good_crit, gold_crit, lib, judge=judge)
    rd = cited_reward(dodge, gold_crit, lib, judge=judge)

    pairs = [
        {
            "pair_id": "hardneg_correct_vs_fabricate",
            "gold": gold,
            "high_reward": {"record": better.model_dump(mode="json"), "score": rb},
            "low_reward": {"record": worse.model_dump(mode="json"), "score": rw},
            "human_prompt": "高分答案是否确实优于低分？若否，说明奖励漏洞。",
            "reviewer": {"better_is_truly_better": None, "notes": ""},
        },
        {
            "pair_id": "critical_vs_tech_dodge",
            "gold": gold_crit,
            "high_reward": {"record": good_crit.model_dump(mode="json"), "score": rg},
            "low_reward": {"record": dodge.model_dump(mode="json"), "score": rd},
            "human_prompt": "紧急复核正确案是否应高于「技术排查躲避」？",
            "reviewer": {"better_is_truly_better": None, "notes": ""},
        },
    ]
    return pairs


def export_spotcheck(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": (
            "nonzero_reward_variance_group_ratio 只表示组内奖励有差异，"
            "不表示差异符合真实质量优劣。正式训练前须完成人工抽查。"
        ),
        "pairs": build_spotcheck_pairs(),
        "instruction": "填写每对 reviewer.better_is_truly_better 为 true/false，并写 notes。",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
