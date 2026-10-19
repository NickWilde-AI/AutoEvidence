"""消融配置：区分需重训 vs 推理输入消融。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AblationKind = Literal["retrain", "inference_input", "reward", "data"]


@dataclass
class AblationSpec:
    name: str
    kind: AblationKind
    description: str
    flags: dict = field(default_factory=dict)
    status: Literal["not_run", "completed"] = "not_run"
    result_path: str | None = None


ABLATIONS: list[AblationSpec] = [
    AblationSpec("no_evidence_locate", "inference_input", "关闭证据定位，使用全量输入", {"locate": False}),
    AblationSpec("no_signal", "inference_input", "去掉信号输入", {"drop_signal": True}),
    AblationSpec("no_filter", "data", "关闭教师三道过滤", {"filter": False}),
    AblationSpec("no_segment_weight", "retrain", "普通交叉熵替代分段加权", {"segment_weight": False}),
    AblationSpec("no_clause_rewrite", "data", "去掉条款改写变体", {"rewrite": False}),
    AblationSpec("no_swap_aug", "data", "去掉换配增广", {"swap": False}),
    AblationSpec("no_citation_reward", "reward", "去掉引证支持奖励与编造扣分", {"citation_reward": False}),
    AblationSpec("no_judge", "reward", "去掉裁判核查", {"judge": False}),
    AblationSpec("no_safety_penalty", "reward", "去掉安全漏放惩罚", {"miss_critical": 0.0}),
    AblationSpec("random_rl_instead_of_hard", "data", "随机 RL 样本替换难例", {"random_rl": True}),
]


def apply_inference_ablation(ticket_dict: dict, spec: AblationSpec) -> dict:
    t = dict(ticket_dict)
    if spec.flags.get("drop_signal"):
        t["signal_summary"] = []
    if spec.flags.get("locate") is False:
        t["meta"] = {**(t.get("meta") or {}), "force_full_input": True}
    return t


def ablation_registry() -> list[dict]:
    return [
        {
            "name": a.name,
            "kind": a.kind,
            "description": a.description,
            "status": a.status,
            "result_path": a.result_path,
            "flags": a.flags,
        }
        for a in ABLATIONS
    ]
