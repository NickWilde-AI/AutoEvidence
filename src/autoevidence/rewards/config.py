"""奖励分值配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardConfig:
    verdict: float = 0.6
    fault_type: float = 0.3
    disposal: float = 0.6
    citation: float = 0.5
    fabricate: float = -0.6
    false_block: float = -0.4
    miss_critical: float = -0.7
    judge_max: float = 1.0
    # 防漏洞：用「技术排查」躲避安全漏放扣分时仍惩罚
    dodge_critical_via_tech: float = -0.7

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_REWARD = RewardConfig()
