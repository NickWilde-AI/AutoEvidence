"""症状句定位：可训练入口 + 规则基线回退。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from autoevidence.schemas.enums import SYSTEM_TO_FAULT

SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "制动": ["刹车", "制动", "踏板发软", "刹不住", "ABS"],
    "转向": ["方向盘", "转向", "发飘", "助力"],
    "电池": ["电池", "续航", "电芯", "绝缘"],
    "充电": ["充电", "快充", "充电口", "塑料味"],
    "动力": ["动力", "加速", "电机", "异响", "跛行"],
    "传动": ["传动", "减速器", "漏油"],
    "电子": ["中控", "黑屏", "死机", "影像"],
    "软件": ["OTA", "升级", "软件", "固件"],
    "空调": ["空调", "制冷"],
}

NEGATIVE_PATTERNS = [
    r"预约",
    r"谢谢",
    r"好的",
    r"没有其他",
    r"帮您安排",
]


@dataclass
class SymptomPrediction:
    score: float
    systems: list[str]


class SymptomClassifier(Protocol):
    def predict(self, texts: list[str]) -> list[SymptomPrediction]: ...


class RuleSymptomClassifier:
    """明确标注的规则基线 / 本地回退。"""

    name = "rule_baseline"
    threshold: float = 0.55

    def predict(self, texts: list[str]) -> list[SymptomPrediction]:
        out = []
        for text in texts:
            if any(re.search(p, text) for p in NEGATIVE_PATTERNS) and not any(
                kw in text for kws in SYMPTOM_KEYWORDS.values() for kw in kws
            ):
                out.append(SymptomPrediction(0.05, []))
                continue
            systems = []
            score = 0.0
            for sys, kws in SYMPTOM_KEYWORDS.items():
                hits = sum(1 for kw in kws if kw in text)
                if hits:
                    systems.append(sys)
                    score = max(score, min(0.95, 0.4 + 0.2 * hits))
            # 含糊描述降权
            if "说不清" in text or "怪怪的" in text:
                score = min(score, 0.25)
            out.append(SymptomPrediction(round(score, 2), systems))
        return out


class TrainableSymptomClassifier:
    """可训练文本分类器入口；未加载权重时回退规则。"""

    name = "trainable"

    def __init__(self, model_path: str | None = None, fallback: SymptomClassifier | None = None):
        self.model_path = model_path
        self.fallback = fallback or RuleSymptomClassifier()
        self._model = None
        if model_path:
            self._try_load(model_path)

    def _try_load(self, path: str) -> None:
        try:
            import torch

            self._model = torch.load(path, map_location="cpu")
        except Exception:
            self._model = None

    def predict(self, texts: list[str]) -> list[SymptomPrediction]:
        if self._model is None:
            return self.fallback.predict(texts)
        # 占位：真实权重应输出 (score, system_logits)
        return self.fallback.predict(texts)


def systems_to_fault_hints(systems: list[str]) -> list[str]:
    hints = []
    for s in systems:
        ft = SYSTEM_TO_FAULT.get(s)
        if ft:
            hints.append(ft.value)
    return sorted(set(hints))
