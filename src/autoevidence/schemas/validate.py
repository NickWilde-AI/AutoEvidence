"""标签与输出约束校验（不依赖提示词约定）。"""

from __future__ import annotations

from typing import Any

from autoevidence.schemas.enums import FAULT_TYPES_EVENT, Disposal, FaultType
from autoevidence.schemas.models import AnalysisConclusion, GoldLabel, ModelOutput


def validate_label_dict(d: dict[str, Any]) -> GoldLabel:
    return GoldLabel.model_validate(d)


def validate_model_output_dict(d: dict[str, Any]) -> ModelOutput:
    return ModelOutput.model_validate(d)


def assert_label_constraints(quality_event: bool, fault_type: str, disposal: str) -> list[str]:
    """返回违反约束的原因列表；空列表表示通过。"""
    errors: list[str] = []
    try:
        ft = FaultType(fault_type)
    except ValueError:
        return [f"非法 fault_type: {fault_type}"]
    try:
        dp = Disposal(disposal)
    except ValueError:
        return [f"非法 disposal: {disposal}"]

    if not quality_event:
        if ft != FaultType.NONE:
            errors.append("正常事件 fault_type 必须为「无」")
    else:
        if ft not in FAULT_TYPES_EVENT:
            errors.append("质量事件 fault_type 必须为四类之一")
        if dp == Disposal.ARCHIVE:
            # 质量事件归档在业务上可疑，但不硬性禁止（人审边界）
            pass
    return errors


def normalize_conclusion(c: AnalysisConclusion | dict[str, Any]) -> AnalysisConclusion:
    if isinstance(c, dict):
        return AnalysisConclusion.model_validate(c)
    return c
