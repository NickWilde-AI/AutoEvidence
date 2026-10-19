"""推理流水线：不读金标、不调用教师。"""

from __future__ import annotations

import uuid
from typing import Any

from autoevidence.evaluation.baseline import RuleBaseline
from autoevidence.evidence.locator import FaultClueLocator
from autoevidence.schemas.enums import Disposal, FaultType, RefusalReason
from autoevidence.schemas.models import AnalysisRecord, ModelOutput, Ticket
from autoevidence.standards.loader import load_clause_library
from autoevidence.synthesis.teacher import RuleTeacher


class Analyzer:
    """统一分析接口。demo 模式用规则教师生成可复核笔录（非训练教师链路）。"""

    def __init__(self, mode: str = "demo", model_path: str | None = None):
        self.mode = mode
        self.model_path = model_path
        self.locator = FaultClueLocator()
        self.baseline = RuleBaseline()
        self.lib = load_clause_library()
        self.demo_writer = RuleTeacher()
        self.model = None
        if mode == "model" and model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.model.eval()
        except Exception:
            self.mode = "demo"
            self.model = None

    def analyze(self, ticket: Ticket, request_id: str | None = None) -> dict[str, Any]:
        rid = request_id or str(uuid.uuid4())
        # 显式丢弃金标
        ticket_inf = ticket.model_copy(deep=True)
        ticket_inf.gold = None

        pack = self.locator.locate(ticket_inf)
        if pack.insufficient_evidence:
            out = ModelOutput(
                quality_event=False,
                fault_type=FaultType.NONE,
                disposal=Disposal.HUMAN,
                status=RefusalReason.INSUFFICIENT_EVIDENCE,
                model_version=f"{self.mode}",
                request_id=rid,
                evidence_summary="证据不足",
            )
            route = route_disposal(out)
            return {
                "request_id": rid,
                "model_output": out.model_dump(mode="json"),
                "evidence_pack": pack.model_dump(mode="json"),
                "routing": route,
                "mode": self.mode,
            }

        base = self.baseline.predict(ticket_inf)
        # demo：用规则写笔录，但结论跟基线（不看金标）
        from autoevidence.schemas.models import GoldLabel

        pseudo = GoldLabel(
            quality_event=base.quality_event,
            fault_type=base.fault_type,
            disposal=base.disposal,
        )
        record = self.demo_writer.synthesize(pack, pseudo, self.lib)
        out = ModelOutput(
            quality_event=record.conclusion.quality_event,
            fault_type=record.conclusion.fault_type,
            disposal=record.conclusion.disposal,
            record=record,
            cited_clauses=[c.clause_id for c in record.clause_check],
            evidence_summary=record.evidence[:200],
            model_version=f"{self.mode}:rule_baseline+record",
            request_id=rid,
        )
        route = route_disposal(out)
        return {
            "request_id": rid,
            "model_output": out.model_dump(mode="json"),
            "evidence_pack": pack.model_dump(mode="json"),
            "routing": route,
            "mode": self.mode,
        }


def route_disposal(out: ModelOutput) -> dict[str, Any]:
    """业务分流与模型判定分开保存。"""
    if out.status == RefusalReason.INSUFFICIENT_EVIDENCE:
        return {
            "queue": "人审",
            "reason": "证据不足",
            "require_human_confirm": True,
        }
    if out.disposal == Disposal.CRITICAL:
        return {
            "queue": "紧急复核",
            "reason": "模型判紧急复核，需质量工程师确认后执行",
            "require_human_confirm": True,
        }
    if out.disposal == Disposal.TECH:
        return {"queue": "技术排查", "reason": "生成回访任务单", "require_human_confirm": False}
    if out.disposal == Disposal.HUMAN:
        return {"queue": "人审", "reason": "边界样本", "require_human_confirm": True}
    return {"queue": "归档", "reason": "直接关闭", "require_human_confirm": False}
