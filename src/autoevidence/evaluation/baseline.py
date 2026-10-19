"""规则基线：症状关键词 + 故障码/信号统计。"""

from __future__ import annotations

from autoevidence.evidence.symptom import RuleSymptomClassifier
from autoevidence.schemas.enums import Disposal, FaultType
from autoevidence.schemas.models import ModelOutput, Ticket


class RuleBaseline:
    name = "rule_engine_plus_signal"

    def __init__(self):
        self.clf = RuleSymptomClassifier()

    def predict(self, ticket: Ticket) -> ModelOutput:
        texts = [t.text for t in ticket.dialogue]
        preds = self.clf.predict(texts) if texts else []
        max_score = max((p.score for p in preds), default=0.0)
        systems = set()
        for p in preds:
            if p.score >= 0.55:
                systems.update(p.systems)

        anomaly = max((s.anomaly_score for s in ticket.signal_summary), default=0.0)
        has_dtc = any(s.dtc for s in ticket.signal_summary)

        # 难负例关键词
        blob = " ".join(texts)
        if "零下" in blob or "手刹" in blob or "保养" in blob and anomaly < 0.4:
            return ModelOutput(
                quality_event=False,
                fault_type=FaultType.NONE,
                disposal=Disposal.ARCHIVE,
                model_version=self.name,
                evidence_summary="规则命中豁免启发式",
            )

        event = max_score >= 0.55 or anomaly >= 0.7 or has_dtc
        if not event:
            return ModelOutput(
                quality_event=False,
                fault_type=FaultType.NONE,
                disposal=Disposal.ARCHIVE,
                model_version=self.name,
            )

        fault = FaultType.EE
        if "制动" in systems or "转向" in systems or "刹车" in blob:
            fault = FaultType.BRAKE
        elif "电池" in systems or "充电" in systems:
            fault = FaultType.BATTERY
        elif "动力" in systems or "传动" in systems:
            fault = FaultType.POWERTRAIN

        disposal = Disposal.CRITICAL if anomaly >= 0.85 or fault == FaultType.BRAKE else Disposal.TECH
        return ModelOutput(
            quality_event=True,
            fault_type=fault,
            disposal=disposal,
            model_version=self.name,
            evidence_summary=f"symptom={max_score}, anomaly={anomaly}, dtc={has_dtc}",
        )
