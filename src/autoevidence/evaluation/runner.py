"""独立评测入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoevidence.evaluation.ablation import ablation_registry
from autoevidence.evaluation.baseline import RuleBaseline
from autoevidence.evaluation.metrics import citation_existence_rate, event_metrics, fault_disposal_accuracy
from autoevidence.schemas.models import Ticket
from autoevidence.standards.loader import load_clause_library


def _to_pred_dict(out) -> dict:
    return {
        "quality_event": out.quality_event,
        "fault_type": out.fault_type.value if hasattr(out.fault_type, "value") else out.fault_type,
        "disposal": out.disposal.value if hasattr(out.disposal, "value") else out.disposal,
        "parse_fail": out.status.value == "解析失败" if out.status else False,
    }


def evaluate_tickets(
    tickets: list[Ticket],
    predictor=None,
    set_name: str = "synthetic_internal",
    set_version: str = "v0",
) -> dict[str, Any]:
    predictor = predictor or RuleBaseline()
    preds, golds = [], []
    hard, crit, mismatch, parse_fail = [], [], [], []
    records = []

    for t in tickets:
        if t.gold is None:
            continue
        out = predictor.predict(t)
        pd = _to_pred_dict(out)
        preds.append(pd)
        golds.append(
            {
                "quality_event": t.gold.quality_event,
                "fault_type": t.gold.fault_type.value,
                "disposal": t.gold.disposal.value,
            }
        )
        hard.append(t.hard_negative or "hard_negative" in t.tags)
        crit.append(t.critical_safety)
        mismatch.append(t.mismatch == "口述信号错位" or "mismatch" in t.tags)
        parse_fail.append(bool(pd.get("parse_fail")))
        if out.record:
            records.append(out.record.model_dump(mode="json"))

    lib = load_clause_library()
    report = {
        "set_name": set_name,
        "set_version": set_version,
        "model_version": getattr(predictor, "name", "unknown"),
        "aggregation": "micro over samples; empty sets marked unavailable",
        "event": event_metrics(preds, golds, hard, crit, mismatch, parse_fail),
        "fault_disposal": fault_disposal_accuracy(preds, golds),
        "citation_existence": citation_existence_rate(records, set(lib.clauses)) if records else {
            "value": None,
            "available": False,
            "numerator": 0,
            "denominator": 0,
        },
        "ablation_registry": ablation_registry(),
        "note": "合成数据评测，不得当作线上真实效果",
    }
    return report


def save_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
