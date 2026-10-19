"""端到端案例集：正常 / 质量事件 / 难负例 / 证据不足 / 安全转人工。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoevidence.data.synthetic import generate_synthetic_corpus
from autoevidence.inference.pipeline import Analyzer


CASE_SELECTORS = [
    ("normal", lambda t: "normal" in t.tags and not t.hard_negative and not t.critical_safety),
    ("quality_event", lambda t: t.gold and t.gold.quality_event and t.critical_safety),
    ("hard_negative", lambda t: t.hard_negative),
    ("insufficient_evidence", lambda t: "evidence_missing" in t.tags),
    ("safety_to_human", lambda t: t.critical_safety),
]


def build_e2e_cases(n_corpus: int = 40) -> list[dict[str, Any]]:
    tickets = generate_synthetic_corpus(n_corpus)
    analyzer = Analyzer(mode="demo")
    cases: list[dict[str, Any]] = []
    used = set()

    for name, pred in CASE_SELECTORS:
        hit = next((t for t in tickets if pred(t) and t.ticket_id not in used), None)
        if hit is None:
            continue
        used.add(hit.ticket_id)
        # 推理前剥金标
        gold = hit.gold.model_dump(mode="json") if hit.gold else None
        result = analyzer.analyze(hit)
        cases.append(
            {
                "case_name": name,
                "ticket_id": hit.ticket_id,
                "input": {
                    "dialogue": [d.model_dump() for d in hit.dialogue],
                    "signal_summary": [s.model_dump() for s in hit.signal_summary],
                    "images": [i.model_dump() for i in hit.images],
                    "service_history": [h.model_dump() for h in hit.service_history],
                },
                "gold_for_reference_only": gold,
                "evidence_pack": result["evidence_pack"],
                "model_output": result["model_output"],
                "routing": result["routing"],
                "citations": result["model_output"].get("cited_clauses", []),
                "note": "synthetic demo；gold 仅供对照，推理未使用",
            }
        )
    return cases


def export_e2e_cases(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = build_e2e_cases()
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
