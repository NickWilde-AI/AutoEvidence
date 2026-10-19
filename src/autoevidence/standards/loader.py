"""条款库加载、版本管理与引用校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ClauseLibrary(BaseModel):
    version: str
    title: str = ""
    source: str = "synthetic"
    sections: dict[str, str] = Field(default_factory=dict)
    clauses: dict[str, str]
    rewrites: dict[str, list[str]] = Field(default_factory=dict)

    def exists(self, clause_id: str) -> bool:
        return clause_id in self.clauses

    def text(self, clause_id: str, variant: int = 0) -> str:
        if variant <= 0:
            return self.clauses[clause_id]
        variants = self.rewrites.get(clause_id, [])
        if variant - 1 < len(variants):
            return variants[variant - 1]
        return self.clauses[clause_id]

    def as_prompt_dict(self, variant: int = 0) -> dict[str, str]:
        return {cid: self.text(cid, variant) for cid in self.clauses}

    def validate_ids(self, clause_ids: list[str]) -> tuple[bool, list[str]]:
        missing = [c for c in clause_ids if c not in self.clauses]
        return (len(missing) == 0, missing)


def default_standards_path(version: str = "v4") -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "standards" / f"{version}.json"


def load_clause_library(path: str | Path | None = None, version: str = "v4") -> ClauseLibrary:
    p = Path(path) if path else default_standards_path(version)
    data = json.loads(p.read_text(encoding="utf-8"))
    return ClauseLibrary.model_validate(data)


def rewrite_consistency_check(
    lib: ClauseLibrary,
    clause_id: str,
    judge_fn,
    sample_tickets: list[dict[str, Any]],
    threshold: float = 0.98,
) -> dict[str, Any]:
    """改写变体校验入口：用裁判对原文/改写判定一致性。

    judge_fn(clause_text, ticket) -> label dict
    """
    if clause_id not in lib.rewrites:
        return {"ok": False, "reason": "no_rewrites", "rate": 0.0}
    original = lib.clauses[clause_id]
    rates = []
    details = []
    for i, rewrite in enumerate(lib.rewrites[clause_id]):
        agree = 0
        for t in sample_tickets:
            a = judge_fn(original, t)
            b = judge_fn(rewrite, t)
            if a == b:
                agree += 1
        rate = agree / len(sample_tickets) if sample_tickets else 0.0
        rates.append(rate)
        details.append({"variant": i + 1, "agree_rate": rate, "accepted": rate >= threshold})
    return {
        "ok": all(d["accepted"] for d in details),
        "threshold": threshold,
        "details": details,
        "min_rate": min(rates) if rates else 0.0,
    }
