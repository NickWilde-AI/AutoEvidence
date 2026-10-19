"""评测指标：分子分母显式输出；空集标记不可用。"""

from __future__ import annotations

from typing import Any


def _rate(num: int, den: int) -> dict[str, Any]:
    if den == 0:
        return {"value": None, "numerator": num, "denominator": den, "available": False}
    return {"value": round(num / den, 6), "numerator": num, "denominator": den, "available": True}


def event_metrics(
    preds: list[dict],
    golds: list[dict],
    hard_negative_mask: list[bool] | None = None,
    critical_mask: list[bool] | None = None,
    mismatch_mask: list[bool] | None = None,
    parse_fail_mask: list[bool] | None = None,
) -> dict[str, Any]:
    assert len(preds) == len(golds)
    tp = sum(p["quality_event"] and g["quality_event"] for p, g in zip(preds, golds))
    fp = sum(p["quality_event"] and not g["quality_event"] for p, g in zip(preds, golds))
    fn = sum(not p["quality_event"] and g["quality_event"] for p, g in zip(preds, golds))
    tn = sum(not p["quality_event"] and not g["quality_event"] for p, g in zip(preds, golds))
    normals = sum(not g["quality_event"] for g in golds)

    prec = _rate(tp, tp + fp)
    rec = _rate(tp, tp + fn)
    if prec["available"] and rec["available"] and (prec["value"] + rec["value"]) > 0:
        f1_v = 2 * prec["value"] * rec["value"] / (prec["value"] + rec["value"])
        f1 = {"value": round(f1_v, 6), "available": True, "from": "P/R"}
    else:
        f1 = {"value": None, "available": False}

    out: dict[str, Any] = {
        "sample_count": len(golds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_block_rate": _rate(fp, normals),
    }

    if hard_negative_mask is not None:
        hard_preds = [p for p, m in zip(preds, hard_negative_mask) if m]
        out["hard_negative_error_rate"] = _rate(
            sum(p["quality_event"] for p in hard_preds), len(hard_preds)
        )
    else:
        out["hard_negative_error_rate"] = _rate(0, 0)

    if critical_mask is not None:
        crit = [p for p, m in zip(preds, critical_mask) if m]
        out["critical_safety_recall"] = _rate(
            sum(p.get("disposal") == "紧急复核" for p in crit), len(crit)
        )
    else:
        out["critical_safety_recall"] = _rate(0, 0)

    if mismatch_mask is not None:
        mm = [(p, g) for p, g, m in zip(preds, golds, mismatch_mask) if m]
        out["mismatch_recall"] = _rate(
            sum(p["quality_event"] and g["quality_event"] for p, g in mm),
            sum(g["quality_event"] for _, g in mm),
        )
    else:
        out["mismatch_recall"] = _rate(0, 0)

    if parse_fail_mask is not None:
        out["parse_fail_rate"] = _rate(sum(parse_fail_mask), len(parse_fail_mask))
    else:
        out["parse_fail_rate"] = _rate(0, 0)

    return out


def fault_disposal_accuracy(preds: list[dict], golds: list[dict]) -> dict[str, Any]:
    events = [(p, g) for p, g in zip(preds, golds) if g["quality_event"]]
    type_ok = sum(p["fault_type"] == g["fault_type"] for p, g in events)
    disp_ok = sum(p["disposal"] == g["disposal"] for p, g in zip(preds, golds))
    return {
        "fault_type_accuracy": _rate(type_ok, len(events)),
        "disposal_accuracy": _rate(disp_ok, len(golds)),
        "note": "问题类型准确率仅在金标质量事件上计算",
    }


def citation_existence_rate(
    records: list[dict],
    clause_ids: set[str],
) -> dict[str, Any]:
    ok = 0
    for r in records:
        cited = [c["clause_id"] for c in r.get("clause_check", [])]
        if cited and all(c in clause_ids for c in cited):
            ok += 1
    return _rate(ok, len(records))
