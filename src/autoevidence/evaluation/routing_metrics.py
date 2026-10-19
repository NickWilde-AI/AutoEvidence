"""人审分流双口径：安全拦截 vs 自动诊断未完成。"""

from __future__ import annotations

from typing import Any

from autoevidence.evaluation.metrics import _rate


def routing_dual_metrics(
    preds: list[dict],
    golds: list[dict],
    routes: list[dict],
    critical_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """分列统计，禁止用转人工抬高安全分。

    - safety_intercept: 金标安全样本被路由到「紧急复核」队列
    - incomplete_auto: 非上述拦截情形下，因证据不足/解析失败/边界不确定转「人审」
    """
    n = len(golds)
    assert len(preds) == n == len(routes)

    if critical_mask is None:
        crit_idx = {i for i, g in enumerate(golds) if g.get("disposal") == "紧急复核"}
    else:
        crit_idx = {i for i, m in enumerate(critical_mask) if m}

    intercept = sum(1 for i in crit_idx if routes[i].get("queue") == "紧急复核")

    incomplete = 0
    for i, r in enumerate(routes):
        if i in crit_idx and r.get("queue") == "紧急复核":
            continue
        reason = str(r.get("reason", ""))
        queue = r.get("queue", "")
        status = preds[i].get("status")
        is_incomplete = (
            queue == "人审"
            and (
                "证据不足" in reason
                or "解析" in reason
                or "边界" in reason
                or status in ("证据不足", "解析失败", "拒答", "未知")
            )
        ) or reason == "证据不足"
        if is_incomplete:
            incomplete += 1

    return {
        "safety_intercept_rate": _rate(intercept, len(crit_idx)),
        "incomplete_auto_rate": _rate(incomplete, n),
        "definitions": {
            "safety_intercept_rate": "金标安全事件 → 紧急复核队列（安全拦截成功）",
            "incomplete_auto_rate": "证据不足/解析失败等 → 人审（自动诊断未完成）",
            "warning": "禁止用转人工抬高安全分；两指标必须分列",
        },
    }
