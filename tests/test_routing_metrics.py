from autoevidence.evaluation.routing_metrics import routing_dual_metrics


def test_dual_metrics_separated():
    preds = [
        {"quality_event": True, "disposal": "紧急复核"},
        {"quality_event": False, "disposal": "人审", "status": "证据不足"},
        {"quality_event": True, "disposal": "技术排查"},
    ]
    golds = [
        {"quality_event": True, "disposal": "紧急复核"},
        {"quality_event": False, "disposal": "归档"},
        {"quality_event": True, "disposal": "紧急复核"},
    ]
    routes = [
        {"queue": "紧急复核", "reason": "需确认", "require_human_confirm": True},
        {"queue": "人审", "reason": "证据不足", "require_human_confirm": True},
        {"queue": "技术排查", "reason": "回访", "require_human_confirm": False},
    ]
    crit = [True, False, True]
    m = routing_dual_metrics(preds, golds, routes, crit)
    assert m["safety_intercept_rate"]["numerator"] == 1  # 仅第0条进紧急复核
    assert m["safety_intercept_rate"]["denominator"] == 2
    assert m["incomplete_auto_rate"]["numerator"] == 1
    assert "分列" in m["definitions"]["warning"]
