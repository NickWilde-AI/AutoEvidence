from autoevidence.evaluation.metrics import event_metrics, fault_disposal_accuracy


def test_event_metrics_hand_example():
    preds = [
        {"quality_event": True, "disposal": "紧急复核"},
        {"quality_event": True, "disposal": "技术排查"},
        {"quality_event": False, "disposal": "归档"},
        {"quality_event": True, "disposal": "技术排查"},
    ]
    golds = [
        {"quality_event": True, "disposal": "紧急复核"},
        {"quality_event": True, "disposal": "紧急复核"},
        {"quality_event": False, "disposal": "归档"},
        {"quality_event": False, "disposal": "归档"},
    ]
    hard = [False, False, True, True]
    crit = [True, True, False, False]
    m = event_metrics(preds, golds, hard, crit)
    assert m["precision"]["value"] == 0.666667 or abs(m["precision"]["value"] - 2 / 3) < 1e-5
    assert m["recall"]["value"] == 1.0
    assert abs(m["f1"]["value"] - 0.8) < 1e-5
    assert m["false_block_rate"]["value"] == 0.5
    assert m["hard_negative_error_rate"]["value"] == 0.5
    assert m["critical_safety_recall"]["value"] == 0.5


def test_empty_set_unavailable():
    m = event_metrics([], [], [], [])
    assert m["precision"]["available"] is False
    assert m["critical_safety_recall"]["available"] is False


def test_safety_gate_160_miss_4_is_97_5():
    # 原材料曾写 97.4%；正确计数 156/160=97.5%
    preds = [{"quality_event": True, "disposal": "紧急复核"}] * 156 + [
        {"quality_event": True, "disposal": "技术排查"}
    ] * 4
    golds = [{"quality_event": True, "disposal": "紧急复核"}] * 160
    crit = [True] * 160
    m = event_metrics(preds, golds, critical_mask=crit)
    assert m["critical_safety_recall"]["numerator"] == 156
    assert m["critical_safety_recall"]["denominator"] == 160
    assert abs(m["critical_safety_recall"]["value"] - 0.975) < 1e-9


def test_fault_type_only_on_events():
    preds = [
        {"quality_event": True, "fault_type": "制动与转向", "disposal": "紧急复核"},
        {"quality_event": True, "fault_type": "电子电气与软件", "disposal": "技术排查"},
        {"quality_event": False, "fault_type": "无", "disposal": "归档"},
    ]
    golds = [
        {"quality_event": True, "fault_type": "制动与转向", "disposal": "紧急复核"},
        {"quality_event": True, "fault_type": "电池与高压系统", "disposal": "紧急复核"},
        {"quality_event": False, "fault_type": "无", "disposal": "归档"},
    ]
    a = fault_disposal_accuracy(preds, golds)
    assert a["fault_type_accuracy"]["value"] == 0.5
    assert abs(a["disposal_accuracy"]["value"] - 2 / 3) < 1e-5
