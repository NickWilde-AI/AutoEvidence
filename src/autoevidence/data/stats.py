"""分布统计、近重复检测、缺失字段报告。脱敏字段不进入日志。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

from autoevidence.schemas.models import Ticket

SENSITIVE_KEYS = {"vehicle_id", "vin", "phone", "name", "plate"}


def _fingerprint(ticket: Ticket) -> str:
    texts = "||".join(t.text for t in ticket.dialogue)
    sigs = "||".join(s.summary for s in ticket.signal_summary)
    raw = f"{ticket.vehicle.model}|{texts}|{sigs}"
    return hashlib.sha1(raw.encode()).hexdigest()


def _near_dup_key(ticket: Ticket) -> str:
    # 粗粒度：对话文本去空白后哈希前缀
    text = re.sub(r"\s+", "", "".join(t.text for t in ticket.dialogue))
    return hashlib.md5(text.encode()).hexdigest()[:12]


def distribution_report(tickets: list[Ticket]) -> dict[str, Any]:
    fault = Counter()
    disposal = Counter()
    source = Counter()
    tags = Counter()
    models = Counter()
    missing = defaultdict(int)
    for t in tickets:
        if t.gold:
            fault[t.gold.fault_type.value] += 1
            disposal[t.gold.disposal.value] += 1
        else:
            missing["gold"] += 1
        source[t.source.value] += 1
        models[t.vehicle.model] += 1
        for tag in t.tags:
            tags[tag] += 1
        if not t.dialogue:
            missing["dialogue"] += 1
        if not t.signal_summary:
            missing["signal_summary"] += 1
        if not t.images:
            missing["images"] += 1
    fps = [_fingerprint(t) for t in tickets]
    exact_dups = len(fps) - len(set(fps))
    near = Counter(_near_dup_key(t) for t in tickets)
    near_groups = sum(1 for c in near.values() if c > 1)
    return {
        "count": len(tickets),
        "fault_type": dict(fault),
        "disposal": dict(disposal),
        "source": dict(source),
        "tags": dict(tags),
        "models": dict(models),
        "missing_fields": dict(missing),
        "exact_duplicate_pairs": exact_dups,
        "near_duplicate_groups": near_groups,
    }


def sanitize_for_log(obj: dict[str, Any]) -> dict[str, Any]:
    """递归移除疑似敏感字段，避免脱敏 ID 明文进日志。"""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        lk = k.lower()
        if any(s in lk for s in SENSITIVE_KEYS) and "hash" not in lk:
            out[k] = "<redacted>"
        elif isinstance(v, dict):
            out[k] = sanitize_for_log(v)
        else:
            out[k] = v
    return out
