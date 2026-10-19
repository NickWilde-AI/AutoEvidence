"""数据隔离：车辆/事件分组，同源变体同分区，泄漏检查。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from autoevidence.schemas.enums import SplitName
from autoevidence.schemas.models import SplitManifest, Ticket

SAFETY_SPLITS = {
    SplitName.SAFETY_GATE,
    SplitName.SAFETY_DEV,
    SplitName.SAFETY_SEAL,
}


def _group_key(ticket: Ticket) -> str:
    return ticket.vehicle.vehicle_id_hash


def _stable_bucket(key: str, seed: int = 42) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _dialogue_fp(ticket: Ticket) -> str:
    text = re.sub(r"\s+", "", "".join(t.text for t in ticket.dialogue))
    return hashlib.md5(text.encode()).hexdigest()[:12] if text else ""


def assign_splits(
    tickets: list[Ticket],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    holdout_safety: bool = True,
) -> list[Ticket]:
    """按车辆分组隔离后再切分。

    critical_safety 车辆以小比例进入 safety_gate（兼容旧名，语义=safety_dev 候选）。
    真正的 safety_seal 需通过 seal_safety_split 显式封存。
    """
    groups: dict[str, list[Ticket]] = defaultdict(list)
    for t in tickets:
        groups[_group_key(t)].append(t)

    safety_vehicles = set()
    if holdout_safety:
        for vid, items in groups.items():
            if any(x.critical_safety for x in items) and _stable_bucket(vid, seed + 7) < 0.15:
                safety_vehicles.add(vid)

    result: list[Ticket] = []
    for vid, items in groups.items():
        if vid in safety_vehicles:
            split = SplitName.SAFETY_GATE  # 开发用安全候选，非封存集
        else:
            r = _stable_bucket(vid, seed)
            if r < train_ratio:
                split = SplitName.TRAIN
            elif r < train_ratio + val_ratio:
                split = SplitName.VAL
            else:
                split = SplitName.TEST
        for t in items:
            cloned = t.model_copy(deep=True)
            cloned.split = split
            if split == SplitName.SAFETY_GATE:
                cloned.meta = {
                    **cloned.meta,
                    "safety_role": "safety_dev_candidate",
                    "note": "未封存；不得单独作为发布硬门槛",
                }
            result.append(cloned)
    return align_augment_parents(result)


def seal_safety_split(
    tickets: list[Ticket],
    ticket_ids: list[str] | None = None,
    version: str = "seal_v0",
) -> list[Ticket]:
    """将指定安全样本标记为 safety_seal（封存）。默认不自动封存全部 gate。"""
    id_set = set(ticket_ids or [])
    out = []
    for t in tickets:
        c = t.model_copy(deep=True)
        if c.ticket_id in id_set or (
            not id_set and c.split in (SplitName.SAFETY_GATE, SplitName.SAFETY_DEV) and c.critical_safety
        ):
            # 仅当显式传入 ticket_ids 时封存；空列表不封存，避免误封
            if id_set and c.ticket_id in id_set:
                c.split = SplitName.SAFETY_SEAL
                c.meta = {**c.meta, "safety_role": "safety_seal", "seal_version": version}
        out.append(c)
    return out


def align_augment_parents(tickets: list[Ticket]) -> list[Ticket]:
    by_id = {t.ticket_id: t for t in tickets}
    for t in tickets:
        if not t.parent_ids:
            continue
        parent_splits = {
            by_id[p].split for p in t.parent_ids if p in by_id and by_id[p].split
        }
        if not parent_splits:
            continue
        if len(parent_splits) > 1:
            t.meta["leakage_flag"] = "parent_split_conflict"
        else:
            t.split = next(iter(parent_splits))
    return tickets


def leakage_check(tickets: Iterable[Ticket]) -> dict:
    """车辆/事件/父样本/近重复文本/共享图片跨区检查。"""
    tickets = list(tickets)
    vehicle_splits: dict[str, set[str]] = defaultdict(set)
    event_splits: dict[str, set[str]] = defaultdict(set)
    parent_issues: list[str] = []
    fp_splits: dict[str, set[str]] = defaultdict(set)
    img_splits: dict[str, set[str]] = defaultdict(set)
    by_id = {t.ticket_id: t for t in tickets}

    for t in tickets:
        if t.split:
            sp = t.split.value
            vehicle_splits[t.vehicle.vehicle_id_hash].add(sp)
            event_splits[t.event_id].add(sp)
            fp = _dialogue_fp(t)
            if fp:
                fp_splits[fp].add(sp)
            for im in t.images:
                key = im.path or im.image_id
                if key:
                    img_splits[key].add(sp)
        for p in t.parent_ids:
            if p in by_id and by_id[p].split and t.split and by_id[p].split != t.split:
                parent_issues.append(f"{t.ticket_id}<={p}")

    vehicle_leak = {k: sorted(v) for k, v in vehicle_splits.items() if len(v) > 1}
    event_leak = {k: sorted(v) for k, v in event_splits.items() if len(v) > 1}
    near_dup_cross = {k: sorted(v) for k, v in fp_splits.items() if len(v) > 1}
    shared_img_cross = {k: sorted(v) for k, v in img_splits.items() if len(v) > 1}

    safety_vehicles = {
        t.vehicle.vehicle_id_hash
        for t in tickets
        if t.split in SAFETY_SPLITS
    }
    train_overlap = [
        t.ticket_id
        for t in tickets
        if t.split == SplitName.TRAIN and t.vehicle.vehicle_id_hash in safety_vehicles
    ]

    # 封存集若与 val 交叉调参痕迹：仅报告共车辆（人工治理）
    seal_vehicles = {
        t.vehicle.vehicle_id_hash for t in tickets if t.split == SplitName.SAFETY_SEAL
    }
    seal_in_train = [
        t.ticket_id
        for t in tickets
        if t.split == SplitName.TRAIN and t.vehicle.vehicle_id_hash in seal_vehicles
    ]

    hard_ok = (
        not vehicle_leak
        and not event_leak
        and not parent_issues
        and not train_overlap
        and not seal_in_train
    )
    strict_ok = hard_ok and not near_dup_cross and not shared_img_cross
    return {
        "vehicle_leak_count": len(vehicle_leak),
        "event_leak_count": len(event_leak),
        "parent_conflict_count": len(parent_issues),
        "near_dup_cross_split_count": len(near_dup_cross),
        "shared_image_cross_split_count": len(shared_img_cross),
        "safety_train_overlap": train_overlap,
        "seal_train_overlap": seal_in_train,
        "vehicle_leak_examples": dict(list(vehicle_leak.items())[:5]),
        "near_dup_examples": dict(list(near_dup_cross.items())[:5]),
        "shared_image_examples": dict(list(shared_img_cross.items())[:5]),
        "ok": hard_ok,
        "strict_ok": strict_ok,
        "notes": [
            "safety_gate 默认=safety_dev 候选，非 safety_seal",
            "按结果反复调参会使开发集失去独立性",
            "near_dup/shared_image 计入 strict_ok；合成复用场景可能触发警告",
        ],
    }


def build_manifest(tickets: list[Ticket], strategy: str = "vehicle_group_hash") -> SplitManifest:
    counts: dict[str, int] = defaultdict(int)
    vehicle_groups: dict[str, list[str]] = defaultdict(list)
    for t in tickets:
        sp = t.split.value if t.split else "unassigned"
        counts[sp] += 1
        vehicle_groups[sp].append(t.vehicle.vehicle_id_hash)
    for k in vehicle_groups:
        vehicle_groups[k] = sorted(set(vehicle_groups[k]))
    report = leakage_check(tickets)
    return SplitManifest(
        version=datetime.utcnow().strftime("%Y%m%d"),
        strategy=strategy,
        counts=dict(counts),
        vehicle_groups={k: v[:20] for k, v in vehicle_groups.items()},
        leakage_report=report,
        notes=["synthetic corpus"] if all(t.source.value == "synthetic" for t in tickets) else [],
    )


def save_split_artifacts(tickets: list[Ticket], out_dir: str | Path) -> SplitManifest:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[Ticket]] = defaultdict(list)
    for t in tickets:
        by_split[t.split.value if t.split else "unassigned"].append(t)
    for name, items in by_split.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for t in items:
                f.write(t.model_dump_json(exclude_none=True) + "\n")
    manifest = build_manifest(tickets)
    (out / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest
