"""safety_seal 封存：清单、哈希、使用规则与失败处置。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from autoevidence.data.split import leakage_check
from autoevidence.schemas.enums import SplitName
from autoevidence.schemas.models import Ticket

SEAL_USAGE_RULES = [
    "不用于训练（SFT / GRPO rollout / 增广父样本）",
    "不用于调奖励权重、阈值或规则",
    "不用于选 checkpoint / 早停",
    "仅在正式候选配置冻结后评测一次",
    "门禁失败则阻断发布；本版保留为已暴露回归集，下一次验收使用新封存集",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _ticket_canonical_bytes(ticket: Ticket) -> bytes:
    # 稳定序列化：排除 split 变更噪声时仍以业务字段为准
    payload = ticket.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def build_seal_manifest(
    tickets: list[Ticket],
    *,
    seal_version: str,
    standard_version: str = "v4",
    label_version: str = "gold_v0",
    clause_file_hash: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    samples = []
    for t in tickets:
        raw = _ticket_canonical_bytes(t)
        samples.append(
            {
                "ticket_id": t.ticket_id,
                "event_id": t.event_id,
                "vehicle_id_hash": t.vehicle.vehicle_id_hash,
                "content_sha256": _sha256_bytes(raw),
                "critical_safety": t.critical_safety,
                "gold": t.gold.model_dump(mode="json") if t.gold else None,
            }
        )
    samples.sort(key=lambda x: x["ticket_id"])
    body = {
        "seal_version": seal_version,
        "sealed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_count": len(samples),
        "sample_ids": [s["ticket_id"] for s in samples],
        "samples": samples,
        "label_version": label_version,
        "standard_version": standard_version,
        "clause_file_hash": clause_file_hash,
        "usage_rules": list(SEAL_USAGE_RULES),
        "failure_policy": {
            "on_fail": "block_release",
            "retain_as": "exposed_regression_set",
            "next_acceptance": "require_new_seal_version",
        },
        "leakage_gate": {
            "formal_requires": "strict_ok=true",
            "unresolved_near_dup_or_shared_image": "must_itemize_justification",
        },
        "notes": notes
        or [
            "synthetic seal；业务效果未验证",
            "nonzero_reward_variance 不表示奖励符合真实质量优劣",
        ],
    }
    # 清单自身哈希（不含 manifest_sha256 字段）
    canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    body["manifest_sha256"] = _sha256_bytes(canon)
    return body


def seal_tickets(
    tickets: Iterable[Ticket],
    out_dir: str | Path,
    *,
    seal_version: str = "seal_v0",
    standard_version: str = "v4",
    label_version: str = "gold_v0_independent_template",
    clause_path: str | Path | None = None,
    require_strict_ok: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """将候选安全样本写入封存目录并生成清单。

    默认只封存 critical_safety 样本；写 split=safety_seal。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = [
        t.model_copy(deep=True)
        for t in tickets
        if t.critical_safety and t.gold is not None
    ]
    if max_samples is not None:
        candidates = candidates[:max_samples]
    if not candidates:
        raise ValueError("没有可封存的 critical_safety 样本")

    for t in candidates:
        t.split = SplitName.SAFETY_SEAL
        t.meta = {
            **t.meta,
            "safety_role": "safety_seal",
            "seal_version": seal_version,
            "independent_gold_status": "template_pending_human_review",
        }

    # 泄漏：仅封存集内部 + 与外部对照由调用方传入全集；此处对封存子集做自检
    leak = leakage_check(candidates)
    if require_strict_ok and not leak.get("strict_ok", False):
        # 封存集内近重复常见于合成；记录解释位，不默认硬失败合成 demo
        justifications = []
        if leak.get("near_dup_cross_split_count", 0) > 0:
            justifications.append(
                {
                    "type": "near_dup",
                    "count": leak["near_dup_cross_split_count"],
                    "examples": leak.get("near_dup_examples", {}),
                    "reason": "封存子集若仅含同类安全场景，对话指纹可能相同；正式业务封存须逐项消解或换样",
                }
            )
        if leak.get("shared_image_cross_split_count", 0) > 0:
            justifications.append(
                {
                    "type": "shared_image",
                    "count": leak["shared_image_cross_split_count"],
                    "examples": leak.get("shared_image_examples", {}),
                    "reason": "共图跨区须业务解释；合成封存若仅单区则不应触发",
                }
            )
        # 单区内 near_dup：leakage_check 按 split 集合，单 seal 区 near_dup 不会跨区
        # 正式验收仍要求调用方对全集跑 strict_ok
        leak["seal_subset_justifications"] = justifications

    tickets_path = out / "tickets.jsonl"
    with tickets_path.open("w", encoding="utf-8") as f:
        for t in candidates:
            f.write(t.model_dump_json(exclude_none=True) + "\n")

    clause_hash = None
    if clause_path and Path(clause_path).exists():
        clause_hash = _sha256_file(Path(clause_path))
    else:
        default_clause = Path(__file__).resolve().parents[3] / "data" / "standards" / "v4.json"
        if default_clause.exists():
            clause_hash = _sha256_file(default_clause)

    manifest = build_seal_manifest(
        candidates,
        seal_version=seal_version,
        standard_version=standard_version,
        label_version=label_version,
        clause_file_hash=clause_hash,
        notes=[
            "synthetic engineering seal",
            "金标独立复核状态见 independent_gold_status；未人工复核前不得宣称业务门禁通过",
        ],
    )
    manifest["tickets_jsonl_sha256"] = _sha256_file(tickets_path)
    manifest["leakage_on_seal_subset"] = leak
    manifest["formal_acceptance"] = {
        "requires_strict_ok_on_full_corpus": True,
        "requires_independent_gold_review": True,
        "requires_config_freeze_before_eval": True,
    }

    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    policy = {
        "seal_version": seal_version,
        "usage_rules": SEAL_USAGE_RULES,
        "failure_policy": manifest["failure_policy"],
        "reward_note": (
            "nonzero_reward_variance_group_ratio 只说明候选奖励有差异，"
            "不说明高奖励答案确实优于低奖励；正式训练前须人工抽查"
        ),
    }
    (out / "USAGE_AND_FAILURE_POLICY.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def mark_seal_failed(
    seal_dir: str | Path,
    *,
    reason: str,
    failed_ticket_ids: list[str] | None = None,
) -> dict[str, Any]:
    """门禁失败：阻断发布，本版转为已暴露回归集。"""
    seal_dir = Path(seal_dir)
    man_path = seal_dir / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    record = {
        "status": "failed_exposed_regression",
        "failed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
        "failed_ticket_ids": failed_ticket_ids or [],
        "original_seal_version": manifest.get("seal_version"),
        "next_step": "改进后创建新 seal_version；本目录仅作回归暴露集保留",
        "block_release": True,
    }
    (seal_dir / "FAILURE_RECORD.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["status"] = "failed_exposed_regression"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def verify_seal_integrity(seal_dir: str | Path) -> dict[str, Any]:
    """复核 tickets 内容哈希与清单是否一致。"""
    seal_dir = Path(seal_dir)
    manifest = json.loads((seal_dir / "manifest.json").read_text(encoding="utf-8"))
    tickets = []
    with (seal_dir / "tickets.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tickets.append(Ticket.model_validate_json(line))
    mismatches = []
    by_id = {s["ticket_id"]: s for s in manifest["samples"]}
    for t in tickets:
        got = _sha256_bytes(_ticket_canonical_bytes(t))
        exp = by_id.get(t.ticket_id, {}).get("content_sha256")
        if got != exp:
            mismatches.append(t.ticket_id)
    file_hash = _sha256_file(seal_dir / "tickets.jsonl")
    return {
        "ok": not mismatches and file_hash == manifest.get("tickets_jsonl_sha256"),
        "mismatched_tickets": mismatches,
        "tickets_jsonl_sha256_match": file_hash == manifest.get("tickets_jsonl_sha256"),
        "sample_count": len(tickets),
        "seal_version": manifest.get("seal_version"),
        "status": manifest.get("status", "sealed"),
    }
