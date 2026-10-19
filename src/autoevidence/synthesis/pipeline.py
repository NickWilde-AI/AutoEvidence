"""笔录合成流水线：教师生成 + 三道过滤 + 变体展开。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoevidence.schemas.enums import DataSource, SplitName
from autoevidence.schemas.models import (
    AnalysisRecord,
    EvidencePack,
    GoldLabel,
    SynthesizedSample,
    Ticket,
)
from autoevidence.standards.citation import apply_three_filters
from autoevidence.standards.loader import ClauseLibrary
from autoevidence.synthesis.teacher import (
    CheckpointedBatch,
    RuleTeacher,
    build_teacher_prompt,
)


def synthesize_one(
    pack: EvidencePack,
    gold: GoldLabel,
    lib: ClauseLibrary,
    teacher: RuleTeacher | Any = None,
    teacher_version: str = "rule_teacher_v1",
    prompt_version: str = "record_prompt_v1",
) -> SynthesizedSample:
    teacher = teacher or RuleTeacher()
    if isinstance(teacher, RuleTeacher):
        record = teacher.synthesize(pack, gold, lib)
        tver = teacher.version
        pver = teacher.prompt_version
    else:
        prompt = build_teacher_prompt(pack, lib)
        raw = teacher.generate(prompt)
        data = json.loads(raw)
        record = AnalysisRecord.model_validate(data)
        tver = getattr(teacher, "version", teacher_version)
        pver = getattr(teacher, "prompt_version", prompt_version)

    gold_dict = {
        "quality_event": gold.quality_event,
        "fault_type": gold.fault_type.value,
        "disposal": gold.disposal.value,
    }
    filt = apply_three_filters(record, pack, gold_dict, lib)
    return SynthesizedSample(
        sample_id=f"rec_{pack.sample_id}",
        evidence_pack_id=pack.sample_id,
        ticket_id=pack.ticket_id,
        standard_version=lib.version,
        record=record,
        gold=gold,
        citation_valid=filt["citation_valid"],
        conclusion_match=filt["conclusion_match"],
        ref_in_pack=filt["ref_in_pack"],
        kept=filt["kept"],
        reject_reasons=filt["reject_reasons"],
        teacher_version=tver,
        prompt_version=pver,
        needs_semantic_review=False,
        source=DataSource.TEACHER,
    )


def run_synthesis(
    tickets: list[Ticket],
    packs: list[EvidencePack],
    lib: ClauseLibrary,
    out_dir: str | Path,
    teacher: Any = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointedBatch(out / "synthesis_checkpoint.jsonl")
    pack_by_tid = {p.ticket_id: p for p in packs}
    kept_path = out / "kept.jsonl"
    rejected_path = out / "rejected.jsonl"
    stats = {"total": 0, "kept": 0, "rejected": 0, "skipped": 0}

    with kept_path.open("a", encoding="utf-8") as fk, rejected_path.open(
        "a", encoding="utf-8"
    ) as fr:
        for t in tickets:
            if t.gold is None:
                continue
            # 验证/测试集不用金标构造推理输入；合成仅允许 train（及明确的 coldstart 源）
            if t.split and t.split not in (SplitName.TRAIN,):
                # val 可用于教师质量抽检但不入库训练
                if t.split != SplitName.VAL:
                    continue
            pack = pack_by_tid.get(t.ticket_id)
            if pack is None:
                continue
            sid = f"rec_{pack.sample_id}"
            if ckpt.is_done(sid):
                stats["skipped"] += 1
                continue
            sample = synthesize_one(pack, t.gold, lib, teacher=teacher)
            sample.split = t.split
            sample.parent_ids = list(t.parent_ids)
            stats["total"] += 1
            row = sample.model_dump(mode="json")
            ckpt.append({"sample_id": sid, "kept": sample.kept})
            if sample.kept and t.split == SplitName.TRAIN:
                fk.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["kept"] += 1
            else:
                if not sample.kept:
                    fr.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
                elif t.split == SplitName.VAL:
                    # val 抽检另存
                    (out / "val_review.jsonl").open("a", encoding="utf-8").write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def expand_clause_variants(
    sample: SynthesizedSample,
    pack: EvidencePack,
    lib: ClauseLibrary,
    segment_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """一条笔录配原文 + 两份改写 → 三条训练条目。验证/测试不走此路径。"""
    weights = segment_weights or {"evidence": 1.0, "clause_check": 1.9, "conclusion": 1.3}
    cited = [c.clause_id for c in sample.record.clause_check]
    # 对引用条款准备 3 个变体索引：0 原文, 1/2 改写
    entries = []
    for variant in range(3):
        clauses = {}
        for cid in lib.clauses:
            if cid in cited or variant == 0:
                clauses[cid] = lib.text(cid, variant if cid in cited else 0)
            else:
                clauses[cid] = lib.clauses[cid]
        target = render_record(sample.record)
        entries.append(
            {
                "sample_id": f"{sample.sample_id}_v{variant}",
                "parent_sample_id": sample.sample_id,
                "evidence_pack": pack.model_dump(mode="json"),
                "clauses_in_prompt": clauses,
                "standard_version": lib.version,
                "variant": variant,
                "target": target,
                "segment_weights": weights,
                "gold": sample.gold.model_dump(mode="json"),
                "provenance": {
                    "ticket_id": sample.ticket_id,
                    "evidence_pack_id": sample.evidence_pack_id,
                    "sample_id": sample.sample_id,
                    "standard_version": lib.version,
                    "teacher_version": sample.teacher_version,
                    "prompt_version": sample.prompt_version,
                    "filter_kept": sample.kept,
                    "reject_reasons": sample.reject_reasons,
                    "split": sample.split.value if sample.split else None,
                    "parent_ids": sample.parent_ids,
                    "data_version": "synthetic",
                    "source": sample.source.value,
                },
            }
        )
    return entries


def render_record(record: AnalysisRecord) -> str:
    payload = {
        "evidence": record.evidence,
        "clause_check": [c.model_dump(mode="json") for c in record.clause_check],
        "conclusion": record.conclusion.model_dump(mode="json"),
    }
    return json.dumps(payload, ensure_ascii=False)
