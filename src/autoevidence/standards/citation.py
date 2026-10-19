"""条款引用与证据摘录越界检查。"""

from __future__ import annotations

import re
from typing import Iterable

from autoevidence.schemas.models import AnalysisRecord, EvidencePack
from autoevidence.standards.loader import ClauseLibrary

TURN_RE = re.compile(r"第(\d+)轮")
# 覆盖 img_02 / img_gallery_01 / sw_01 等编号
REF_RE = re.compile(r"(sw_[0-9a-z_]+|img_[0-9a-z_]+)", re.IGNORECASE)


def extract_cited_turns(evidence_text: str) -> set[str]:
    return set(TURN_RE.findall(evidence_text))


def extract_cited_refs(evidence_text: str) -> set[str]:
    return {m.lower() for m in REF_RE.findall(evidence_text)}


def check_citation_ids(record: AnalysisRecord, lib: ClauseLibrary) -> tuple[bool, list[str]]:
    ids = [c.clause_id for c in record.clause_check]
    return lib.validate_ids(ids)


def check_refs_in_pack(evidence_text: str, pack: EvidencePack) -> tuple[bool, dict[str, list[str]]]:
    turns = {str(t.turn) for t in pack.dialogue}
    refs = {w.window_id.lower() for w in pack.signal_windows} | {
        img.image_id.lower() for img in pack.evidence_images if img.from_ticket
    }
    cited_turns = extract_cited_turns(evidence_text)
    cited_refs = extract_cited_refs(evidence_text)
    bad_turns = sorted(cited_turns - turns)
    bad_refs = sorted(cited_refs - refs)
    # 图库示例不得作为本车证据引用通过
    gallery_ids = {
        img.image_id.lower() for img in pack.evidence_images if not img.from_ticket
    }
    gallery_cited = sorted(cited_refs & gallery_ids)
    ok = not bad_turns and not bad_refs and not gallery_cited
    return ok, {
        "bad_turns": bad_turns,
        "bad_refs": bad_refs,
        "gallery_cited": gallery_cited,
    }


def check_conclusion_match(record: AnalysisRecord, gold: dict) -> bool:
    c = record.conclusion
    return (
        c.quality_event == gold["quality_event"]
        and c.fault_type.value == gold["fault_type"]
        and c.disposal.value == gold["disposal"]
    )


def apply_three_filters(
    record: AnalysisRecord,
    pack: EvidencePack,
    gold: dict,
    lib: ClauseLibrary,
) -> dict:
    citation_valid, missing = check_citation_ids(record, lib)
    conclusion_match = check_conclusion_match(
        record,
        {
            "quality_event": gold["quality_event"],
            "fault_type": gold["fault_type"] if isinstance(gold["fault_type"], str) else gold["fault_type"].value,
            "disposal": gold["disposal"] if isinstance(gold["disposal"], str) else gold["disposal"].value,
        },
    )
    ref_in_pack, ref_detail = check_refs_in_pack(record.evidence, pack)
    reasons: list[str] = []
    if not citation_valid:
        reasons.append(f"illegal_clauses:{','.join(missing)}")
    if not conclusion_match:
        reasons.append("conclusion_mismatch")
    if not ref_in_pack:
        reasons.append(f"ref_out_of_pack:{ref_detail}")
    return {
        "citation_valid": citation_valid,
        "conclusion_match": conclusion_match,
        "ref_in_pack": ref_in_pack,
        "kept": citation_valid and conclusion_match and ref_in_pack,
        "reject_reasons": reasons,
        "ref_detail": ref_detail,
    }


def detect_fabricated_clauses(clause_ids: Iterable[str], lib: ClauseLibrary) -> list[str]:
    return [c for c in clause_ids if not lib.exists(c)]
