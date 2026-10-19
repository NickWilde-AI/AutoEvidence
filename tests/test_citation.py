from autoevidence.evidence.locator import FaultClueLocator
from autoevidence.data.synthetic import generate_synthetic_corpus
from autoevidence.schemas.enums import Disposal, FaultType
from autoevidence.schemas.models import AnalysisConclusion, AnalysisRecord, ClauseHit, GoldLabel
from autoevidence.standards.citation import apply_three_filters, check_refs_in_pack
from autoevidence.standards.loader import load_clause_library
from autoevidence.synthesis.pipeline import synthesize_one


def test_illegal_clause_rejected():
    tickets = generate_synthetic_corpus(5)
    pack = FaultClueLocator().locate(tickets[0])
    lib = load_clause_library()
    record = AnalysisRecord(
        evidence=f"对话第{pack.dialogue[0].turn}轮；信号窗口{pack.signal_windows[0].window_id if pack.signal_windows else 'sw_01'}",
        clause_check=[ClauseHit(clause_id="Q99-99", hit=True, why="编造")],
        conclusion=AnalysisConclusion(
            quality_event=True, fault_type=FaultType.BRAKE, disposal=Disposal.CRITICAL
        ),
    )
    gold = {"quality_event": True, "fault_type": "制动与转向", "disposal": "紧急复核"}
    filt = apply_three_filters(record, pack, gold, lib)
    assert not filt["citation_valid"]
    assert not filt["kept"]


def test_gallery_not_as_ticket_evidence():
    tickets = generate_synthetic_corpus(3)
    pack = FaultClueLocator().locate(tickets[0])
    # 人为加入图库伪证据
    from autoevidence.schemas.enums import EvidenceImageSource
    from autoevidence.schemas.models import EvidenceImage

    pack.evidence_images.append(
        EvidenceImage(
            image_id="img_gallery_01",
            source=EvidenceImageSource.EXEMPLAR_LIB,
            from_ticket=False,
        )
    )
    ok, detail = check_refs_in_pack("照片img_gallery_01可见渗漏", pack)
    assert not ok
    assert "img_gallery_01" in detail["gallery_cited"] or "img_gallery_01" in detail["bad_refs"]


def test_rule_teacher_kept():
    tickets = generate_synthetic_corpus(8)
    loc = FaultClueLocator()
    lib = load_clause_library()
    kept = 0
    for t in tickets:
        if not t.gold:
            continue
        pack = loc.locate(t)
        s = synthesize_one(pack, t.gold, lib)
        if s.kept:
            kept += 1
    assert kept >= 1
