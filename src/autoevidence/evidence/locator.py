"""多源故障线索定位器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autoevidence.evidence.symptom import RuleSymptomClassifier, SymptomClassifier
from autoevidence.evidence.vision import HashVisionEncoder, VisionEncoder, cosine_sim_matrix
from autoevidence.schemas.enums import EvidenceImageSource, SignalAnchor
from autoevidence.schemas.models import (
    DialogueTurn,
    EvidenceImage,
    EvidencePack,
    SignalWindow,
    Ticket,
    TicketImage,
)


@dataclass
class LocatorConfig:
    max_windows: int = 3
    max_images: int = 3
    symptom_threshold: float = 0.55
    anomaly_threshold: float = 0.6
    sim_threshold: float = 0.35
    dedup_sim: float = 0.92


@dataclass
class ExemplarLibrary:
    """问题类型示例图库：仅作检索参考，不得伪装为本车故障证据。"""

    vectors: np.ndarray  # (M, D)
    types: list[str]
    image_ids: list[str] = field(default_factory=list)

    @classmethod
    def synthetic(cls, encoder: VisionEncoder, types: list[str] | None = None) -> "ExemplarLibrary":
        types = types or ["电池与高压系统", "制动与转向", "动力总成与传动", "电子电气与软件"]
        ids = [f"ex_{i}" for i in range(len(types) * 2)]
        keys = [f"{t}:{j}" for t in types for j in range(2)]
        vecs = encoder.encode_images(keys)
        type_of = [t for t in types for _ in range(2)]
        return cls(vectors=vecs, types=type_of, image_ids=ids)


class FaultClueLocator:
    def __init__(
        self,
        symptom_clf: SymptomClassifier | None = None,
        vision: VisionEncoder | None = None,
        exemplar: ExemplarLibrary | None = None,
        config: LocatorConfig | None = None,
    ):
        self.symptom_clf = symptom_clf or RuleSymptomClassifier()
        self.vision = vision or HashVisionEncoder()
        self.exemplar = exemplar or ExemplarLibrary.synthetic(self.vision)
        self.config = config or LocatorConfig()

    def locate(self, ticket: Ticket) -> EvidencePack:
        cfg = self.config
        reasons: list[str] = []

        # 无对话：用空对话不中断；车机自助可用单轮描述（由上游构造）
        dialogue = list(ticket.dialogue)
        if not dialogue and ticket.meta.get("self_service_text"):
            dialogue = [
                DialogueTurn(turn=1, speaker="车主", text=str(ticket.meta["self_service_text"]))
            ]
            reasons.append("used_self_service_text_as_dialogue")

        texts = [t.text for t in dialogue] or [""]
        preds = self.symptom_clf.predict(texts)
        scored_turns: list[DialogueTurn] = []
        systems: set[str] = set()
        for turn, pred in zip(dialogue, preds):
            scored_turns.append(
                DialogueTurn(
                    turn=turn.turn,
                    speaker=turn.speaker,
                    text=turn.text,
                    symptom_score=pred.score,
                    systems=pred.systems,
                )
            )
            if pred.score >= cfg.symptom_threshold:
                systems.update(pred.systems)

        # 信号窗口
        windows: list[SignalWindow] = []
        for w in ticket.signal_summary:
            if w.system in systems:
                ww = w.model_copy(deep=True)
                ww.anchor = SignalAnchor.SYMPTOM
                windows.append(ww)
                if len(windows) >= cfg.max_windows:
                    break
        for w in sorted(ticket.signal_summary, key=lambda x: -x.anomaly_score):
            if len(windows) >= cfg.max_windows:
                break
            if w.anomaly_score < cfg.anomaly_threshold:
                break
            if all(w.window_id != x.window_id for x in windows):
                ww = w.model_copy(deep=True)
                ww.anchor = SignalAnchor.ANOMALY
                windows.append(ww)

        # 照片：仅从当前工单候选；图库只提供相似度参考
        evidence_images: list[EvidenceImage] = []
        picked: list[int] = []
        image_vecs = None
        best = None
        best_idx = None

        if ticket.images:
            keys = [im.path or im.image_id for im in ticket.images]
            image_vecs = self.vision.encode_images(keys)
            sims = cosine_sim_matrix(image_vecs, self.exemplar.vectors)
            best = sims.max(axis=1) if sims.size else np.zeros(len(ticket.images))
            best_idx = sims.argmax(axis=1) if sims.size else np.zeros(len(ticket.images), dtype=int)

            def _take(i: int, source: EvidenceImageSource) -> None:
                if len(picked) >= cfg.max_images:
                    return
                if any(
                    float(image_vecs[i] @ image_vecs[j]) > cfg.dedup_sim for j in picked
                ):
                    return
                picked.append(i)
                im = ticket.images[i]
                evidence_images.append(
                    EvidenceImage(
                        image_id=im.image_id,
                        source=source,
                        sim=round(float(best[i]), 4),
                        nearest_type=self.exemplar.types[int(best_idx[i])],
                        path=im.path,
                        from_ticket=True,
                    )
                )

            for i, im in enumerate(ticket.images):
                if im.tech_tag and len(picked) < cfg.max_images:
                    _take(i, EvidenceImageSource.TECHNICIAN)
            order = list(np.argsort(-best)) if best is not None else []
            for i in order:
                if len(picked) >= cfg.max_images or float(best[i]) < cfg.sim_threshold:
                    break
                if i not in picked:
                    # 标注为示例图库相似度补入，但仍是本车照片
                    _take(i, EvidenceImageSource.EXEMPLAR_LIB)

        insufficient = False
        if not windows and not picked:
            reasons.append("all_locate_failed_using_default")
            insufficient = not ticket.signal_summary and not ticket.images and not any(
                (t.symptom_score or 0) >= cfg.symptom_threshold for t in scored_turns
            )
            if ticket.signal_summary:
                w = max(ticket.signal_summary, key=lambda x: x.anomaly_score).model_copy(deep=True)
                w.anchor = SignalAnchor.DEFAULT
                windows.append(w)
            if ticket.images and image_vecs is not None:
                for i in range(min(2, len(ticket.images))):
                    if i not in picked:
                        # reuse _take via inline
                        picked.append(i)
                        im = ticket.images[i]
                        evidence_images.append(
                            EvidenceImage(
                                image_id=im.image_id,
                                source=EvidenceImageSource.DEFAULT,
                                sim=round(float(best[i]), 4) if best is not None else 0.0,
                                nearest_type=self.exemplar.types[int(best_idx[i])]
                                if best_idx is not None
                                else None,
                                path=im.path,
                                from_ticket=True,
                            )
                        )

        if insufficient:
            reasons.append("insufficient_evidence")

        return EvidencePack(
            sample_id=f"evp_{ticket.ticket_id}",
            ticket_id=ticket.ticket_id,
            vehicle=ticket.vehicle,
            dialogue=scored_turns,
            signal_windows=windows,
            evidence_images=evidence_images,
            service_history=ticket.service_history[:5],
            insufficient_evidence=insufficient,
            locate_reasons=reasons,
            source=ticket.source,
            standard_version=ticket.standard_version,
        )


def locate_batch(tickets: list[Ticket], **kwargs) -> list[EvidencePack]:
    locator = FaultClueLocator(**kwargs)
    return [locator.locate(t) for t in tickets]
