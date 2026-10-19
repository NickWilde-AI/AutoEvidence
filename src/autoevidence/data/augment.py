"""口述/信号换配增广与难负例过采样。"""

from __future__ import annotations

import copy
from typing import Literal

from autoevidence.schemas.enums import DataSource, Disposal, FaultType, SplitName
from autoevidence.schemas.models import GoldLabel, Ticket
from autoevidence.standards.loader import ClauseLibrary

SwapType = Literal["换口述", "换信号"]


def _compatible(a: Ticket, b: Ticket) -> bool:
    """车型与时间粗兼容：同车型系列前缀匹配。"""
    return a.vehicle.model.split("版")[0] == b.vehicle.model.split("版")[0]


def _auto_label_oral_hit(donor: Ticket) -> GoldLabel:
    """只剩口述命中：按口述症状对应问题类型，处置技术排查。"""
    assert donor.gold and donor.gold.quality_event
    return GoldLabel(
        quality_event=True,
        fault_type=donor.gold.fault_type,
        disposal=Disposal.TECH,
    )


def _auto_label_signal_hit(base: Ticket) -> GoldLabel:
    assert base.gold and base.gold.quality_event
    return GoldLabel(
        quality_event=True,
        fault_type=base.gold.fault_type,
        disposal=base.gold.disposal if base.gold.disposal != Disposal.ARCHIVE else Disposal.TECH,
    )


def swap_augment(
    tickets: list[Ticket],
    lib: ClauseLibrary | None = None,
    max_pairs: int = 20,
) -> list[Ticket]:
    """仅使用训练分区来源；记录父样本 ID；歧义样本标记人工复核。"""
    train = [t for t in tickets if t.split == SplitName.TRAIN and t.gold]
    events = [t for t in train if t.gold and t.gold.quality_event]
    normals = [t for t in train if t.gold and not t.gold.quality_event]
    out: list[Ticket] = []
    n = 0
    # 换口述：正常工单保留信号/照片，对话换成质量事件口述
    for base in normals:
        for donor in events:
            if n >= max_pairs:
                break
            if not _compatible(base, donor):
                continue
            if not donor.dialogue:
                continue
            t = base.model_copy(deep=True)
            t.ticket_id = f"swap_oral_{base.ticket_id}_{donor.ticket_id}"
            t.dialogue = copy.deepcopy(donor.dialogue)
            t.parent_ids = [base.ticket_id, donor.ticket_id]
            t.source = DataSource.AUGMENT_SWAP
            t.mismatch = "口述信号错位"
            t.tags = list(set(t.tags + ["augment_swap", "换口述", "needs_review"]))
            label = _auto_label_oral_hit(donor)
            # 信号仍正常 → 歧义，进人工复核标记，不强制当可靠金标
            t.gold = label
            t.meta = {
                **t.meta,
                "swap_type": "换口述",
                "auto_label_rule": "信号不命中、口述命中→技术排查",
                "label_confidence": "ambiguous_needs_human",
                "dialogue_from": donor.ticket_id,
            }
            t.split = SplitName.TRAIN
            out.append(t)
            n += 1
        if n >= max_pairs:
            break

    # 换信号：事件工单保留口述，信号/照片换成正常车
    m = 0
    for base in events:
        for donor in normals:
            if m >= max_pairs:
                break
            if not _compatible(base, donor):
                continue
            t = base.model_copy(deep=True)
            t.ticket_id = f"swap_sig_{base.ticket_id}_{donor.ticket_id}"
            t.signal_summary = copy.deepcopy(donor.signal_summary)
            t.images = copy.deepcopy(donor.images)
            t.parent_ids = [base.ticket_id, donor.ticket_id]
            t.source = DataSource.AUGMENT_SWAP
            t.mismatch = "口述信号错位"
            t.tags = list(set(t.tags + ["augment_swap", "换信号", "needs_review"]))
            t.gold = GoldLabel(
                quality_event=True,
                fault_type=base.gold.fault_type,
                disposal=Disposal.TECH,
            )
            t.meta = {
                **t.meta,
                "swap_type": "换信号",
                "auto_label_rule": "口述命中、信号替换为正常→技术排查",
                "label_confidence": "ambiguous_needs_human",
                "signal_from": donor.ticket_id,
            }
            t.split = SplitName.TRAIN
            out.append(t)
            m += 1
        if m >= max_pairs:
            break
    _ = lib  # 条款可用于更细自动定标；当前用规则
    return out


def oversample_hard_negatives(
    tickets: list[Ticket],
    ratio: float = 0.25,
    seed: int = 42,
) -> list[Ticket]:
    """难负例过采样；记录比例与来源。ratio 为难负例在结果中的目标占比上限。"""
    import random

    rng = random.Random(seed)
    hard = [t for t in tickets if t.hard_negative or "hard_negative" in t.tags]
    others = [t for t in tickets if t not in hard]
    if not hard:
        return list(tickets)
    target = max(len(hard), int(len(others) * ratio / max(1e-6, 1 - ratio)))
    extra = []
    while len(hard) + len(extra) < target:
        src = rng.choice(hard)
        c = src.model_copy(deep=True)
        c.ticket_id = f"{src.ticket_id}_os{len(extra)}"
        c.parent_ids = list(set(c.parent_ids + [src.ticket_id]))
        c.meta = {**c.meta, "oversample": True, "oversample_ratio": ratio}
        extra.append(c)
    result = others + hard + extra
    for t in result:
        t.meta.setdefault("hard_neg_sampling", {"ratio": ratio, "source": "hard_negative_pool"})
    return result


def select_unstable_samples(
    sample_hits: dict[str, list[bool]],
    min_hit: int = 1,
    max_hit: int = 6,
) -> list[str]:
    """按 G 次采样正确次数筛选边界样本。"""
    selected = []
    for sid, hits in sample_hits.items():
        c = sum(1 for h in hits if h)
        if min_hit <= c <= max_hit:
            selected.append(sid)
    return selected
