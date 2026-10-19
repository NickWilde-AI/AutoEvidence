"""分段加权交叉熵：按 tokenizer 边界赋权，不按字符串长度。"""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn.functional as F


def segment_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weight: torch.Tensor,
) -> torch.Tensor:
    """
    logits: (B, L, V)
    labels: (B, L) 忽略位 -100
    token_weight: (B, L) prompt/pad 为 0
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    w = token_weight.float()
    denom = w.sum().clamp(min=1.0)
    return (ce * w).sum() / denom


def find_json_segment_spans(target_text: str) -> dict[str, tuple[int, int]]:
    """在目标 JSON 文本上定位三段字符跨度（用于再映射到 token）。"""
    data = json.loads(target_text)
    # 重新 dumps 保证与 render 一致
    evidence = json.dumps(data["evidence"], ensure_ascii=False)
    clause = json.dumps(data["clause_check"], ensure_ascii=False)
    conclusion = json.dumps(data["conclusion"], ensure_ascii=False)
    # 在完整 target 中找子串
    spans = {}
    for name, frag in [
        ("evidence", f'"evidence": {evidence}'),
        ("clause_check", f'"clause_check": {clause}'),
        ("conclusion", f'"conclusion": {conclusion}'),
    ]:
        idx = target_text.find(frag)
        if idx < 0:
            # 宽松：只找 key
            key = f'"{name}"'
            idx = target_text.find(key)
            end = len(target_text) if idx < 0 else idx + len(key)
            spans[name] = (max(idx, 0), end)
        else:
            spans[name] = (idx, idx + len(frag))
    return spans


def build_token_weights_from_offsets(
    offset_mapping: list[tuple[int, int]],
    spans: dict[str, tuple[int, int]],
    weights: dict[str, float],
    assistant_start_char: int = 0,
) -> list[float]:
    """
    使用 tokenizer offset_mapping 将字符跨度映射到 token 权重。
    prompt 段权重为 0；仅目标输出计损失。
    """
    out = []
    for start, end in offset_mapping:
        if end <= assistant_start_char or start == end:
            out.append(0.0)
            continue
        # 相对目标文本的字符位置
        rel_s = start - assistant_start_char
        rel_e = end - assistant_start_char
        w = weights.get("evidence", 1.0)
        for name, (a, b) in spans.items():
            if rel_s < b and rel_e > a:
                w = weights.get(name, 1.0)
                break
        out.append(float(w))
    return out


def mask_prompt_labels(input_ids: torch.Tensor, prompt_lens: list[int]) -> torch.Tensor:
    labels = input_ids.clone()
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100
    return labels


def collate_segment_batch(
    tokenizer,
    prompts: list[str],
    targets: list[str],
    segment_weights: dict[str, float],
    max_length: int = 1024,
) -> dict[str, torch.Tensor]:
    """构造 chat 风格拼接后的 batch；只对目标算损失。"""
    texts = []
    prompt_lens = []
    weight_rows = []
    for prompt, target in zip(prompts, targets):
        full = prompt + target
        texts.append(full)
        enc_p = tokenizer(prompt, add_special_tokens=False)
        prompt_lens.append(len(enc_p["input_ids"]))

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    labels = mask_prompt_labels(batch["input_ids"], prompt_lens)
    # pad 位置
    labels = labels.masked_fill(batch["attention_mask"] == 0, -100)

    offsets = batch.pop("offset_mapping")
    for i, target in enumerate(targets):
        try:
            spans = find_json_segment_spans(target)
        except Exception:
            spans = {
                "evidence": (0, len(target) // 3),
                "clause_check": (len(target) // 3, 2 * len(target) // 3),
                "conclusion": (2 * len(target) // 3, len(target)),
            }
        # offset 相对 full text；assistant 从 prompt_len 字符边界近似
        prompt = prompts[i]
        row = []
        for j, (s, e) in enumerate(offsets[i].tolist()):
            if labels[i, j] == -100:
                row.append(0.0)
                continue
            rel_s = s - len(prompt)
            rel_e = e - len(prompt)
            w = segment_weights.get("evidence", 1.0)
            for name, (a, b) in spans.items():
                if rel_s < b and rel_e > a:
                    w = segment_weights.get(name, 1.0)
                    break
            row.append(float(w))
        # 对齐长度
        while len(row) < labels.size(1):
            row.append(0.0)
        weight_rows.append(row[: labels.size(1)])

    token_weight = torch.tensor(weight_rows, dtype=torch.float32)
    batch["labels"] = labels
    batch["token_weight"] = token_weight
    return batch
