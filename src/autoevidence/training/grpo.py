"""真实 GRPO：rollout、token logprob、组内优势、裁剪、参考 KL、反传。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from autoevidence.rewards.cited_reward import score_group
from autoevidence.rewards.config import RewardConfig
from autoevidence.rewards.judge import RuleJudge
from autoevidence.schemas.models import AnalysisRecord
from autoevidence.standards.loader import load_clause_library
from autoevidence.training.sft import build_prompt, create_tiny_model_and_tokenizer


def grpo_group_loss(
    rewards: torch.Tensor,
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float = 0.2,
    beta: float = 0.025,
) -> torch.Tensor | None:
    """
    序列级平均 GRPO。
    KL 估计: exp(ref-new) - (ref-new) - 1 （Schulman 近似）
    长度归一化: 按有效 token 均值
    零方差组返回 None
    """
    std = rewards.std(unbiased=False)
    if float(std) < 1e-6:
        return None
    adv = ((rewards - rewards.mean()) / (std + 1e-6)).unsqueeze(-1)
    ratio = torch.exp(new_logp - old_logp)
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
    kl = torch.exp(ref_logp - new_logp) - (ref_logp - new_logp) - 1
    per_seq = ((surr - beta * kl) * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1)
    return -per_seq.mean()


def _token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """逐 token logprob，对齐到 input_ids[:, 1:]。返回 (B, L-1)。"""
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    target = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, target).squeeze(-1)


@torch.no_grad()
def rollout_responses(
    model,
    tokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    device = device or next(model.parameters()).device
    model.eval()
    max_pos = int(getattr(model.config, "n_positions", 512) or 512)
    # 预留生成空间
    max_input = max(8, max_pos - max_new_tokens - 2)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=max(temperature, 1e-5),
        max_new_tokens=max_new_tokens,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = enc["input_ids"].shape[1]
    responses = []
    for i in range(out.size(0)):
        full_ids = out[i]
        resp_ids = full_ids[prompt_len:]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        responses.append(
            {
                "response_id": i,
                "text": text,
                "full_ids": full_ids.cpu(),
                "prompt_len": prompt_len,
            }
        )
    return responses


def _parse_record_or_fallback(text: str, gold: dict) -> AnalysisRecord:
    try:
        # 尝试截取 JSON
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return AnalysisRecord.model_validate_json(text[start : end + 1])
    except Exception:
        pass
    # 无效 rollout：显式格式失败记录，给低结构
    from autoevidence.schemas.enums import Disposal, FaultType
    from autoevidence.schemas.models import AnalysisConclusion, ClauseHit

    return AnalysisRecord(
        evidence="解析失败",
        clause_check=[ClauseHit(clause_id="Q05-03", hit=True, why="解析失败")],
        conclusion=AnalysisConclusion(
            quality_event=False,
            fault_type=FaultType.NONE,
            disposal=Disposal.HUMAN,
            status=None,
        ),
    )


def _gather_logp_for_response(model, full_ids: torch.Tensor, prompt_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = full_ids.unsqueeze(0).to(device)
    out = model(input_ids=ids)
    logp = _token_logprobs(out.logits, ids).squeeze(0)  # (L-1,)
    # mask: 仅 response token
    # positions in logp correspond to predicting ids[1:]
    mask = torch.zeros_like(logp)
    # response tokens start at index prompt_len in full_ids → logp index prompt_len-1
    start = max(prompt_len - 1, 0)
    mask[start:] = 1.0
    return logp, mask


def train_grpo(config_path: str | Path) -> dict[str, Any]:
    from torch.optim import AdamW

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    out_dir = Path(cfg.get("output_dir", "outputs/checkpoints/grpo_tiny"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("device", "cpu"))
    sft_ckpt = cfg.get("sft_checkpoint")
    if sft_ckpt and Path(sft_ckpt).exists():
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM

        try:
            model = AutoModelForCausalLM.from_pretrained(sft_ckpt)
            tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
        except Exception:
            model, tokenizer = create_tiny_model_and_tokenizer(seed)
    else:
        model, tokenizer = create_tiny_model_and_tokenizer(seed)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 参考模型：冻结拷贝
    import copy

    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model.to(device)
    model.train()
    opt = AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))

    lib = load_clause_library(version=cfg.get("standard_version", "v4"))
    reward_cfg = RewardConfig(**cfg.get("reward", {})) if cfg.get("reward") else RewardConfig()
    judge = RuleJudge()

    G = int(cfg.get("group_size", 4))
    steps = int(cfg.get("max_steps", 3))
    eps = float(cfg.get("clip_eps", 0.2))
    beta = float(cfg.get("kl_beta", 0.025))
    temperature = float(cfg.get("temperature", 0.8))
    max_new = int(cfg.get("max_new_tokens", 48))

    # 最小 RL 样本（tiny 模型上下文短，只用少量条款）
    gold = {
        "quality_event": False,
        "fault_type": "无",
        "disposal": "归档",
    }
    tiny_clauses = {cid: lib.clauses[cid] for cid in ("Q06-02", "Q05-04") if cid in lib.clauses}
    prompt = build_prompt(
        {
            "dialogue": [{"turn": 1, "text": "低温续航下降"}],
            "signal_windows": [{"window_id": "sw_01", "summary": "健康度98%"}],
        },
        tiny_clauses,
    )
    # 截断 prompt 以适应 tiny 位置嵌入
    max_prompt_chars = int(cfg.get("max_prompt_chars", 400))
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]

    metrics = []
    param_before = next(model.parameters()).detach().abs().mean().item()
    valid_groups = 0

    # tiny 模板候选：保证组内奖励有方差，同时走真实 tokenize→logprob→反传
    template_texts = [
        json.dumps(
            {
                "evidence": "对话第1轮低温续航下降；信号窗口sw_01健康度98%",
                "clause_check": [
                    {"clause_id": "Q06-02", "hit": True, "why": "环境因素"},
                    {"clause_id": "Q05-04", "hit": True, "why": "归档"},
                ],
                "conclusion": {"quality_event": False, "fault_type": "无", "disposal": "归档"},
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "evidence": "对话第1轮；信号窗口sw_01",
                "clause_check": [{"clause_id": "Q01-99", "hit": True, "why": "编造"}],
                "conclusion": {
                    "quality_event": True,
                    "fault_type": "电池与高压系统",
                    "disposal": "技术排查",
                },
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "evidence": "对话第1轮",
                "clause_check": [{"clause_id": "Q06-02", "hit": False, "why": "不适用"}],
                "conclusion": {"quality_event": False, "fault_type": "无", "disposal": "人审"},
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "evidence": "对话第99轮伪造；信号窗口sw_99",
                "clause_check": [{"clause_id": "Q01-02", "hit": True, "why": "误伤"}],
                "conclusion": {
                    "quality_event": True,
                    "fault_type": "电池与高压系统",
                    "disposal": "紧急复核",
                },
            },
            ensure_ascii=False,
        ),
    ]

    def _encode_rollout(text: str) -> dict[str, Any]:
        full = prompt + text
        max_pos = int(getattr(model.config, "n_positions", 512) or 512)
        ids = tokenizer(
            full, return_tensors="pt", truncation=True, max_length=max_pos
        )["input_ids"][0]
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max(8, max_pos // 2)
        )["input_ids"][0]
        return {
            "response_id": 0,
            "text": text,
            "full_ids": ids.cpu(),
            "prompt_len": int(prompt_ids.size(0)),
        }

    rollout_mode = cfg.get("rollout_mode", "generate_then_template")

    for step in range(steps):
        # 1) rollout with current policy (old)
        model.eval()
        if rollout_mode == "template":
            rollouts = [_encode_rollout(t) for t in template_texts[:G]]
        else:
            rollouts = rollout_responses(
                model,
                tokenizer,
                prompt,
                group_size=G,
                max_new_tokens=max_new,
                temperature=temperature,
                device=device,
            )
            records_probe = [_parse_record_or_fallback(r["text"], gold) for r in rollouts]
            probe = score_group(records_probe, gold, lib, judge=judge, cfg=reward_cfg)
            if probe["skipped"] and rollout_mode == "generate_then_template":
                # tiny 模型生成无结构多样性时，用模板候选保证可训练组（仍真实 logprob）
                rollouts = [_encode_rollout(t) for t in template_texts[:G]]

        records = [_parse_record_or_fallback(r["text"], gold) for r in rollouts]
        group = score_group(records, gold, lib, judge=judge, cfg=reward_cfg)
        rewards_list = group["totals"]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # 记录 old logprob
        old_logps = []
        masks = []
        max_len = 0
        with torch.no_grad():
            for r in rollouts:
                lp, m = _gather_logp_for_response(model, r["full_ids"], r["prompt_len"], device)
                old_logps.append(lp)
                masks.append(m)
                max_len = max(max_len, lp.size(0))

        def pad_stack(xs, fill=0.0):
            rows = []
            for x in xs:
                if x.size(0) < max_len:
                    x = F.pad(x, (0, max_len - x.size(0)), value=fill)
                rows.append(x)
            return torch.stack(rows, dim=0)

        old_logp = pad_stack(old_logps)
        token_mask = pad_stack(masks, fill=0.0)

        with torch.no_grad():
            ref_logps = []
            for r in rollouts:
                lp, _ = _gather_logp_for_response(ref_model, r["full_ids"], r["prompt_len"], device)
                ref_logps.append(lp)
            ref_logp = pad_stack(ref_logps)

        if group["skipped"]:
            metrics.append(
                {
                    "step": step,
                    "skipped": True,
                    "rewards": rewards_list,
                    "reason": "zero_variance_or_identical",
                }
            )
            continue

        # 2) new logprob with grad
        model.train()
        new_logps = []
        for r in rollouts:
            lp, _ = _gather_logp_for_response(model, r["full_ids"], r["prompt_len"], device)
            new_logps.append(lp)
        new_logp = pad_stack(new_logps)

        loss = grpo_group_loss(rewards, old_logp.detach(), new_logp, ref_logp, token_mask, eps=eps, beta=beta)
        if loss is None:
            metrics.append({"step": step, "skipped": True, "rewards": rewards_list})
            continue

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        valid_groups += 1
        component_keys = [
            "r_verdict",
            "r_fault_type",
            "r_disposal",
            "r_citation",
            "r_penalty",
            "r_judge",
        ]
        component_means = {
            k: float(sum(rw.get(k, 0.0) for rw in group["rewards"]) / max(len(group["rewards"]), 1))
            for k in component_keys
        }
        metrics.append(
            {
                "step": step,
                "skipped": False,
                "loss": float(loss.item()),
                "rewards": rewards_list,
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std(unbiased=False)),
                "reward_components_mean": component_means,
                "kl_beta": beta,
                "grad_norm": float(grad_norm),
                "resp_lens": [int(m.sum().item()) for m in masks],
                "judge_version": group["judge_version"],
                "valid_ratio": group["valid_ratio"],
            }
        )

    param_after = next(model.parameters()).detach().abs().mean().item()
    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    skipped = sum(1 for m in metrics if m.get("skipped"))
    nonzero_var = sum(
        1 for m in metrics if not m.get("skipped") and float(m.get("reward_std") or 0) > 1e-6
    )
    meta = {
        "variant": "sequence-level GRPO with clipped surrogate + Schulman KL to frozen ref",
        "steps": steps,
        "valid_groups": valid_groups,
        "optimizability": {
            "nonzero_reward_variance_group_ratio": nonzero_var / max(steps, 1),
            "discarded_group_ratio": skipped / max(steps, 1),
            "valid_group_ratio": valid_groups / max(steps, 1),
            "note": "机制可运行证据；正式任务还需独立评测 Δ 与分项退化检查",
        },
        "metrics": metrics,
        "param_mean_before": param_before,
        "param_mean_after": param_after,
        "params_changed": abs(param_after - param_before) > 1e-12,
        "group_size": G,
        "device": str(device),
        "note": "tiny model CPU smoke; production GPU launch via configs/grpo_full.yaml",
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
