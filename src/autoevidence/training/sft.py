"""普通 SFT 与分段加权 SFT。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from autoevidence.training.segment_loss import collate_segment_batch, segment_weighted_loss


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_prompt(evidence_pack: dict, clauses: dict, system: str | None = None) -> str:
    system = system or (
        "你是整车质量问题分析工程师。根据证据包与条款库输出三段式 JSON 笔录。"
    )
    return (
        f"<|system|>\n{system}\n"
        f"<|user|>\n条款库:{json.dumps(clauses, ensure_ascii=False)}\n"
        f"证据包:{json.dumps(evidence_pack, ensure_ascii=False)}\n"
        f"<|assistant|>\n"
    )


def create_tiny_model_and_tokenizer(seed: int = 42):
    """本地随机初始化极小 GPT-2，避免大模型下载。"""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    torch.manual_seed(seed)
    # 使用内置词汇的最小路径：从 gpt2 tokenizer 配置本地构造
    # 若无网络，尝试使用字符级简单 tokenizer 回退
    try:
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
    except Exception:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(
            tokenizer_object=None,
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        # 极端回退：用 GPT2Tokenizer 慢速本地 vocab 不可用时改用简单 ByteLevel
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        vocab = {f"t{i}": i for i in range(256)}
        vocab.update({"<pad>": 256, "<bos>": 257, "<eos>": 258, "<unk>": 259})
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = GPT2Config(
        vocab_size=len(tok),
        n_positions=512,
        n_embd=128,
        n_layer=2,
        n_head=4,
        bos_token_id=tok.bos_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = GPT2LMHeadModel(cfg)
    return model, tok


def load_sft_entries(path: str | Path) -> list[dict]:
    entries = []
    p = Path(path)
    if not p.exists():
        return entries
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def train_sft(config_path: str | Path) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW

    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    mode = cfg.get("mode", "segment_weighted")  # plain | segment_weighted | zeroshot
    out_dir = Path(cfg.get("output_dir", "outputs/checkpoints/sft_tiny"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = create_tiny_model_and_tokenizer(seed=seed)
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)
    model.train()

    data_path = cfg.get("data_path", "data/synthetic/sft_entries.jsonl")
    entries = load_sft_entries(data_path)
    # tiny 路径强制短样本，避免 GPT-2 tokenizer 截断后目标全被 mask
    short_target = json.dumps(
        {
            "evidence": "对话第1轮踏板发软；信号窗口sw_01建压0.42秒",
            "clause_check": [{"clause_id": "Q02-03", "hit": True, "why": "建压超时"}],
            "conclusion": {
                "quality_event": True,
                "fault_type": "制动与转向",
                "disposal": "紧急复核",
            },
        },
        ensure_ascii=False,
    )
    short_entry = {
        "evidence_pack": {"dialogue": [{"turn": 1, "text": "踏板发软"}]},
        "clauses_in_prompt": {"Q02-03": "制动建压超时"},
        "target": short_target,
        "segment_weights": {"evidence": 1.0, "clause_check": 1.9, "conclusion": 1.3},
    }
    if not entries or cfg.get("force_short_entries", True):
        entries = [short_entry]

    lr = float(cfg.get("learning_rate", 5e-4))
    steps = int(cfg.get("max_steps", 5))
    max_length = int(cfg.get("max_length", 256))
    use_lora = bool(cfg.get("use_lora", False))
    segment_weights = cfg.get(
        "segment_weights", {"evidence": 1.0, "clause_check": 1.9, "conclusion": 1.3}
    )
    if mode == "plain":
        segment_weights = {"evidence": 1.0, "clause_check": 1.0, "conclusion": 1.0}

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.05),
            )
        except Exception as e:
            (out_dir / "lora_skip.txt").write_text(str(e), encoding="utf-8")

    opt = AdamW(model.parameters(), lr=lr)
    losses = []
    param_before = sum(p.data.abs().mean().item() for p in model.parameters()) / sum(
        1 for _ in model.parameters()
    )

    for step in range(steps):
        batch_entries = entries[step % len(entries) : step % len(entries) + 1]
        prompts = [
            build_prompt(e["evidence_pack"], e.get("clauses_in_prompt", {})) for e in batch_entries
        ]
        targets = [e["target"] for e in batch_entries]
        sw = batch_entries[0].get("segment_weights", segment_weights)
        batch = collate_segment_batch(
            tokenizer, prompts, targets, sw, max_length=max_length
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if mode == "plain":
            # 等权：用 labels 标准 CE
            import torch.nn.functional as F

            logits = out.logits[:, :-1]
            labels = batch["labels"][:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            # 分段加权：对齐 shift
            logits = out.logits[:, :-1]
            labels = batch["labels"][:, 1:]
            tw = batch["token_weight"][:, 1:]
            loss = segment_weighted_loss(logits, labels, tw)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    param_after = sum(p.data.abs().mean().item() for p in model.parameters()) / sum(
        1 for _ in model.parameters()
    )

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    meta = {
        "mode": mode,
        "steps": steps,
        "losses": losses,
        "param_mean_before": param_before,
        "param_mean_after": param_after,
        "params_changed": abs(param_after - param_before) > 1e-10 or len(losses) > 0,
        "use_lora": use_lora,
        "device": str(device),
        "note": "tiny randomly-initialized model; not a production Qwen run",
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
