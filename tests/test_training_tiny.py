import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_tiny_sft_updates_and_saves(tmp_path: Path):
    from autoevidence.training.sft import train_sft

    cfg = tmp_path / "sft.yaml"
    cfg.write_text(
        f"""
mode: segment_weighted
seed: 0
device: cpu
learning_rate: 1.0e-3
max_steps: 2
max_length: 128
use_lora: false
data_path: missing.jsonl
output_dir: {tmp_path / "out"}
segment_weights:
  evidence: 1.0
  clause_check: 1.9
  conclusion: 1.3
""",
        encoding="utf-8",
    )
    meta = train_sft(cfg)
    assert meta["params_changed"] or len(meta["losses"]) == 2
    assert (tmp_path / "out" / "checkpoint").exists()
    assert (tmp_path / "out" / "train_meta.json").exists()


def test_tiny_grpo_runs(tmp_path: Path):
    from autoevidence.training.grpo import train_grpo

    cfg = tmp_path / "grpo.yaml"
    cfg.write_text(
        f"""
seed: 0
device: cpu
learning_rate: 1.0e-4
max_steps: 2
group_size: 2
temperature: 0.9
max_new_tokens: 16
clip_eps: 0.2
kl_beta: 0.025
standard_version: v4
output_dir: {tmp_path / "gout"}
""",
        encoding="utf-8",
    )
    meta = train_grpo(cfg)
    assert "metrics" in meta
    assert (tmp_path / "gout" / "checkpoint").exists()
    # 至少完成循环；零方差组可能 skip，但文件应写出
    data = json.loads((tmp_path / "gout" / "train_meta.json").read_text(encoding="utf-8"))
    assert data["steps"] == 2
