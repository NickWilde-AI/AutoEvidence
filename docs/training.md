# 训练

## SFT

- 配置：`configs/sft_tiny.yaml`（已验证）/ `sft_plain.yaml` / `sft_full.yaml`（待 GPU）
- 损失：`segment_weighted_loss`；权重默认 evidence=1.0, clause_check=1.9, conclusion=1.3
- 仅目标 token 计损失；权重按 tokenizer offset 映射，不用字符串长度代替
- LoRA 与全参差异：tiny 可开 `use_lora`；正式全参见 `sft_full.yaml`

## 奖励公式

```
total = r_verdict + r_fault_type + r_disposal + r_citation + r_penalty + r_judge
```

- 误伤：`false_block=-0.4`
- 安全漏放（含判成技术排查躲避）：`miss_critical=-0.7`
- 编造条款：`fabricate=-0.6`
- 裁判失败 → 0 分，不默认真高分

## GRPO

- 变体：序列级优势归一化 + PPO-style clip + Schulman KL 到冻结参考模型
- 零方差 / 奖励全同组丢弃
- tiny：`configs/grpo_tiny.yaml`；正式：`grpo_full.yaml`（vLLM rollout，本机未跑）

### 可优化性报告字段（`train_meta.json`）

- `optimizability.nonzero_reward_variance_group_ratio`
- `optimizability.discarded_group_ratio`
- `metrics[].reward_components_mean` / `valid_ratio`
- 正式实验另需：GRPO 前后独立评测 Δ；检查奖励升但漏判/误伤/引证变差

规则教师 + 规则奖励 + 规则评测可能共享偏好；闭环一致 ≠ 业务正确。

## 恢复

checkpoint 目录含 `train_meta.json` 与 HF `save_pretrained` 权重；GRPO 可从 SFT checkpoint 启动。
