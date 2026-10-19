# 实验登记

| ID | 描述 | 配置 | 状态 | 结果 |
|----|------|------|------|------|
| smoke-synth | 合成链路端到端 | CLI smoke | 以 progress 为准 | outputs/reports/smoke_* |
| sft-tiny | 分段加权 SFT 短步 | configs/sft_tiny.yaml | 以 progress 为准 | outputs/checkpoints/sft_tiny |
| sft-plain | 普通 SFT | configs/sft_plain.yaml | 可选 | outputs/checkpoints/sft_plain |
| grpo-tiny | GRPO 短步 | configs/grpo_tiny.yaml | 以 progress 为准 | outputs/checkpoints/grpo_tiny |
| sft-full | 正式 35B SFT | configs/sft_full.yaml | **not_run** | - |
| grpo-full | 正式 GRPO | configs/grpo_full.yaml | **not_run** | - |

## 消融

见 `autoevidence.evaluation.ablation.ABLATIONS`。需重训的消融不能只改评测开关。未跑实验一律 `not_run`。

## 参考文档指标

`20261018_risk.md` 中 F1/误伤等数字为设计材料中的参考目标，**不是**本仓库默认报告实测值。
