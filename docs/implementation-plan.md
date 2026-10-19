# AutoEvidence 实现计划

> 日期: 2026-10-19 | 环境: macOS arm64 (M2 Pro), Python 3.10, uv, PyTorch 2.9.1 + MPS；无 NVIDIA GPU；磁盘约 16GB 可用。

## 目标

从零实现：工单 → 证据包 → 教师笔录 → 引证式 SFT → GRPO → 评测/消融 → 推理服务。

## 模块与依赖

| 阶段 | 模块 | 依赖组 | 验收命令 |
|------|------|--------|----------|
| 1 | schemas / data / standards / CLI / 规则基线 | core | `uv run pytest tests/test_schemas.py tests/test_data.py -q` + `uv run autoevidence smoke` |
| 2 | evidence / synthesis / 隔离 / 训练集 | core (+vision 可选) | `uv run autoevidence build-dataset --synthetic` |
| 3 | training SFT / 分段加权 | train | `uv run autoevidence train-sft --config configs/sft_tiny.yaml` |
| 4 | rewards / GRPO | train | `uv run autoevidence train-grpo --config configs/grpo_tiny.yaml` |
| 5 | evaluation / api / 展示 | serve | `uv run autoevidence eval` + `uv run autoevidence serve` |

## 资源约束与策略

- **GPU**: 本地无 CUDA；正式训练脚本与配置按 Linux H800 写好，标为「未在本机验证」。
- **磁盘**: 不下载 35B/27B；tiny 验证用本地随机初始化 GPT-2 级小模型（~5M 参数）。
- **视觉**: SigLIP2 提供真实适配器；无权重时回退到确定性哈希嵌入（明确标注）。
- **教师/裁判**: OpenAI 兼容适配器 + 规则教师（synthetic 路径默认，不调用付费 API）。

## 模型兼容性（已核验）

| 角色 | 文档声称 | 官方核实 (2026-10-19) | 本仓库默认 |
|------|----------|----------------------|------------|
| 学生 | Qwen3.6-35B-A3B | HF `Qwen/Qwen3.6-35B-A3B` 存在，Apache-2.0，原生多模态 MoE | `configs/models.yaml` 正式项；本地用 `tiny-gpt2` |
| 教师 | Qwen3.6-27B | 官方新闻 2026-04-22 发布 | OpenAI 兼容 ID 可配置 |
| 裁判 | qwen3.6-plus | 百炼 API 商业模型 | `rule_judge` / 兼容接口 |
| 视觉 | SigLIP2 | `google/siglip2-base-patch16-224` | 适配器 + 哈希回退 |

## 数据边界

- 全部示例与评测 fixture 标注 `source: synthetic`。
- 参考文档指标仅作目标，不写入默认报告实测值。
- 安全门禁 160 条漏 4 → 召回按 **156/160 = 97.5%** 计数；代码用真实分子分母。

## 完成定义

每阶段更新 `docs/progress.md`；最终分别标明：已编码 / 已验证 / 待 GPU / 未完成。
