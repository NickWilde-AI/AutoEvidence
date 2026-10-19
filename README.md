# AutoEvidence

**智能汽车质量分析模型** · Citation-grounded SFT + LLM-as-Judge GRPO

面向车企售后质量工单：把对话、车辆信号、故障照片与服务历史收成可复核证据包，再输出质量事件判定、问题类型、处置等级、条款对照与分流建议。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English · [完成度](docs/completion-levels.md) · [验收矩阵](docs/acceptance-matrix.md) · [架构](docs/architecture.md) · [训练](docs/training.md)

仓库：[github.com/NickWilde-AI/AutoEvidence](https://github.com/NickWilde-AI/AutoEvidence)  
同系列公开工程：[AutoCare-Guard-ML](https://github.com/NickWilde-AI/AutoCare-Guard-ML)（服务风险决策平台）

---

## 公开边界

本仓库是**可运行的算法工程原型**，不含真实业务工单、车主身份、生产权重与线上效果数字。

| 层级 | 状态 |
|------|------|
| 工程闭环（CPU + 合成数据 + tiny 模型） | 可公开验证 |
| 正式训练适配（Qwen3.6-35B-A3B / vLLM / DeepSpeed） | 配置与入口已预留，需自备 GPU 与数据 |
| 业务效果 | 未验证；设计文档中的指标仅为参考目标 |

> 哈希视觉回退证明接口可用，不证明视觉识别能力。tiny 参数更新证明训练机制可运行，不证明真实任务有效。

---

## 为什么需要它

规则引擎 + 小模型常见漏判与误伤；零样本大模型不稳定，且难给出可复核引证。

```
原始工单（对话 / 信号 / 照片 / 服务历史）
        │
        ▼
   多源证据定位  →  EvidencePack
        │
        ▼
 教师笔录合成 + 三道过滤（条款存在 / 结论对齐 / 引证越界）
        │
        ▼
 引证式分段加权 SFT  →  引证奖励 GRPO
        │
        ▼
 独立评测 / 消融 / 推理服务与分流
```

---

## 能力一览

| 模块 | 内容 |
|------|------|
| 数据协议 | 工单 / 证据 / 金标硬约束；车辆隔离；train/val/test/safety_dev/safety_seal 契约 |
| 证据定位 | 症状句锚定、信号窗口、图相似度（SigLIP2 适配 / 哈希回退） |
| 教师合成 | 三段式笔录；条款改写变体；可恢复批处理 |
| 训练 | 普通 SFT / 分段加权 SFT；真实 GRPO（rollout、logprob、组内优势、clip、KL） |
| 奖励 | 判定 / 类型 / 处置 / 引证 + 编造·误伤·安全漏放惩罚；裁判失败不默认真高分 |
| 评测服务 | 分子分母显式指标；人审双口径；FastAPI `/analyze` |

---

## 快速开始（无需 GPU / 付费 API）

```bash
git clone https://github.com/NickWilde-AI/AutoEvidence.git
cd AutoEvidence

uv sync --extra all
# 或: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"

uv run pytest -q
uv run autoevidence smoke
uv run autoevidence train-sft --config configs/sft_tiny.yaml
uv run autoevidence train-grpo --config configs/grpo_tiny.yaml
uv run autoevidence eval
uv run autoevidence export-cases
uv run autoevidence serve --port 8000
```

| 命令 | 作用 |
|------|------|
| `smoke` | 合成数据 → 定位 → 合成 → 评测 |
| `train-sft` / `train-grpo` | tiny 短步，验证参数更新 |
| `export-cases` | 正常 / 事件 / 难负例 / 证据不足 / 安全转人工案例 |
| `serve` | `GET /health` · `POST /analyze` · `GET /demo` |

---

## 正式训练路径（需自备资源）

| 阶段 | 配置 | 说明 |
|------|------|------|
| 全参分段加权 SFT | `configs/sft_full.yaml` | Qwen3.6-35B-A3B + DeepSpeed ZeRO-3 |
| GRPO | `configs/grpo_full.yaml` | G=7、vLLM rollout、参考 KL |
| 模型登记 | `configs/models.yaml` | 官方 ID 与替代说明 |

准确表述：**已预留正式模型训练方案，目标环境适配与效果待验证。**

---

## 项目结构

```text
AutoEvidence/
├── configs/                 # tiny / full SFT·GRPO、models、DeepSpeed
├── data/examples/           # 示例工单
├── data/standards/          # 合成条款库 v4
├── docs/                    # 架构、契约、验收、运行记录
├── scripts/
├── src/autoevidence/        # 核心包
├── tests/
├── pyproject.toml
└── README.md
```

敏感数据、权重、checkpoint、`.env` 默认不提交。见 `.gitignore`、`.env.example`。

---

## 文档

| 主题 | 文档 |
|------|------|
| 完成度层级 | [completion-levels](docs/completion-levels.md) |
| 需求验收矩阵 | [acceptance-matrix](docs/acceptance-matrix.md) |
| 可复现运行记录 | [run-record](docs/run-record.md) |
| 数据与评测契约 | [data-and-evaluation](docs/data-and-evaluation.md) |
| 架构 | [architecture](docs/architecture.md) |
| 训练与奖励 | [training](docs/training.md) |
| 模型兼容性 | [model-compatibility](docs/model-compatibility.md) |
| 进度 | [progress](docs/progress.md) |

设计参考材料（非已跑通实验证明）：`20261018_risk.md`、`20261018_risk_QA.md`、`从零到一开发提示词.md`。

---

## 与 AutoCare-Guard-ML 的关系

| 仓库 | 定位 |
|------|------|
| [AutoEvidence](https://github.com/NickWilde-AI/AutoEvidence) | 质量事件分析：引证笔录、分段加权 SFT、引证奖励 GRPO |
| [AutoCare-Guard-ML](https://github.com/NickWilde-AI/AutoCare-Guard-ML) | 服务风险决策：多证据 Judge、路由、审计与企业级治理骨架 |

二者同属智能汽车售后智能方向，协议与场景不同，可对照阅读。

---

MIT License · [LICENSE](LICENSE)
