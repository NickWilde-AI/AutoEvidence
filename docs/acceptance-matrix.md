# 需求验收矩阵

对照 `从零到一开发提示词.md`。状态：`done` 已编码并有本机证据；`partial` 有实现但资源/数据受限；`pending` 待目标环境或真实数据。

| 要求 | 模块 | 验证证据 | 当前限制 |
|------|------|----------|----------|
| 统一数据协议与标签硬约束 | `schemas/` | `tests/test_schemas.py` | — |
| 合成样本覆盖多场景且标注 synthetic | `data/synthetic.py` | `autoevidence smoke` → `data/synthetic/manifest.json` | 非真实业务 |
| 车辆/事件/父样本隔离与泄漏检查 | `data/split.py` | smoke `leakage_ok`；`tests/test_data.py` | 近重复/共图检查已加，规模仍为合成 |
| 数据使用契约 train/val/test/safety | `docs/data-and-evaluation.md` | 文档 + SplitName | — |
| safety_seal 封存清单与哈希 | `data/seal.py` `data/seals/seal_v0` | `verify-seal` ok | 金标仍待独立人工复核 |
| 正式泄漏门槛 strict_ok | `leakage_check` | seal 时写入 full_corpus_leakage | 生产近重复须逐项解释 |
| 奖励人工抽查入口 | `rewards/spotcheck.py` | `reward-spotcheck` | 抽查结果需人工填写 |
| DeepSeek 真实 API 教师/裁判 | `clients/` `experiments/deepseek_pilot.py` | `docs/deepseek-pilot-run.md`；PILOT_REPORT | 同一模型非独立金标；业务效果未验证 |
| 多源证据定位 + 规则回退 | `evidence/` | smoke packs；locator 单测路径 | SigLIP2 默认可能哈希回退 |
| 条款库与三道过滤 | `standards/` `synthesis/` | `tests/test_citation.py`；synthesis stats | 规则教师为主 |
| 换配增广与难负例过采样 | `data/augment.py` | smoke augmented_count | 歧义标签需人审标记 |
| 普通/分段加权 SFT | `training/sft.py` | tiny：loss↓ `params_changed` | 正式 35B 未跑 |
| 真实 GRPO（非空循环） | `training/grpo.py` | tiny：`valid_groups=3` loss/grad | 正式 GPU/vLLM 未跑 |
| 奖励漏洞与裁判失败不默认真高分 | `rewards/` | `tests/test_rewards.py` | 规则裁判 ≠ 业务裁判 |
| 评测指标分子分母 / 空集不可用 | `evaluation/metrics.py` | `tests/test_metrics.py`（含 97.5%） | 合成评测集 |
| 人审双口径 | `evaluation/routing_metrics.py` | 单测 + 报告字段 | 需在正式评测启用 |
| 消融登记（区分重训/推理） | `evaluation/ablation.py` | `autoevidence ablations` | 多数消融 `not_run` |
| 推理不读金标 + 分流 | `inference/` `api/` | API 测试丢弃 gold | demo 模式非大模型 |
| 文档与配置交付 | `docs/` `configs/` | 本矩阵 + run-record | 正式训练适配待验证 |

## 完成度层级（统一口径）

| 层级 | 状态 |
|------|------|
| 工程闭环 | **已验证**：CPU + 合成数据 + tiny 模型 |
| 正式训练适配 | **已预留方案与入口**；资源适配/稳定性/效果待目标环境验证。**不得**表述为「已支持 35B 正式训练完成」 |
| 业务效果 | **未验证**：需真实数据与独立评测 |

哈希回退证明视觉**接口**可用，不证明视觉识别能力。  
tiny 参数更新证明训练**机制**可运行，不证明真实任务有效。
