# 模型兼容性（核实日期：2026-10-19）

| 角色 | 文档名称 | 核实结果 | 本仓库用法 |
|------|----------|----------|------------|
| 学生 | Qwen3.6-35B-A3B | HF `Qwen/Qwen3.6-35B-A3B` 存在；Apache-2.0；原生多模态 MoE；Transformers/vLLM/SGLang | `configs/models.yaml` 正式项；本地 tiny GPT-2 随机初始化 |
| 教师 | Qwen3.6-27B | 官方 GitHub 新闻 2026-04-22 宣布发布 | OpenAI 兼容可配置；默认 `RuleTeacher` |
| 裁判 | qwen3.6-plus | 商业 API（百炼类） | `RuleJudge` / OpenAI 兼容 |
| 视觉 | SigLIP2 | `google/siglip2-base-patch16-224` | `SigLIP2Encoder`；无权重时 `HashVisionEncoder` |

## 替代说明

- 无法加载 SigLIP2 时使用哈希嵌入，**明确标注**，不宣称视觉实验完成。
- 无法调用教师/裁判 API 时使用规则实现完成链路验证。
- tiny 训练验证 ≠ 正式大模型实验完成。
