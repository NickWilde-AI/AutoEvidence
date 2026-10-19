# 数据与评测

## 标签定义

- `quality_event` + `fault_type` ∈ {电池与高压系统, 制动与转向, 动力总成与传动, 电子电气与软件}；正常时 `fault_type=无`
- `disposal` ∈ {紧急复核, 技术排查, 人审, 归档}
- 约束由 Pydantic 校验，不靠提示词

## 数据使用契约（固定）

| 分区 | 允许用途 | 禁止用途 |
|------|----------|----------|
| `train` | 增广、教师合成入库、SFT、GRPO rollout、难例挖掘回流源 | 最终效果宣称、安全门禁封存验收 |
| `val` / `validation` | 选 checkpoint、调阈值、调奖励权重、教师质量抽检 | 写入最终对外效果表；不得把调参结果当独立测试 |
| `test` | 配置冻结后的最终效果评测 | 回流训练、按逐条错误改规则/奖励 |
| `safety_dev` | 安全相关开发与阈值摸底（可迭代） | 当作发布硬门槛的唯一证据 |
| `safety_seal` / 最终封存安全集 | 发布前独立安全验收；失败则阻断发布 | 训练、难例回流、反复按结果改奖励/阈值 |

说明：

- **「不回流训练」不够。** 若反复根据某安全集结果修改奖励、阈值或规则，该集已参与调参，不得再称为独立封存验收。
- 本仓库 synthetic 切分中：`safety_gate` 默认视为 **`safety_dev` 候选**；真正的 `safety_seal` 见 `docs/safety-seal-protocol.md` 与 `data/seals/`。
- **正式验收**要求全集 `leakage_check.strict_ok=true`；无法消除的共图/近重复必须逐项解释。
- 奖励方差比 ≠ 质量优劣；正式训练前执行 `autoevidence reward-spotcheck`。

## 隔离与泄漏

切分前完成。按 `vehicle_id_hash` 分组是**必要非充分**条件，还需检查：

| 检查项 | 要求 |
|--------|------|
| 车辆 | 同一车辆不得跨 train/val/test/safety |
| 事件 `event_id` | 同一事件不得跨区 |
| 增广父样本 `parent_ids` | 子样本必须与父样本同区；冲突记泄漏 |
| 近重复文本 | 对话指纹跨区命中记 `near_dup_cross_split` |
| 共享图片 path/`image_id` | 跨区共图记 `shared_image_cross_split` |
| 安全封存 | `safety_seal` / `safety_gate` 车辆不得出现在 train |

实现：`autoevidence.data.split.leakage_check`。

换配样本：仅允许训练分区父样本；自动标签默认 `label_confidence=ambiguous_needs_human`。

## 版本与溯源

每条训练条目应能追溯（见 `Provenance` / SFT entry 字段）：

- `ticket_id` → `evidence_pack_id` → `sample_id`
- `standard_version`、`teacher_version`、`prompt_version`
- `filter_result`（kept / reject_reasons）
- `split`、`parent_ids`、`data_version`

一次模型产物应对应：`data_version` + `config_path` + `code_git`（若有）+ `seed` + `metrics_path`（见 `docs/run-record.md`）。

## 指标分母

每项输出 `numerator/denominator/available`。空集 `available=false`，不伪装 0% 或 100%。

- 问题类型准确率：仅金标质量事件
- 关键安全召回（模型侧）：金标需紧急复核且模型 `disposal==紧急复核`
- 160 漏 4 → **156/160 = 97.5%**（代码按真实计数）
- 解析失败 / 拒答 / 证据不足单独统计，不丢弃抬指标

## 人审分流双口径（必须分开）

| 指标 | 含义 | 不可混用 |
|------|------|----------|
| `safety_intercept_rate` | 金标安全事件被路由到「紧急复核/人审确认」队列的比例 | **安全拦截成功** |
| `incomplete_auto_rate` | 因证据不足、解析失败、策略不确定而转人审的比例 | **自动诊断未完成** |

禁止：靠大量转人工抬高「安全分」。报告中两者并列，分母各自定义。

## 独立评测边界

- 规则教师 / 规则奖励 / 规则基线评测可能共享偏好 → **闭环内一致 ≠ 业务正确**
- 最终效果需：冻结配置后的 `test` + 封存 `safety_seal`；有条件时加独立人工标注或独立裁判样本
- 参考文档 F1/误伤等数字仅为设计目标，不得写入默认实测报告

## GRPO 可优化性证据（训练报告必含）

- 非零奖励方差组占比、丢弃组占比
- 奖励分项分布、裁判失败率 / 有效样本比例
- （正式实验）GRPO 前后独立评测 Δ；是否出现奖励升但漏判/误伤/引证变差
