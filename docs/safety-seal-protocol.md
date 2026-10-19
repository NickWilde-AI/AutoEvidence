# safety_seal 封存与独立评测协议

统一口径：**工程闭环已验证；正式训练已预留；业务效果未验证。**

`nonzero_reward_variance_group_ratio=1.0` 只说明候选奖励有差异，**不表示差异符合真实质量优劣**。正式训练前须人工抽查「高奖励是否确实优于低奖励」（`autoevidence reward-spotcheck`）。

---

## 1. 封存清单字段

| 字段 | 含义 |
|------|------|
| `seal_version` | 封存版本号，如 `seal_v0` |
| `sealed_at` | UTC 封存日期 |
| `sample_ids` / `samples[].content_sha256` | 样本 ID 与内容哈希 |
| `tickets_jsonl_sha256` | 整文件哈希 |
| `label_version` | 标签版本（独立复核前标为 template） |
| `standard_version` / `clause_file_hash` | 条款版本与条款文件哈希 |
| `manifest_sha256` | 清单自洽哈希 |

产物目录：`data/seals/<seal_version>/`

```bash
uv run autoevidence seal-safety --version seal_v0
uv run autoevidence verify-seal --dir data/seals/seal_v0
```

---

## 2. 使用规则（硬约束）

封存集**不得**用于：

1. 训练（SFT / GRPO / 增广父样本）
2. 调奖励、调阈值、改规则
3. 选 checkpoint / 早停

**仅允许**：正式候选配置冻结后做一次门禁评测。

---

## 3. 独立评测边界

| 要求 | 说明 |
|------|------|
| 金标独立复核 | 不得完全复用教师输出或奖励规则自动生成答案作为最终金标 |
| 配置冻结 | 评测前锁定模型、提示词、阈值、奖励版本 |
| 规则同偏好风险 | 规则教师 + 规则奖励 + 规则基线可能共享偏好；闭环一致 ≠ 业务正确 |
| 合成 seal | 本仓库 `seal_v0` 为工程样例；`independent_gold_status=template_pending_human_review` |

---

## 4. 失败处置

1. 门禁失败 → **阻断发布**（`block_release=true`）
2. 该封存版本标记为 **已暴露回归集**（`FAILURE_RECORD.json`）
3. 根据失败样本改进后，**下一次验收必须使用新的 `seal_version`**
4. 旧版保留，用于防回归，不再作为唯一发布门槛

```bash
uv run autoevidence mark-seal-failed --dir data/seals/seal_v0 --reason "critical_recall_below_threshold"
```

---

## 5. 泄漏门槛

| 验收级别 | 要求 |
|----------|------|
| 工程 smoke | `leakage_check.ok == true`（硬泄漏：车辆/事件/父样本/安全进 train） |
| **正式验收** | **`strict_ok == true`** |
| 无法消除的共图 / 近重复 | 必须在清单中 **逐项解释**（`justifications`），否则不通过正式验收 |

---

## 6. 与完成度层级的关系

| 动作 | 证明什么 | 不证明什么 |
|------|----------|------------|
| 生成 `seal_v0` + 哈希可复核 | 封存流程可运行 | 业务安全门禁通过 |
| 奖励人工抽查模板 | 有质量审查入口 | 奖励已对齐真实优劣 |
| `strict_ok` 门槛写入协议 | 正式验收标准明确 | 当前合成语料已满足业务严格隔离 |
