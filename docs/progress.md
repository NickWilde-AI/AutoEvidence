# 进度

> 更新时间：2026-10-19（框架审查后补强）

## 完成度（统一口径）

见 `docs/completion-levels.md`：

1. **工程闭环** — 已验证（CPU + 合成 + tiny）
2. **正式训练适配** — 已预留配置/入口，目标环境未验证
3. **业务效果** — 未验证

## 框架审查后新增

- 数据使用契约：`train/val/test/safety_dev/safety_seal`（`docs/data-and-evaluation.md`）
- 泄漏扩展：近重复文本 / 共享图片 → `strict_ok`；硬泄漏仍用 `ok`
- 溯源：`Provenance` + SFT entry `provenance` 字段
- 验收矩阵 / 运行记录 / 完成度层级文档
- 人审双口径：`safety_intercept_rate` vs `incomplete_auto_rate`
- 端到端案例：`autoevidence export-cases` → `outputs/reports/e2e_cases.json`
- GRPO `optimizability` 统计写入 `train_meta.json`

## 已验证（本机）

| 命令 | 结果 |
|------|------|
| `uv run pytest -q` | 以最近一次为准（含 routing/cases） |
| `uv run autoevidence smoke` | 工程闭环 |
| tiny SFT / GRPO | 参数更新 + GRPO 可优化性字段 |
| `uv run autoevidence export-cases` | 案例集 |

## 下一步（正式训练之前）

1. 封存一版 `safety_seal` 并锁定版本号
2. 独立评测边界：避免规则教师/奖励/基线同偏好自嗨
3. 有 GPU 后再跑 `sft_full` / `grpo_full`，并报告前后评测 Δ
