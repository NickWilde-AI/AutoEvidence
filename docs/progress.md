# 进度

> 更新：safety_seal 封存协议落地

## 完成度（统一口径）

1. **工程闭环** — 已验证（CPU + 合成 + tiny）
2. **正式训练适配** — 已预留配置/入口，目标环境未验证
3. **业务效果** — 未验证

`nonzero_reward_variance_group_ratio` 只说明奖励有差异，**不表示符合真实质量优劣**。

## 本轮交付

| 项 | 位置 |
|----|------|
| 封存清单（ID/哈希/标签·条款版本/日期） | `data/seals/seal_v0/` + `data/seal.py` |
| 使用规则与失败处置 | `docs/safety-seal-protocol.md` |
| 正式验收 `strict_ok` | 协议 + `leakage_check.strict_ok` |
| 独立评测边界 | 同上；金标状态 `template_pending_human_review` |
| 奖励人工抽查 | `autoevidence reward-spotcheck` |

```bash
uv run autoevidence seal-safety --version seal_v0
uv run autoevidence verify-seal --dir data/seals/seal_v0
uv run autoevidence reward-spotcheck
uv run pytest -q
```
