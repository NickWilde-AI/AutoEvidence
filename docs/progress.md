# 进度

## 完成度（统一口径）

1. **工程闭环** — 已验证（CPU + 合成 + tiny）
2. **真实 API 链路** — DeepSeek 小批量已实测（见 `docs/deepseek-pilot-run.md`）；**≠ 质量提升**
3. **正式训练适配** — 已预留，目标环境未验证
4. **业务效果** — 未验证

## DeepSeek 验收（本轮）

- 入口：`deepseek-ping` / `deepseek-teacher` / `deepseek-judge` / `deepseek-pilot`
- 教师 train 20：kept 17，conclusion_mismatch 3
- 裁判 10 对：成功 9，顺序一致 9/9；人工表待填
- 未触碰 `safety_seal`
