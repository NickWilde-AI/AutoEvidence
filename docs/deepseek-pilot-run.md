# DeepSeek API 小批量验收运行记录

> 口径：工程闭环已验证；**真实 API 链路以本轮实测为准**；正式训练已预留；业务效果未验证。  
> **API 调用成功 ≠ 模型质量提升。** 教师与裁判为同一 `deepseek-flash` 时**不构成独立金标**。

## 官方依据（查阅 2026-10-19）

| 项 | 值 |
|----|-----|
| base_url | `https://api.deepseek.com`（[First API Call](https://api-docs.deepseek.com/)） |
| 模型 | `deepseek-flash`（实际返回 `model=deepseek-flash`） |
| 结构化输出 | `response_format={"type":"json_object"}`（[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)） |
| 鉴权 | 环境变量 `DEEPSEEK_API_KEY`（未写入仓库） |

## 复现

```bash
export DEEPSEEK_API_KEY=...   # 本机，勿提交
uv run autoevidence deepseek-ping --out outputs/deepseek_pilot
uv run autoevidence deepseek-teacher --n 20 --out outputs/deepseek_pilot
uv run autoevidence deepseek-judge --out outputs/deepseek_pilot
# 或一次性: uv run autoevidence deepseek-pilot
```

产物（默认 gitignore `outputs/`）：`outputs/deepseek_pilot/PILOT_REPORT.json` 等。

## 本轮实测摘要

| 项目 | 结果 |
|------|------|
| Ping | ok；返回 model=`deepseek-flash` |
| 教师 | train-only **20** 条；**kept 17 / filter reject 3 / parse_fail 0**；通过率 **85%** |
| 过滤原因 | 全部为 `conclusion_mismatch`（3） |
| 裁判 | 10 对；成功 9；parse_fail 1；顺序交换一致率 **9/9=1.0** |
| 与规则分歧 | `insufficient_but_certain` 及其变体（待人工判断） |
| Token（汇总约） | prompt≈48k + completion≈40k；粗算 **~$0.018**（Flash 公开价 cache-miss，会变动） |
| safety_seal | **未触碰** |

## 人工复核

填写 `outputs/deepseek_pilot/human_review_sheet.json` 中 `human_review.*`。  
**禁止模型代填为已通过。**

## 下一轮小规模实验条件

**具备条件（有限）**：真实 API 连通、教师 train 过滤链路、裁判对比与位置一致性可测、失败不计高分。  
**不具备**：业务效果、独立金标、正式 GPU 训练验收。
