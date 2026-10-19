# 可复现运行记录（本机工程闭环）

| 字段 | 值 |
|------|-----|
| 日期 | 2026-10-19 |
| 主机 | macOS 15.3.1 arm64，Apple M2 Pro |
| GPU | 无 NVIDIA；未跑 CUDA 正式训练 |
| Python | 3.10.13（pyenv）；uv 管理 `.venv` |
| 代码树 | AutoEvidence 本地工作区（无强制要求 git remote） |
| 数据版本 | `synthetic` smoke 生成；`data/synthetic/manifest.json` + `splits/manifest.json` |
| 条款版本 | `v4`（`data/standards/v4.json`，source=synthetic） |
| 配置 | `configs/sft_tiny.yaml`、`configs/grpo_tiny.yaml` |
| Seed | 42（配置内） |

## 命令与关键输出位置

```bash
uv sync --extra all --python /Users/chenpeng/.pyenv/versions/3.10.13/bin/python3.10
uv run pytest -q
# → 21 passed

uv run autoevidence smoke
# → outputs/reports/smoke_summary.json
# → outputs/reports/smoke_eval.json
# → data/synthetic/{tickets,splits,synthesis,sft_entries}

uv run autoevidence train-sft --config configs/sft_tiny.yaml
# → outputs/checkpoints/sft_tiny/train_meta.json
# → outputs/checkpoints/sft_tiny/checkpoint/

uv run autoevidence train-grpo --config configs/grpo_tiny.yaml
# → outputs/checkpoints/grpo_tiny/train_meta.json（含可优化性统计）
# → outputs/checkpoints/grpo_tiny/checkpoint/

uv run autoevidence eval
# → outputs/reports/eval.json

uv run autoevidence export-cases
# → outputs/reports/e2e_cases.json
```

## 模型产物对应关系

| 产物 | 数据 | 配置 | 评测 |
|------|------|------|------|
| `outputs/checkpoints/sft_tiny/` | synthetic short entries（tiny 强制短样本） | `sft_tiny.yaml` | smoke_eval / 非业务效果 |
| `outputs/checkpoints/grpo_tiny/` | 同上 + 模板/生成混合 rollout | `grpo_tiny.yaml` | train_meta 内组统计 |

**声明**：本记录证明工程闭环可复现，不证明业务效果或 35B 正式训练。

---

## DeepSeek 真实 API 小批量（2026-10-19）

详见 [`docs/deepseek-pilot-run.md`](deepseek-pilot-run.md)。

```bash
export DEEPSEEK_API_KEY=...
uv run autoevidence deepseek-pilot --out outputs/deepseek_pilot
```

- 教师：train 20 条，kept 17，conclusion_mismatch 3，parse_fail 0  
- 裁判：10 对，位置一致 9/9（成功对），1 对解析失败单独计数  
- 未使用 safety_seal；人工复核表待真人填写  
- **不**将本轮表述为模型质量提升
