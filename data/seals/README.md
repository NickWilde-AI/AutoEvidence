# 封存集目录

| 版本 | 用途 |
|------|------|
| `seal_v0/` | 工程样例封存（合成 critical 样本）；**未**经独立人工金标复核 |

协议全文：[`docs/safety-seal-protocol.md`](../../docs/safety-seal-protocol.md)

```bash
uv run autoevidence seal-safety --version seal_v0
uv run autoevidence verify-seal --dir data/seals/seal_v0
```
