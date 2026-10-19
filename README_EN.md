# AutoEvidence

**Automotive quality analysis model** — citation-grounded SFT + LLM-as-Judge GRPO.

Public engineering prototype for after-sales quality tickets: dialogue, vehicle signals, fault photos, and service history → evidence pack → structured citation record → event / fault type / disposal / routing.

- Repo: https://github.com/NickWilde-AI/AutoEvidence
- Related: https://github.com/NickWilde-AI/AutoCare-Guard-ML

## Public boundary

No real tickets, PII, production weights, or claimed online metrics. Design-doc numbers are **targets only**.

| Level | Status |
|-------|--------|
| Engineering loop (CPU, synthetic, tiny model) | Publicly verifiable |
| Full training adapters (Qwen3.6-35B-A3B, vLLM, DeepSpeed) | Configs reserved; need your GPU/data |
| Business effectiveness | Not validated |

## Quick start

```bash
git clone https://github.com/NickWilde-AI/AutoEvidence.git
cd AutoEvidence
uv sync --extra all
uv run pytest -q
uv run autoevidence smoke
uv run autoevidence serve --port 8000
```

See Chinese [README.md](README.md) for full docs index.

MIT License.
