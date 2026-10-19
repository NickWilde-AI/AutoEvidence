#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra all
uv run pytest -q
uv run autoevidence smoke
uv run autoevidence train-sft --config configs/sft_tiny.yaml
uv run autoevidence train-grpo --config configs/grpo_tiny.yaml
uv run autoevidence eval
echo "OK: see outputs/reports and outputs/checkpoints"
