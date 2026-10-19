"""DeepSeek 官方 OpenAI 兼容接入。

依据（2026-10-19 查阅）：
- https://api-docs.deepseek.com/  base_url=`https://api.deepseek.com`
- 模型：`deepseek-flash` / `deepseek-v4-pro`
- JSON：`response_format={"type":"json_object"}`（见 JSON Output 指南）
"""

from __future__ import annotations

from pathlib import Path

from autoevidence.clients.openai_compat import OpenAICompatClient, require_api_key

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DOCS_REF = {
    "first_call": "https://api-docs.deepseek.com/",
    "json_mode": "https://api-docs.deepseek.com/guides/json_mode/",
    "checked_at": "2026-10-19",
}


def build_deepseek_client(
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 90.0,
    cache_dir: str | Path | None = None,
    max_retries: int = 3,
) -> OpenAICompatClient:
    key = require_api_key("DEEPSEEK_API_KEY")
    return OpenAICompatClient(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=key,
        model=model or DEFAULT_MODEL,
        timeout=timeout,
        max_retries=max_retries,
        cache_dir=cache_dir,
        provider="deepseek",
        min_interval_s=0.4,
    )
