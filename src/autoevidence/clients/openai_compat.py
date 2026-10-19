"""OpenAI 兼容 Chat Completions 客户端（DeepSeek 等）。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class MissingAPIKeyError(RuntimeError):
    pass


class RateLimitError(RuntimeError):
    pass


class APIRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class EmptyContentError(RuntimeError):
    """JSON Output 偶发空 content（DeepSeek 文档已提示）。"""


def require_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"缺少环境变量 {env_name}。请在本机导出后重试；密钥不会写入代码或报告。"
        )
    return key


@dataclass
class ChatResult:
    content: str
    model: str
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False
    raw_keys: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        """脱敏日志：不含 API key / 完整 prompt。"""
        return {
            "model": self.model,
            "request_id": self.request_id,
            "usage": self.usage,
            "latency_s": round(self.latency_s, 3),
            "cached": self.cached,
            "content_chars": len(self.content),
        }


class OpenAICompatClient:
    """统一 OpenAI 兼容接口；支持超时、有限重试、429、缓存去重。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 90.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        provider: str = "openai_compat",
        min_interval_s: float = 0.3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = f"{provider}:{model}"

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{key}.json"

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = True,
        extra_body: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> ChatResult:
        cache_key = dedupe_key or hashlib.sha256(
            json.dumps(
                {"model": self.model, "messages": messages, "json_mode": json_mode},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cp = self._cache_path(cache_key)
        if cp and cp.exists():
            data = json.loads(cp.read_text(encoding="utf-8"))
            return ChatResult(
                content=data["content"],
                model=data.get("model", self.model),
                request_id=data.get("request_id"),
                usage=data.get("usage", {}),
                latency_s=0.0,
                cached=True,
            )

        self._throttle()

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            retry=retry_if_exception_type(
                (RateLimitError, EmptyContentError, httpx.TransportError)
            ),
        )
        def _once() -> ChatResult:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if extra_body:
                payload.update(extra_body)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            t0 = time.time()
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            self._last_call = time.time()
            if r.status_code == 429:
                raise RateLimitError(f"rate limited: {r.text[:200]}")
            if r.status_code >= 400:
                # 不回显 Authorization；body 截断
                raise APIRequestError(
                    f"API HTTP {r.status_code}",
                    status_code=r.status_code,
                    body=r.text[:500],
                )
            data = r.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # 部分思考模式字段；内容仍以 content 为准
            if not str(content).strip():
                raise EmptyContentError("empty message.content from API")
            usage = data.get("usage") or {}
            usage_norm = {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
            result = ChatResult(
                content=content,
                model=str(data.get("model") or self.model),
                request_id=r.headers.get("x-request-id") or data.get("id"),
                usage=usage_norm,
                latency_s=time.time() - t0,
                cached=False,
                raw_keys=sorted(data.keys()),
            )
            return result

        result = _once()
        if cp and result.content.strip():
            cp.write_text(
                json.dumps(
                    {
                        "content": result.content,
                        "model": result.model,
                        "request_id": result.request_id,
                        "usage": result.usage,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return result

    def generate(self, prompt: str, images: list[Any] | None = None) -> str:
        """TeacherClient / 简单兼容接口。images 忽略（文本-only 验收）。"""
        _ = images
        messages = [
            {
                "role": "system",
                "content": "You are a careful analyst. Reply with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, json_mode=True).content
