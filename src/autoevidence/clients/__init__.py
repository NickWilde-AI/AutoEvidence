"""外部 API 客户端。"""

from autoevidence.clients.openai_compat import OpenAICompatClient, require_api_key

__all__ = ["OpenAICompatClient", "require_api_key"]
