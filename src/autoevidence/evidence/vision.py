"""SigLIP2 视觉编码器适配 + 向量缓存 + 哈希回退。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# 官方核实 (2026-10-19): google/siglip2-base-patch16-224 存在于 Hugging Face
DEFAULT_SIGLIP2_ID = "google/siglip2-base-patch16-224"


class VisionEncoder:
    """统一接口：encode_images(paths_or_arrays) -> (N, D) L2 归一化向量。"""

    name: str = "base"
    dim: int = 64

    def encode_images(self, images: list[Any]) -> np.ndarray:
        raise NotImplementedError


class HashVisionEncoder(VisionEncoder):
    """无权重时的确定性回退；明确标注非 SigLIP2。"""

    name = "hash_fallback"
    dim = 64

    def encode_images(self, images: list[Any]) -> np.ndarray:
        vecs = []
        for im in images:
            key = str(im)
            h = hashlib.sha256(key.encode()).digest()
            raw = np.frombuffer(h * 2, dtype=np.uint8)[: self.dim].astype(np.float32)
            raw = raw - raw.mean()
            n = np.linalg.norm(raw) + 1e-8
            vecs.append(raw / n)
        if not vecs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(vecs, axis=0)


class SigLIP2Encoder(VisionEncoder):
    """真实 SigLIP2 适配；权重缺失时自动回退并记录。"""

    name = "siglip2"
    dim = 768

    def __init__(
        self,
        model_id: str = DEFAULT_SIGLIP2_ID,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        allow_fallback: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_fallback = allow_fallback
        self._model = None
        self._processor = None
        self.backend = "unloaded"
        self._fallback = HashVisionEncoder()
        self._try_load()

    def _try_load(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
            import torch

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.eval()
            self._model.to(self.device)
            self.backend = "transformers"
            self.dim = int(getattr(self._model.config, "projection_dim", 768))
        except Exception as e:
            self.backend = f"fallback_hash:{type(e).__name__}"
            if not self.allow_fallback:
                raise
            self._model = None

    def encode_images(self, images: list[Any]) -> np.ndarray:
        if self._model is None:
            return self._fallback.encode_images(images)
        from PIL import Image
        import torch

        pil_images = []
        for im in images:
            if isinstance(im, (str, Path)):
                try:
                    pil_images.append(Image.open(im).convert("RGB"))
                except Exception:
                    # 缺失文件：用纯色占位，避免虚构本车证据语义
                    pil_images.append(Image.new("RGB", (224, 224), color=(128, 128, 128)))
            else:
                pil_images.append(im)
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


class VectorCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, list[float]] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> np.ndarray | None:
        if key in self._data:
            return np.asarray(self._data[key], dtype=np.float32)
        return None

    def set(self, key: str, vec: np.ndarray) -> None:
        self._data[key] = vec.astype(float).tolist()

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data), encoding="utf-8")


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T


def build_encoder(prefer_siglip: bool = False, **kwargs) -> VisionEncoder:
    if prefer_siglip:
        enc = SigLIP2Encoder(**kwargs)
        return enc
    return HashVisionEncoder()
