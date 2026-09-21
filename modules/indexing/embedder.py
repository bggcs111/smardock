"""向量化器（AI 能力 · 云端 Embedding API）。

职责：把文本字符串转成向量。
输入：文本字符串
输出：与 config.EMBEDDING_DIMENSION 一致的向量（默认 1024 维）

可替换性：通过 EMBEDDING_PROVIDER 可在 DashScope 与任意 OpenAI 兼容端点
（OpenAI / SiliconFlow / 本地 vLLM / Ollama）之间切换，无需改动上层代码。
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import config


class EmbeddingError(RuntimeError):
    """向量化失败。"""


class Embedder:
    """云端 Embedding API 封装（带重试与批内缓存）。"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retry: Optional[int] = None,
    ) -> None:
        self.provider = (provider or config.EMBEDDING_PROVIDER).lower()
        self.model = model or config.EMBEDDING_MODEL
        self.dimension = int(dimension or config.EMBEDDING_DIMENSION)
        self.api_key = api_key if api_key is not None else config.EMBEDDING_API_KEY
        self.base_url = base_url if base_url is not None else config.EMBEDDING_BASE_URL
        self.batch_size = max(1, int(batch_size or config.EMBEDDING_BATCH_SIZE))
        self.max_retry = max(1, int(max_retry or config.EMBEDDING_MAX_RETRY))
        self._cache: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        if not self.api_key:
            return False
        if self.provider == "openai" and not self.base_url:
            return False
        return True

    def describe(self) -> str:
        return f"{self.provider}:{self.model}({self.dimension}d)"

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，返回与输入等长的向量列表。"""
        if not texts:
            return []
        if not self.is_available():
            raise EmbeddingError(
                "向量化器未配置：请设置 DASHSCOPE_API_KEY（或 EMBEDDING_API_KEY / EMBEDDING_BASE_URL）"
            )

        results: List[Optional[List[float]]] = [None] * len(texts)
        pending_indexes: List[int] = []
        pending_texts: List[str] = []

        for index, raw in enumerate(texts):
            text = self._prepare(raw)
            key = self._cache_key(text)
            if key in self._cache:
                results[index] = self._cache[key]
                continue
            pending_indexes.append(index)
            pending_texts.append(text)

        for start in range(0, len(pending_texts), self.batch_size):
            batch_texts = pending_texts[start:start + self.batch_size]
            batch_indexes = pending_indexes[start:start + self.batch_size]
            vectors = self._embed_batch(batch_texts)
            for offset, vector in enumerate(vectors):
                index = batch_indexes[offset]
                results[index] = vector
                self._cache[self._cache_key(batch_texts[offset])] = vector

        if any(vector is None for vector in results):
            raise EmbeddingError("向量化结果不完整，请重试")
        return [vector for vector in results if vector is not None]

    def embed_query(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise EmbeddingError("查询向量化失败")
        return vectors[0]

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retry + 1):
            try:
                if self.provider == "openai":
                    return self._embed_openai(texts)
                return self._embed_dashscope(texts)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retry:
                    time.sleep(min(2 ** attempt, 8))
        raise EmbeddingError(f"向量化请求失败（已重试 {self.max_retry} 次）：{last_error}")

    def _embed_dashscope(self, texts: List[str]) -> List[List[float]]:
        import dashscope

        response = dashscope.TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key,
            dimension=self.dimension,
            output_type="dense",
        )
        status = getattr(response, "status_code", 200)
        if status != 200:
            raise EmbeddingError(
                f"DashScope 返回错误 {status}: {getattr(response, 'code', '')} "
                f"{getattr(response, 'message', '')}".strip()
            )
        output = getattr(response, "output", None) or {}
        embeddings = output.get("embeddings") if isinstance(output, dict) else None
        if not embeddings:
            raise EmbeddingError("DashScope 未返回 embeddings 字段")

        ordered = sorted(embeddings, key=lambda item: item.get("text_index", 0))
        return [list(item.get("embedding") or []) for item in ordered]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        kwargs: Dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimension:
            try:
                response = client.embeddings.create(dimensions=self.dimension, **kwargs)
            except Exception:
                response = client.embeddings.create(**kwargs)
        else:
            response = client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in response.data]

    @staticmethod
    def _prepare(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = "（空内容）"
        if len(cleaned) > config.EMBEDDING_MAX_CHARS:
            cleaned = cleaned[: config.EMBEDDING_MAX_CHARS]
        return cleaned

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        self._cache.clear()
