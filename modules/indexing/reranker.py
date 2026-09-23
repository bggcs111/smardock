"""重排序器（AI 能力 · 云端 Rerank API）。

职责：对检索融合后的候选片段做精排，把最相关的排到前面。
实现：DashScope 的 gte-rerank 系列（OpenAI 兼容协议之外的原生接口）。

说明：重排序是**可选增强**。不可用时（未配置 Key、模型未开通、网络异常）
会抛出 :class:`RerankError`，调用方应当退回到融合排序结果而不是中断问答。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import config
from modules.logutil import get_logger, next_seq


class RerankError(RuntimeError):
    """重排序失败。"""


class Reranker:
    """DashScope 重排序封装。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.DASHSCOPE_API_KEY
        self.model = model or config.RERANK_MODEL
        self.top_n = config.RERANK_TOP_N if top_n is None else int(top_n)
        self.enabled = config.RERANK_ENABLED if enabled is None else bool(enabled)

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self.enabled and self.api_key)

    def describe(self) -> str:
        return f"{self.model}（{'启用' if self.is_available() else '未启用'}）"

    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """返回 ``[(原文下标, 相关性得分), ...]``，按相关性从高到低。"""
        if not self.is_available():
            raise RerankError("重排序未启用或未配置 API Key")
        query = (query or "").strip()
        documents = [doc or "" for doc in (documents or [])]
        if not query or not documents:
            return [(index, 0.0) for index in range(len(documents))]

        import dashscope

        seq = next_seq("Rerank")
        started = time.perf_counter()
        try:
            response = dashscope.TextReRank.call(
                model=self.model,
                api_key=self.api_key,
                query=query[:2000],
                # 接口要求非空字符串，这里用占位符兜底
                documents=[doc[:2000] if doc.strip() else "-" for doc in documents],
                top_n=min(self.top_n, len(documents)),
                return_documents=False,
            )

            status = getattr(response, "status_code", None)
            if status != 200:
                raise RerankError(
                    f"重排序返回错误 {status}: {getattr(response, 'message', '')}".strip()
                )

            output = getattr(response, "output", None) or {}
            results: List[Dict[str, Any]] = output.get("results") or []
            if not results:
                raise RerankError("重排序未返回结果")

            ranked: List[Tuple[int, float]] = []
            for item in results:
                try:
                    ranked.append((int(item["index"]), float(item.get("relevance_score") or 0.0)))
                except (KeyError, TypeError, ValueError):
                    continue

            if not ranked:
                raise RerankError("重排序结果无法解析")
        except Exception as exc:  # noqa: BLE001
            get_logger().error(
                "Rerank调用 #%d 失败 文档=%d段 耗时=%.1fs：%s",
                seq, len(documents), time.perf_counter() - started, exc,
            )
            raise
        get_logger().info(
            "Rerank调用 #%d %s 文档=%d段 耗时=%.1fs",
            seq, self.model, len(documents), time.perf_counter() - started,
        )
        return ranked
