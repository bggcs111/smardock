"""检索器（纯代码编排；AI 能力由 Embedder / Reranker 提供）。

职责：把"用户问题"变成"可直接喂给大模型的完整上下文"。

流程：
    1. 向量检索（语义相近，跨语言也有效）
    2. BM25 关键词检索（精确词面，擅长 MIO12 / DS987 这类标识符）
    3. RRF 融合两路候选，取各家所长
    4. 可选重排序精排
    5. 整节展开（small-to-big）：命中块 → 所在章节的完整内容
    6. 收敛到 ``config.MAX_CONTEXTS`` 段

不负责：向量化（Embedder）、重排序（Reranker）、存储（VectorStore / MetadataStore）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import config
from modules.indexing.reranker import RerankError


class Retriever:
    """混合检索编排。"""

    def __init__(self, vector_store, embedder, metadata_store, reranker=None) -> None:
        self.vector = vector_store
        self.embedder = embedder
        self.metadata = metadata_store
        self.reranker = reranker

    # ------------------------------------------------------------------
    def retrieve(
        self,
        kb_names: List[str],
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """返回可直接作为上下文的片段列表。"""
        kbs = [str(kb).strip() for kb in (kb_names or []) if str(kb or "").strip()]
        if not kbs:
            return []
        limit = config.MAX_CONTEXTS if top_k is None else max(1, int(top_k))

        vector_hits = self._vector_search(kbs, question, doc_ids)
        keyword_pairs = self._keyword_search(kbs, question, doc_ids)
        merged = self._fuse(vector_hits, keyword_pairs)
        merged = self._rerank(question, merged)
        return self._expand(merged, limit)

    # ------------------------------------------------------------------
    # 1) 向量检索
    # ------------------------------------------------------------------
    def _vector_search(
        self,
        kbs: List[str],
        question: str,
        doc_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        try:
            query_vector = self.embedder.embed_query(question)
        except Exception:  # noqa: BLE001
            return []
        try:
            return self.vector.search_all(
                kbs,
                query_vector,
                top_k=config.CANDIDATE_VECTOR,
                per_kb_top_k=config.CANDIDATE_VECTOR,
                doc_ids=doc_ids,
            )
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # 2) 关键词检索
    # ------------------------------------------------------------------
    def _keyword_search(
        self,
        kbs: List[str],
        question: str,
        doc_ids: Optional[List[str]],
    ) -> List[Tuple[str, str, float]]:
        if not config.KEYWORD_SEARCH:
            return []
        try:
            return self.metadata.keyword_search(
                question,
                kb_names=kbs,
                doc_ids=doc_ids,
                top_n=config.CANDIDATE_KEYWORD,
            )
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------
    # 3) RRF 融合
    # ------------------------------------------------------------------
    def _fuse(
        self,
        vector_hits: List[Dict[str, Any]],
        keyword_pairs: List[Tuple[str, str, float]],
    ) -> List[Dict[str, Any]]:
        by_chunk: Dict[str, Dict[str, Any]] = {}
        vector_rank: Dict[str, int] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            chunk_id = (hit.get("metadata") or {}).get("chunk_id") or ""
            if not chunk_id:
                continue
            vector_rank[chunk_id] = rank
            by_chunk[chunk_id] = hit

        # 关键词命中里可能有向量没召回的块，需要补取内容
        missing_by_kb: Dict[str, List[str]] = {}
        keyword_rank: Dict[str, int] = {}
        for rank, (chunk_id, kb_name, _score) in enumerate(keyword_pairs, start=1):
            if not chunk_id or chunk_id in keyword_rank:
                continue
            keyword_rank[chunk_id] = rank
            if chunk_id not in by_chunk and kb_name:
                missing_by_kb.setdefault(kb_name, []).append(chunk_id)

        for kb_name, chunk_ids in missing_by_kb.items():
            for item in self.vector.fetch_chunks(kb_name, chunk_ids):
                chunk_id = (item.get("metadata") or {}).get("chunk_id") or ""
                if chunk_id:
                    by_chunk[chunk_id] = item

        scores: Dict[str, float] = {}
        for chunk_id, rank in vector_rank.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)
        for chunk_id, rank in keyword_rank.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)

        merged: List[Dict[str, Any]] = []
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            hit = by_chunk.get(chunk_id)
            if hit is None:
                continue
            item = dict(hit)
            item["fusion_score"] = round(float(score), 6)
            merged.append(item)
        return merged

    # ------------------------------------------------------------------
    # 4) 重排序精排（失败时退回融合顺序）
    # ------------------------------------------------------------------
    def _rerank(self, question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits or self.reranker is None or not self.reranker.is_available():
            return hits

        try:
            ranked = self.reranker.rerank(question, [hit.get("content") or "" for hit in hits])
        except RerankError:
            return hits
        except Exception:  # noqa: BLE001
            return hits

        ordered: List[Dict[str, Any]] = []
        covered: set = set()
        for index, score in ranked:
            if 0 <= index < len(hits):
                item = dict(hits[index])
                item["rerank_score"] = round(float(score), 6)
                ordered.append(item)
                covered.add(index)

        # 未被重排序覆盖的项保持融合顺序跟随其后
        ordered.extend(hit for index, hit in enumerate(hits) if index not in covered)
        return ordered

    # ------------------------------------------------------------------
    # 5) 整节展开（small-to-big）
    # ------------------------------------------------------------------
    def _expand(self, hits: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        if not config.SECTION_EXPAND:
            return hits[:limit]

        expanded: List[Dict[str, Any]] = []
        seen: set = set()

        for hit in hits:
            metadata = dict(hit.get("metadata") or {})
            section_id = str(metadata.get("section_id") or "")
            key = section_id or f"chunk::{metadata.get('chunk_id')}"
            if key in seen:
                continue
            seen.add(key)

            if not section_id:
                expanded.append(hit)
            else:
                kb_name = hit.get("source_kb") or str(metadata.get("kb_name") or "")
                parts = self.vector.fetch_section(
                    kb_name, section_id, limit=config.SECTION_EXPAND_MAX_CHUNKS
                )
                if len(parts) <= 1:
                    expanded.append(hit)
                else:
                    item = dict(hit)
                    item["content"] = "\n".join(part["content"] for part in parts)[
                        : config.MAX_SECTION_CHARS
                    ]
                    item["expanded_chunks"] = len(parts)
                    expanded.append(item)

            if len(expanded) >= limit:
                break

        return expanded[:limit]
