"""向量存储库（ChromaDB）。

职责：存储文本块的向量表示、原文和块级元数据，提供语义相似度检索。
存储内容：向量、文本块、块级元数据（页码、位置、类型、来源文件）

每个知识库对应一个独立 collection，实现知识库之间的物理隔离。
不负责：文档列表管理、脱敏映射、知识库统计。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config


def collection_name(kb_name: str) -> str:
    """ChromaDB 的 collection 名有字符集限制，这里用名称哈希生成稳定安全的名字。"""
    digest = hashlib.sha1((kb_name or "").encode("utf-8")).hexdigest()[:16]
    return f"kb_{digest}"


class VectorStore:
    """ChromaDB 的增删查改封装，支持多知识库隔离。"""

    def __init__(self, persist_dir: Optional[str | Path] = None) -> None:
        import chromadb

        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collections: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # collection 管理
    # ------------------------------------------------------------------
    def get_or_create(self, kb_name: str):
        if kb_name in self._collections:
            return self._collections[kb_name]
        name = collection_name(kb_name)
        try:
            collection = self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # 已存在的 collection 若使用了不同配置，回退为直接获取
            collection = self._client.get_collection(name=name)
        self._collections[kb_name] = collection
        return collection

    def delete_collection(self, kb_name: str) -> None:
        self._collections.pop(kb_name, None)
        try:
            self._client.delete_collection(name=collection_name(kb_name))
        except Exception:
            pass

    def count(self, kb_name: str) -> int:
        try:
            return int(self.get_or_create(kb_name).count())
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_documents(
        self,
        kb_name: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        if not ids:
            return
        collection = self.get_or_create(kb_name)
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=[self._clean_metadata(m) for m in metadatas],
            embeddings=embeddings,
        )

    def delete_by_doc(self, kb_name: str, doc_id: str) -> None:
        try:
            self.get_or_create(kb_name).delete(where={"doc_id": doc_id})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(
        self,
        kb_name: str,
        query_embedding: List[float],
        top_k: int = 5,
        doc_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """检索单个知识库；``doc_ids`` 非空时限定为这些文档（用于单篇文档问答）。"""
        try:
            collection = self.get_or_create(kb_name)
            if collection.count() == 0:
                return []
            where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []
        return self._unpack(result, kb_name)

    def search_all(
        self,
        kb_names: Iterable[str],
        query_embedding: List[float],
        top_k: int = 8,
        per_kb_top_k: Optional[int] = None,
        doc_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        per_kb = per_kb_top_k or top_k
        merged: List[Dict[str, Any]] = []
        for kb_name in kb_names:
            merged.extend(self.search(kb_name, query_embedding, top_k=per_kb, doc_ids=doc_ids))
        merged.sort(key=lambda item: item["similarity"], reverse=True)
        return merged[: max(1, int(top_k))]

    def fetch_chunks(self, kb_name: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """按 id 批量取回块（混合检索里补全关键词命中的内容用）。"""
        if not chunk_ids:
            return []
        try:
            collection = self.get_or_create(kb_name)
            got = collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
        except Exception:
            return []

        ids = got.get("ids") or []
        documents = got.get("documents") or []
        metadatas = got.get("metadatas") or []

        items: List[Dict[str, Any]] = []
        for index, content in enumerate(documents):
            metadata = dict(metadatas[index]) if index < len(metadatas) else {}
            metadata.setdefault("chunk_id", ids[index] if index < len(ids) else "")
            items.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "similarity": 0.0,
                    "source_kb": kb_name,
                }
            )
        return items

    # ------------------------------------------------------------------
    # 章节取回（small-to-big 检索：命中一个块 -> 取回整节内容）
    # ------------------------------------------------------------------
    def fetch_section(
        self,
        kb_name: str,
        section_id: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """按 ``section_id`` 取回同一章节的全部块，按文档顺序返回。"""
        if not section_id:
            return []
        try:
            collection = self.get_or_create(kb_name)
            got = collection.get(
                where={"section_id": section_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        ids = got.get("ids") or []
        documents = got.get("documents") or []
        metadatas = got.get("metadatas") or []

        items: List[Dict[str, Any]] = []
        for index, doc in enumerate(documents):
            metadata = dict(metadatas[index]) if index < len(metadatas) else {}
            items.append(
                {
                    "chunk_id": ids[index] if index < len(ids) else "",
                    "content": doc,
                    "metadata": metadata,
                }
            )
        # chunk_id 形如 "{doc_id}-{序号}"，按序号升序即为文档顺序
        items.sort(key=lambda item: str(item["chunk_id"]))
        return items[: max(1, int(limit))]

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB 只接受 str/int/float/bool 类型的元数据值。"""
        cleaned: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (dict, list, tuple)):
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            else:
                cleaned[key] = str(value)
        return cleaned

    @staticmethod
    def _unpack(result: Dict[str, Any], kb_name: str) -> List[Dict[str, Any]]:
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        items: List[Dict[str, Any]] = []
        for idx, content in enumerate(documents):
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            metadata = dict(metadatas[idx]) if idx < len(metadatas) else {}
            metadata.setdefault("chunk_id", ids[idx] if idx < len(ids) else "")
            items.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "similarity": round(max(0.0, 1.0 - distance), 4),
                    "source_kb": kb_name,
                }
            )
        return items
