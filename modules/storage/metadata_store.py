"""元数据存储库（SQLite）。

职责：管理系统运行所需的结构化信息。
核心表：
    knowledge_bases  知识库列表
    documents        文档列表
    chunks_index     块级索引
    redaction_map    脱敏映射（占位符 <-> 原文）
    query_log        问答历史

不负责：向量检索、文件读写、文本处理。
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config

_PLACEHOLDER_RE = re.compile(r"^(?P<label>.+?)(?P<index>\d+)$")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    name        TEXT PRIMARY KEY,
    -- 遗留字段：界面上的「领域」已移除，代码不再读取；保留列只为不破坏已有库结构
    type        TEXT NOT NULL DEFAULT 'general',
    collection  TEXT NOT NULL DEFAULT '',
    doc_count   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    -- 处理模式与知识库绑定：'privacy' 脱敏 / 'normal' 不脱敏，建库时确定
    mode        TEXT NOT NULL DEFAULT 'normal'
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    kb_name         TEXT NOT NULL,
    upload_time     TEXT NOT NULL,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    redaction_count INTEGER NOT NULL DEFAULT 0,
    file_path       TEXT NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'normal',
    FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(kb_name);

CREATE TABLE IF NOT EXISTS chunks_index (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    kb_name     TEXT NOT NULL,
    page        INTEGER,
    position    TEXT,
    type        TEXT NOT NULL DEFAULT 'text',
    char_count  INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks_index(doc_id);

CREATE TABLE IF NOT EXISTS redaction_map (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_name     TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    placeholder TEXT NOT NULL UNIQUE,
    original    TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_redaction_kb ON redaction_map(kb_name);
CREATE INDEX IF NOT EXISTS idx_redaction_entity ON redaction_map(entity_type, original);

CREATE TABLE IF NOT EXISTS query_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_names    TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'normal',
    created_at  TEXT NOT NULL
);

-- 问答缓存：问题 + 范围 + 模式完全一致时直接复用，避免重复调用模型
CREATE TABLE IF NOT EXISTS qa_cache (
    fingerprint TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    kb_names    TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'normal',
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- 会话：一次「新建对话」对应一条会话记录
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    kb_names    TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);

-- BM25 关键词索引：tokens 列存放分词后的词元串，其余列仅作过滤用
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    kb_name UNINDEXED,
    doc_id UNINDEXED,
    tokens,
    tokenize='unicode61'
);
"""


class MetadataStore:
    """SQLite 元数据库封装（线程安全，供 Gradio 多线程调用）。"""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = Path(db_path or config.METADATA_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._migrate()

    # ------------------------------------------------------------------
    # 结构迁移：给已存在的库补列，并把老数据归纳好
    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        # 知识库模式：老库按已有文档推断——含隐私文档的库视为隐私库
        kb_columns = {row["name"] for row in self._query("PRAGMA table_info(knowledge_bases)")}
        if "mode" not in kb_columns:
            self._execute(
                "ALTER TABLE knowledge_bases ADD COLUMN mode TEXT NOT NULL DEFAULT 'normal'"
            )
            self._execute(
                "UPDATE knowledge_bases SET mode = 'privacy' WHERE name IN"
                " (SELECT DISTINCT kb_name FROM documents WHERE mode = 'privacy')"
            )

        columns = {row["name"] for row in self._query("PRAGMA table_info(query_log)")}
        added_session = False
        if "session_id" not in columns:
            self._execute("ALTER TABLE query_log ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
            added_session = True
        if "refs_json" not in columns:
            self._execute("ALTER TABLE query_log ADD COLUMN refs_json TEXT NOT NULL DEFAULT ''")

        # 老库的问答记录没有会话归属，统一归入一个「历史对话」，
        # 否则它们会因为找不到会话而从列表里消失。
        if added_session:
            rows = self._query("SELECT COUNT(*) AS c FROM query_log WHERE session_id = ''")
            if rows and int(rows[0]["c"]) > 0:
                session_id = self.create_session(
                    title="历史对话", session_id="history-default"
                )
                self._execute(
                    "UPDATE query_log SET session_id = ? WHERE session_id = ''", (session_id,)
                )

    # ------------------------------------------------------------------
    # 基础执行封装
    # ------------------------------------------------------------------
    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cursor

    def _query(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # 知识库
    # ------------------------------------------------------------------
    def ensure_kb(
        self,
        name: str,
        collection: str = "",
        mode: str = "normal",
    ) -> Dict[str, Any]:
        """建库（已存在则原样返回）。

        说明：``type`` 列是早期遗留字段，界面上已移除、代码里不再读取；
        保留列本身只为不破坏已有库的结构，新库统一写入 'general'。
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("知识库名称不能为空")
        rows = self._query("SELECT name FROM knowledge_bases WHERE name = ?", (name,))
        if not rows:
            self._execute(
                "INSERT INTO knowledge_bases(name, type, collection, doc_count, created_at, mode)"
                " VALUES(?,?,?,0,?,?)",
                (name, "general", collection, _now(), mode or "normal"),
            )
        return self.get_kb(name) or {}

    def kb_mode(self, name: str) -> str:
        """知识库的处理模式（privacy / normal）；不存在时按普通模式处理。"""
        kb = self.get_kb(name)
        return (kb or {}).get("mode") or "normal"

    def get_kb(self, name: str) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM knowledge_bases WHERE name = ?", (name,))
        return dict(rows[0]) if rows else None

    def list_kbs(self) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM knowledge_bases ORDER BY created_at ASC, name ASC"
        )
        return [dict(r) for r in rows]

    def kb_names(self) -> List[str]:
        return [r["name"] for r in self._query("SELECT name FROM knowledge_bases ORDER BY created_at ASC")]

    def delete_kb(self, name: str) -> None:
        self._execute("DELETE FROM knowledge_bases WHERE name = ?", (name,))

    def _refresh_doc_count(self, kb_name: str) -> None:
        self._execute(
            "UPDATE knowledge_bases SET doc_count = (SELECT COUNT(*) FROM documents WHERE kb_name = ?) WHERE name = ?",
            (kb_name, kb_name),
        )

    # ------------------------------------------------------------------
    # 文档
    # ------------------------------------------------------------------
    def add_document(
        self,
        doc_id: str,
        filename: str,
        kb_name: str,
        file_path: str,
        mode: str = "normal",
        chunk_count: int = 0,
        redaction_count: int = 0,
    ) -> None:
        self._execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, filename, kb_name, upload_time, chunk_count, redaction_count, file_path, mode)
               VALUES(?,?,?,?,?,?,?,?)""",
            (doc_id, filename, kb_name, _now(), chunk_count, redaction_count, file_path, mode),
        )
        self._refresh_doc_count(kb_name)

    def update_document_stats(self, doc_id: str, chunk_count: int, redaction_count: int) -> None:
        self._execute(
            "UPDATE documents SET chunk_count = ?, redaction_count = ? WHERE doc_id = ?",
            (chunk_count, redaction_count, doc_id),
        )

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return dict(rows[0]) if rows else None

    def list_documents(self, kb_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if kb_name:
            rows = self._query(
                "SELECT * FROM documents WHERE kb_name = ? ORDER BY upload_time DESC", (kb_name,)
            )
        else:
            rows = self._query("SELECT * FROM documents ORDER BY upload_time DESC")
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        doc = self.get_document(doc_id)
        if not doc:
            return None
        # chunks_index 通过外键级联删除；redaction_map 为全局共享映射，故意保留，
        # 避免同一敏感实体被其它文档复用时映射丢失。
        self._execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._refresh_doc_count(doc["kb_name"])
        return doc

    # ------------------------------------------------------------------
    # 块级索引
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        rows = [
            (
                c["chunk_id"],
                c["doc_id"],
                c["kb_name"],
                c.get("page"),
                json.dumps(c.get("position") or {}, ensure_ascii=False),
                c.get("type", "text"),
                len(c.get("content") or ""),
            )
            for c in chunks
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO chunks_index
                   (chunk_id, doc_id, kb_name, page, position, type, char_count)
                   VALUES(?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()

    def delete_chunks_by_doc(self, doc_id: str) -> None:
        """删除某文档的全部块索引（重建索引时使用）。"""
        self._execute("DELETE FROM chunks_index WHERE doc_id = ?", (doc_id,))

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM chunks_index WHERE chunk_id = ?", (chunk_id,))
        return dict(rows[0]) if rows else None

    def count_chunks(self, kb_name: Optional[str] = None) -> int:
        if kb_name:
            rows = self._query("SELECT COUNT(*) AS c FROM chunks_index WHERE kb_name = ?", (kb_name,))
        else:
            rows = self._query("SELECT COUNT(*) AS c FROM chunks_index")
        return int(rows[0]["c"]) if rows else 0

    # ------------------------------------------------------------------
    # 关键词索引（BM25，与向量检索互补）
    # ------------------------------------------------------------------
    def index_chunk_texts(self, chunks: List[Dict[str, Any]]) -> None:
        """把 chunk 正文分词后写入 FTS 索引（同一文档会先清后写）。"""
        if not chunks:
            return
        from modules.indexing.tokenizer import to_index_text

        rows = [
            (
                chunk["chunk_id"],
                chunk.get("kb_name", ""),
                chunk.get("doc_id", ""),
                to_index_text(chunk.get("content") or ""),
            )
            for chunk in chunks
        ]
        doc_ids = {row[2] for row in rows if row[2]}
        with self._lock:
            for doc_id in doc_ids:
                self._conn.execute("DELETE FROM chunk_fts WHERE doc_id = ?", (doc_id,))
            self._conn.executemany(
                "INSERT INTO chunk_fts(chunk_id, kb_name, doc_id, tokens) VALUES(?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def delete_chunk_texts(self, doc_id: str) -> None:
        self._execute("DELETE FROM chunk_fts WHERE doc_id = ?", (doc_id,))

    def keyword_search(
        self,
        query: str,
        kb_names: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> List[Tuple[str, str, float]]:
        """BM25 关键词检索，返回 ``[(chunk_id, kb_name, score), ...]``，分数越小越相关。"""
        from modules.indexing.tokenizer import tokenize

        tokens = tokenize(query)
        if not tokens:
            return []

        # 只取前若干个词元，避免超长查询拖慢 MATCH
        match = " OR ".join(f'"{token}"' for token in tokens[:32])
        sql = (
            "SELECT chunk_id, kb_name, bm25(chunk_fts) AS score "
            "FROM chunk_fts WHERE chunk_fts MATCH ?"
        )
        params: List[Any] = [match]

        if kb_names:
            placeholders = ",".join("?" for _ in kb_names)
            sql += f" AND kb_name IN ({placeholders})"
            params.extend(kb_names)
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            sql += f" AND doc_id IN ({placeholders})"
            params.extend(doc_ids)

        # bm25() 越小越相关
        sql += " ORDER BY score LIMIT ?"
        params.append(max(1, int(top_n)))

        try:
            rows = self._query(sql, params)
        except Exception:
            return []
        return [(row["chunk_id"], row["kb_name"], float(row["score"])) for row in rows]

    # ------------------------------------------------------------------
    # 脱敏映射（占位符在全库范围内全局唯一，避免跨知识库/跨文档还原歧义）
    # ------------------------------------------------------------------
    def find_placeholder(self, entity_type: str, original: str) -> Optional[str]:
        rows = self._query(
            "SELECT placeholder FROM redaction_map WHERE entity_type = ? AND original = ? LIMIT 1",
            (entity_type, original),
        )
        return rows[0]["placeholder"] if rows else None

    def next_placeholder_index(self, entity_type: str) -> int:
        """返回该实体类型下一个可用的全局编号（占位符全局唯一）。"""
        rows = self._query(
            "SELECT placeholder FROM redaction_map WHERE entity_type = ?", (entity_type,)
        )
        used: set[int] = set()
        for row in rows:
            match = _PLACEHOLDER_RE.match(str(row["placeholder"]).strip("[]"))
            if match:
                try:
                    used.add(int(match.group("index")))
                except ValueError:
                    continue
        index = 1
        while index in used:
            index += 1
        return index

    def add_redactions(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        rows = [
            (
                r["kb_name"],
                r.get("doc_id", ""),
                r["placeholder"],
                r["original"],
                r["entity_type"],
                _now(),
            )
            for r in records
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT OR IGNORE INTO redaction_map
                   (kb_name, doc_id, placeholder, original, entity_type, created_at)
                   VALUES(?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()

    def get_redaction_map(
        self, kb_name: Optional[str] = None, doc_ids: Optional[List[str]] = None
    ) -> Dict[str, str]:
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            rows = self._query(
                f"SELECT placeholder, original FROM redaction_map WHERE doc_id IN ({placeholders})",
                doc_ids,
            )
            if rows:
                return {r["placeholder"]: r["original"] for r in rows}
        if kb_name:
            rows = self._query(
                "SELECT placeholder, original FROM redaction_map WHERE kb_name = ?", (kb_name,)
            )
            if rows:
                return {r["placeholder"]: r["original"] for r in rows}
        rows = self._query("SELECT placeholder, original FROM redaction_map")
        return {r["placeholder"]: r["original"] for r in rows}

    def count_redactions(self, kb_name: Optional[str] = None) -> int:
        if kb_name:
            rows = self._query("SELECT COUNT(*) AS c FROM redaction_map WHERE kb_name = ?", (kb_name,))
        else:
            rows = self._query("SELECT COUNT(*) AS c FROM redaction_map")
        return int(rows[0]["c"]) if rows else 0

    # ------------------------------------------------------------------
    # 会话（新建对话 / 历史定位）
    # ------------------------------------------------------------------
    def create_session(
        self,
        title: str = "",
        kb_names: Optional[List[str]] = None,
        session_id: str = "",
    ) -> str:
        session_id = session_id or uuid.uuid4().hex[:12]
        now = _now()
        self._execute(
            "INSERT OR IGNORE INTO sessions(session_id, title, kb_names, created_at, updated_at)"
            " VALUES(?,?,?,?,?)",
            (session_id, (title or "").strip(), "、".join(kb_names or []), now, now),
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        rows = self._query("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return dict(rows[0]) if rows else None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """会话列表（最新在前），附带轮数。"""
        rows = self._query(
            "SELECT s.*, (SELECT COUNT(*) FROM query_log q WHERE q.session_id = s.session_id)"
            " AS turn_count FROM sessions s ORDER BY s.updated_at DESC, s.rowid DESC LIMIT ?",
            (int(limit),),
        )
        sessions = []
        for row in rows:
            item = dict(row)
            item["turn_count"] = int(item.get("turn_count") or 0)
            sessions.append(item)
        return sessions

    def latest_session_id(self) -> str:
        sessions = self.list_sessions(limit=1)
        return sessions[0]["session_id"] if sessions else ""

    def touch_session(
        self,
        session_id: str,
        kb_names: Optional[List[str]] = None,
        title: str = "",
    ) -> None:
        if not session_id:
            return
        if title:
            self._execute(
                "UPDATE sessions SET updated_at = ?, title = ?, kb_names = ? WHERE session_id = ?",
                (_now(), title, "、".join(kb_names or []), session_id),
            )
        else:
            self._execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?", (_now(), session_id)
            )

    def delete_session(self, session_id: str) -> int:
        """删除会话及其全部问答记录，返回删除的记录条数。"""
        rows = self._query(
            "SELECT COUNT(*) AS c FROM query_log WHERE session_id = ?", (session_id,)
        )
        count = int(rows[0]["c"]) if rows else 0
        self._execute("DELETE FROM query_log WHERE session_id = ?", (session_id,))
        self._execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return count

    def session_records(self, session_id: str) -> List[Dict[str, Any]]:
        """某个会话的全部问答记录（按发生顺序升序）。"""
        if not session_id:
            return []
        rows = self._query(
            "SELECT * FROM query_log WHERE session_id = ? ORDER BY id ASC", (session_id,)
        )
        return [dict(r) for r in rows]

    def session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """把会话记录展开成聊天区的消息列表（一问一答为一轮）。"""
        messages: List[Dict[str, str]] = []
        for record in self.session_records(session_id):
            question = (record.get("question") or "").strip()
            answer = (record.get("answer") or "").strip()
            if question:
                messages.append({"role": "user", "content": question})
            if answer:
                messages.append({"role": "assistant", "content": answer})
        return messages

    # ------------------------------------------------------------------
    # 问答历史
    # ------------------------------------------------------------------
    def log_query(
        self,
        kb_names: List[str],
        question: str,
        answer: str,
        mode: str = "normal",
        session_id: str = "",
        references: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._execute(
            "INSERT INTO query_log(kb_names, question, answer, mode, session_id, refs_json, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (
                "、".join(kb_names),
                question,
                answer,
                mode,
                session_id or "",
                json.dumps(references or [], ensure_ascii=False),
                _now(),
            ),
        )

    def list_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (int(limit),)
        )
        return [dict(r) for r in rows]

    def recent_records(self, limit: int = 30) -> List[Dict[str, Any]]:
        """最近的问答记录（最新在前），附带所属会话标题，供历史列表展示与点击定位。"""
        rows = self._query(
            "SELECT q.*, COALESCE(s.title, '') AS session_title FROM query_log q"
            " LEFT JOIN sessions s ON s.session_id = q.session_id"
            " ORDER BY q.id DESC LIMIT ?",
            (int(limit),),
        )
        return [dict(r) for r in rows]

    def get_query(self, record_id: int) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM query_log WHERE id = ?", (int(record_id),))
        return dict(rows[0]) if rows else None

    def turn_index(self, record_id: int) -> int:
        """该记录在其所属会话内是第几轮（从 1 开始；找不到则返回 1）。"""
        record = self.get_query(record_id)
        if not record:
            return 1
        rows = self._query(
            "SELECT COUNT(*) AS c FROM query_log WHERE session_id = ? AND id <= ?",
            (record.get("session_id") or "", int(record_id)),
        )
        return int(rows[0]["c"]) if rows else 1

    def clear_queries(self) -> int:
        """清空全部问答历史、会话与缓存，返回清掉的历史条数。"""
        rows = self._query("SELECT COUNT(*) AS c FROM query_log")
        count = int(rows[0]["c"]) if rows else 0
        self._execute("DELETE FROM query_log")
        self._execute("DELETE FROM sessions")
        self.clear_cached_answers()
        return count

    # ------------------------------------------------------------------
    # 问答缓存：相同问题 + 相同范围 + 相同模式时直接复用
    # ------------------------------------------------------------------
    def save_cached_answer(
        self,
        fingerprint: str,
        question: str,
        kb_names: List[str],
        mode: str,
        payload: Dict[str, Any],
    ) -> None:
        if not fingerprint:
            return
        try:
            self._execute(
                "INSERT OR REPLACE INTO qa_cache"
                "(fingerprint, question, kb_names, mode, payload, created_at) VALUES(?,?,?,?,?,?)",
                (
                    fingerprint,
                    question,
                    "、".join(kb_names or []),
                    mode,
                    json.dumps(payload, ensure_ascii=False),
                    _now(),
                ),
            )
        except Exception:
            pass

    def find_cached_answer(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        if not fingerprint:
            return None
        try:
            rows = self._query("SELECT * FROM qa_cache WHERE fingerprint = ?", (fingerprint,))
        except Exception:
            return None
        if not rows:
            return None

        record = dict(rows[0])
        try:
            payload = json.loads(record.get("payload") or "{}")
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return {
            "question": record.get("question", ""),
            "kb_names": record.get("kb_names", ""),
            "mode": record.get("mode", ""),
            "created_at": record.get("created_at", ""),
            **payload,
        }

    def clear_cached_answers(self) -> None:
        """清空问答缓存。索引发生变更后必须调用，避免复用过期答案。"""
        try:
            self._execute("DELETE FROM qa_cache")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "kb_count": len(self.kb_names()),
            "doc_count": len(self.list_documents()),
            "chunk_count": self.count_chunks(),
            "redaction_count": self.count_redactions(),
            "query_count": len(self.list_queries(limit=10**9)),
        }
