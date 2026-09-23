"""事件处理器（纯代码）。

职责：协调各单元调用，处理用户交互事件（建库、上传索引、提问、删库删文档）。
UI 层只负责收集输入与展示输出，所有业务编排集中在本模块。

实现说明（含对原方案的偏离）：
    1. 上传使用「单选目标知识库」，避免一次上传被复制到多个知识库产生重复索引；
       问答侧仍按方案要求使用多选知识库。
    2. 隐私模式的还原不再依赖问答时的手动模式开关：只要回答或引用中出现
       已登记的占位符，就自动在本地还原，避免用户忘记切换模式。
    3. 问答范围支持「全部知识库」与「单篇文档」两种；选定单篇文档时忽略知识库
       多选框，只在该文档内检索（文档本身已隐含其归属知识库）。
    4. 删除属于不可恢复操作：入口放在折叠的危险操作区，且必须勾选确认框才会执行。
    5. 所有会改动列表型组件的事件统一使用 :data:`CANON_KEYS` 定义的输出契约，
       新增组件时只需改一处，不会出现输出顺序错位。
    6. 回答走流式输出：每收到一段增量就重算整段展示文本（转义 → 占位符还原 →
       引用编号转上标），内容单调增长，不会出现光标乱跳。
    7. 同一时刻只处理一轮问答：回答期间输入区被锁定、只留「停止」按钮，
       停止是纯前端取消（不进队列），正常结束或被停止后由 ``finish_query``
       这个 ``then`` 收尾事件统一解锁输入区。
    8. 索引类任务（上传建库 / 重建索引）同样可手动停止，且同一时刻只允许一项：
       取消信号由 :meth:`KnowledgeQAApp.stop_indexing` 置位，向量化每批检查一次；
       已完成文档保留索引，未完成的那篇不落库（归档原件一并回滚）。
    9. 问答的「停止」不只靠前端取消：前端取消要等 Gradio 把生成器关掉才算数
       （生成器只能停在 yield 上，检索阶段中间没有 yield，界面会锁着好几秒）。
       因此停止按钮另挂一个后端处理器 :meth:`KnowledgeQAApp.stop_query`，
       先解冻界面，再靠 :attr:`_query_cancel` 让检索与生成在下一个检查点收手。
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr

import config
from modules.indexing.embedder import Embedder, EmbeddingCancelled, EmbeddingError
from modules.indexing.reference_locator import ReferenceLocator
from modules.indexing.reranker import Reranker
from modules.indexing.retriever import Retriever
from modules.intelligence.placeholder_restorer import PlaceholderRestorer
from modules.intelligence.qa_agent import LLMError, QAAgent
from modules.intelligence.router import Router
from modules.logutil import get_logger
from modules.parsing.document_parser import DocumentParser, UnsupportedDocumentError
from modules.parsing.figure_detector import FigureDetector
from modules.parsing.pii_redactor import PIIRedactor
from modules.storage.document_store import DocumentStore
from modules.storage.metadata_store import MetadataStore
from modules.storage.vector_store import VectorStore, collection_name

DOC_TABLE_HEADERS = ["文件名", "知识库", "模式", "块数", "脱敏"]
MODE_NORMAL = "普通模式"
MODE_PRIVACY = "隐私模式"
#: 落库用的模式值（知识库与文档记录统一使用）
MODE_NORMAL_VALUE = "normal"
MODE_PRIVACY_VALUE = "privacy"
#: 建库时的模式选项：模式与知识库绑定，库内文档一律按此处理
KB_MODE_CHOICES = [
    ("普通模式 · 原文入库，不做脱敏", MODE_NORMAL_VALUE),
    ("隐私模式 · 入库前脱敏，云端只见占位符", MODE_PRIVACY_VALUE),
]

#: 问答范围 = 全部知识库（按左侧勾选）
SCOPE_ALL = "__ALL__"

#: 回答里的引用编号，形如 [1]、[2]
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

#: 列表型组件的统一输出契约，顺序必须与 app.py 中 outputs 的构造顺序一致
CANON_KEYS: List[str] = [
    "kb_checkbox",          # 问答使用的知识库（多选）
    "upload_kb",            # 上传目标知识库
    "scope_dropdown",       # 问答范围（全部 / 单篇文档）
    "doc_table",            # 文档列表
    "delete_kb_dropdown",   # 危险区：待删除知识库
    "delete_doc_dropdown",  # 危险区：待删除文档
    "kb_overview",          # 知识库概览
    "config_md",            # 运行环境自检
    "status_md",            # 左侧状态区
    "kb_name_box",          # 新建知识库输入框（建库后清空）
    "confirm_checkbox",     # 删除确认框（执行后复位）
    "rebuild_kb",           # 重建索引：目标知识库
    "doc_status_md",        # 「文档与索引」区的状态
    "upload_btn",           # 上传并建立索引（执行期间换成「停止」）
    "upload_stop_btn",      # 停止上传
    "rebuild_btn",          # 重建所选知识库的索引
    "rebuild_stop_btn",     # 停止重建
]

#: 提问事件的固定输出顺序（流式阶段只更新对话区与输入区，其余保持不变）
QUERY_KEYS: List[str] = [
    "chatbot",
    "question_box",
    "send_btn",
    "stop_btn",
    "session_state",
    "session_radio",
    "session_md",
    "references_md",
    "downloads",
    "summary_md",
]

CONFIRM_REQUIRED_MESSAGE = "删除不可恢复。请先勾选「确认要删除，此操作不可恢复」，再点击删除按钮。"

# 空状态：一句话，无插图
STATUS_EMPTY = "--"
REFS_EMPTY = "_提问后，这里会列出引用来源。_"
SUMMARY_EMPTY = "_等待提问_"
DOC_STATUS_EMPTY = "--"
SESSION_EMPTY = "当前：**新对话**"
#: 检索阶段的对话区提示（进度只出现在对话区，不打扰侧边栏）
RETRIEVING_HINT = "_正在检索相关片段…_"
#: 回答被「停止」且还没进入正文时，替换上面的检索提示
STOPPED_HINT = "_（已停止生成）_"
#: 停止后对话页底部那条状态栏的提示
STOPPED_SUMMARY = "⏹ 已停止生成，本轮未存档"

#: 「问答历史」面板最多展示多少条
HISTORY_LIMIT = 30
#: 会话类事件的固定输出顺序（避免输出错位）
SESSION_KEYS = [
    "chatbot",
    "session_state",
    "session_radio",
    "session_md",
    "references_md",
    "downloads",
    "summary_md",
    "scroll_box",
]


def _update(**kwargs: Any) -> Any:
    """构造组件增量更新。"""
    return gr.update(**kwargs)


def _message_text(content: Any) -> str:
    """把聊天消息的 ``content`` 取成纯文本。

    Gradio 6 的 ``Chatbot.preprocess`` 在事件之间传的是**分块结构**——
    ``[{"type": "text", "text": "..."}]``，而不是字符串。
    这里两种形态都兼容；按字符串硬处理会抛
    ``AttributeError: 'list' object has no attribute 'strip'``。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _pack_query(values: Dict[str, Any]) -> Tuple[Any, ...]:
    """按 :data:`QUERY_KEYS` 顺序拼装提问相关事件的输出。

    未在 ``values`` 中给出的组件一律用空更新占位（保持原值），
    这样新增组件只需改 :data:`QUERY_KEYS` 一处，不会出现输出错位。
    """
    return tuple(values.get(key, _update()) for key in QUERY_KEYS)


class KnowledgeQAApp:
    """系统总装配与事件编排。"""

    def __init__(self) -> None:
        config.ensure_dirs()
        self.metadata = MetadataStore()
        self.vector = VectorStore()
        self.documents = DocumentStore()
        self.parser = DocumentParser()
        self.qa = QAAgent()
        self.figures = FigureDetector(describe_fn=self.qa.complete)
        self.redactor = PIIRedactor(metadata_store=self.metadata)
        self.embedder = Embedder()
        self.locator = ReferenceLocator(self.documents)
        self.reranker = Reranker()
        self.retriever = Retriever(self.vector, self.embedder, self.metadata, self.reranker)
        self.router = Router(available_provider=self.metadata.kb_names)
        self.restorer = PlaceholderRestorer()
        #: 当前索引任务："" / "upload" / "rebuild"。同一时刻只允许一项，
        #: 也是「停止」按钮要知道该停哪一个的依据。
        self._index_task = ""
        #: 协作式取消信号：索引循环在检查点（向量化每批之前）读它
        self._index_cancel = threading.Event()
        #: 问答的协作式取消信号：检索各阶段之间、流式生成每段增量之前读它。
        #: 有它才能在点「停止」后立刻收手，而不是等 Gradio 把生成器关掉。
        self._query_cancel = threading.Event()

    # ==================================================================
    # 索引任务的运行状态
    # ==================================================================
    def _indexing_state(self, kind: str = "") -> Dict[str, Any]:
        """索引类控件的状态。

        :param kind: ``"upload"`` / ``"rebuild"`` 表示该任务正在执行——它自己的
            触发按钮换成「停止」，另一条的触发按钮置灰（同一时刻只允许一项索引任务）；
            传 ``""`` 表示空闲，触发按钮恢复可用、「停止」全部隐藏。
        """
        return {
            "upload_btn": _update(visible=kind != "upload", interactive=kind == ""),
            "upload_stop_btn": _update(visible=kind == "upload"),
            "rebuild_btn": _update(visible=kind != "rebuild", interactive=kind == ""),
            "rebuild_stop_btn": _update(visible=kind == "rebuild"),
        }

    def _busy_message(self, status_key: str) -> Dict[str, Any]:
        """已有索引任务在跑时，给另一次点击的提示（带上该任务名）。"""
        label = "上传并建立索引" if self._index_task == "upload" else "重建索引"
        return {
            status_key: f"正在{label}，请等它结束或点「停止」后再试。",
            **self._indexing_state(self._index_task),
        }

    def _begin_index_task(self, kind: str) -> None:
        self._index_task = kind
        self._index_cancel.clear()

    def _end_index_task(self) -> None:
        self._index_task = ""
        self._index_cancel.clear()

    def stop_indexing(self, _session_hash: str = "") -> Tuple[Any, ...]:
        """「停止」按钮：请求中断当前的索引任务，并把控件复位。

        取消是协作式的——正在跑的循环会在下一个检查点退出（向量化每批检查一次），
        已经处理完的文档保持已建好的索引；正在处理的那篇不会落库。
        """
        task = self._index_task
        self._index_cancel.set()
        self._index_task = ""
        state = self._indexing_state("")
        if task == "upload":
            state["status_md"] = "⏹ 已停止上传：已完成的文档保留了索引，未完成的不入库。"
        elif task == "rebuild":
            state["doc_status_md"] = "⏹ 已停止重建：已重建的文档用新索引，其余保持原索引。"
        return self._pack({}, **state)

    # ==================================================================
    # 列表与状态
    # ==================================================================
    def _doc_rows(self) -> List[List[Any]]:
        return [
            [
                doc["filename"],
                doc["kb_name"],
                "隐私" if doc["mode"] == "privacy" else "普通",
                doc["chunk_count"],
                doc["redaction_count"],
            ]
            for doc in self.metadata.list_documents()
        ]

    def _doc_choices(self) -> List[Tuple[str, str]]:
        return [
            (f"{doc['filename']} · {doc['kb_name']} · {doc['upload_time']}", doc["doc_id"])
            for doc in self.metadata.list_documents()
        ]

    def _scope_choices(self, kb_filter: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """问答范围下拉框选项：全部文档 + 所选知识库下的文档。

        只列出 ``kb_filter`` 中知识库的文档；未勾选任何知识库时不列出任何文档，
        因此未选中知识库的文档不会出现在这个列表里。
        """
        choices: List[Tuple[str, str]] = [("全部文档（按左侧勾选的知识库检索）", SCOPE_ALL)]
        if not kb_filter:
            return choices
        allowed = set(kb_filter)
        for doc in self.metadata.list_documents():
            if doc["kb_name"] not in allowed:
                continue
            label = f"{doc['filename']} · {doc['kb_name']} · {doc['upload_time']}"
            choices.append((label, doc["doc_id"]))
        return choices

    def _keep_scope(self, scope: Optional[str], kb_filter: Optional[List[str]]) -> str:
        """知识库筛选变化后尽量保留当前选择，已失效则回退为「全部文档」。"""
        values = {value for _label, value in self._scope_choices(kb_filter)}
        return scope if scope in values else SCOPE_ALL

    def _kb_overview(self) -> str:
        kbs = self.metadata.list_kbs()
        if not kbs:
            return "还没有知识库。先建一个，再上传文档。"
        lines = []
        for kb in kbs:
            mode = MODE_PRIVACY if (kb.get("mode") or "") == MODE_PRIVACY_VALUE else MODE_NORMAL
            lines.append(
                f"- `{kb['name']}` · **{mode}** · 文档 {kb['doc_count']} 个 · "
                f"向量块 {self.vector.count(kb['name'])} 个 · 脱敏项 "
                f"{self.metadata.count_redactions(kb['name'])} 条"
            )
        return "\n".join(lines)

    def config_markdown(self) -> str:
        lines = []
        for item in config.config_status():
            icon = "✅" if item["ok"] else "⚠️"
            text = f"- {icon} {item['name']}"
            if item["hint"]:
                text += f" —— {item['hint']}"
            lines.append(text)

        status = self.redactor.status()
        if not status["ner_enabled"]:
            lines.append("- ℹ️ 本地 NER 已关闭，隐私模式仅使用正则脱敏")
        elif status["ner_available"]:
            lines.append(f"- ✅ 本地 NER 已加载：`{status['model_id']}`")
        elif status["error"]:
            lines.append(f"- ⚠️ 本地 NER 不可用：{status['error']}")
        else:
            lines.append(f"- ℹ️ 本地 NER 待首次使用时加载：`{status['model_id']}`")

        icon = "✅" if self.reranker.is_available() else "ℹ️"
        lines.append(f"- {icon} 重排序：{self.reranker.describe()}")
        lines.append(f"- 📁 数据目录：`{config.DATA_DIR}`")
        lines.append(f"- 🧩 PDF 引擎：{config.PDF_ENGINE}")
        return "\n".join(lines)

    # ==================================================================
    # 会话与问答历史（新建对话 / 点击历史定位）
    # ==================================================================
    @staticmethod
    def _shorten(text: str, limit: int = 20) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    @staticmethod
    def _clock(created_at: str) -> str:
        return (created_at or "")[-8:-3] or "--:--"

    @staticmethod
    def _mode_label(mode: str) -> str:
        """问答记录里存的是界面层的中文模式名（普通模式 / 隐私模式）。"""
        return "隐私" if (mode or "") == MODE_PRIVACY else "普通"

    def _session_choices(self, limit: int = HISTORY_LIMIT) -> List[Tuple[str, str]]:
        """会话列表（可点击切换）：一个会话一条，值是该会话 id。

        按会话而不是按单轮归类，便于按对话主题追踪历史。
        """
        choices: List[Tuple[str, str]] = []
        for session in self.metadata.list_sessions(limit=limit):
            label = (
                f"{self._clock(session.get('updated_at') or '')} ｜ "
                f"{self._shorten(session.get('title') or '未命名对话', 16)} ｜ "
                f"{int(session.get('turn_count') or 0)} 轮"
            )
            choices.append((label, session["session_id"]))
        return choices

    def _session_md(self, session_id: str) -> str:
        """当前会话的一行摘要。"""
        if not session_id:
            return "当前：**新对话**（提问后自动创建会话）"
        session = self.metadata.get_session(session_id)
        if not session:
            return "当前：**新对话**（原有会话可能已被删除）"
        turns = len(self.metadata.session_records(session_id))
        title = session.get("title") or "未命名对话"
        return (
            f"当前：**{title}** ｜ {turns} 轮 ｜ "
            f"更新于 {self._clock(session.get('updated_at') or '')}"
        )

    def _session_updates(self, session_id: str) -> Tuple[Any, Any, Any]:
        """会话相关组件的统一刷新：会话状态 / 会话列表 / 会话摘要。"""
        return (
            _update(value=session_id or None),
            _update(choices=self._session_choices(), value=session_id or None),
            self._session_md(session_id),
        )

    def _load_session(self, session_id: str) -> List[Dict[str, str]]:
        """载入某个会话的全部消息（一问一答为一轮）。"""
        return self.metadata.session_messages(session_id) if session_id else []

    def _session_view(
        self,
        session_id: str,
        refs_md: Optional[str] = None,
        downloads: Optional[List[str]] = None,
        summary: str = "",
    ) -> Tuple[Any, ...]:
        """按 :data:`SESSION_KEYS` 的顺序拼装会话类事件的输出。"""
        return (
            self._load_session(session_id),
            *self._session_updates(session_id),
            refs_md if refs_md is not None else REFS_EMPTY,
            _update(value=downloads or None, visible=bool(downloads)),
            summary or SUMMARY_EMPTY,
            str(time.time()),
        )

    @staticmethod
    def _session_noop() -> Tuple[Any, ...]:
        return tuple(_update() for _ in SESSION_KEYS)

    def restore_history(self) -> Tuple[Any, ...]:
        """页面加载：恢复最近一次会话（存档默认生效）。"""
        return self._session_view(self.metadata.latest_session_id())

    def handle_new_session(self) -> Tuple[Any, ...]:
        """新建对话：清空聊天区，下一条提问会创建新会话。"""
        return self._session_view("", summary="新对话已就绪 ｜ 提问后会创建新的会话")

    def handle_history_jump(self, session_id: Optional[str]) -> Tuple[Any, ...]:
        """切换到历史会话：载入该会话的全部对话，并滚到最新一轮。"""
        session_id = (session_id or "").strip()
        if not session_id or not self.metadata.get_session(session_id):
            return self._session_noop()

        records = self.metadata.session_records(session_id)
        references = self._decode_refs(records[-1]) if records else []
        refs_md = (
            self.locator.render_markdown(references)
            if references
            else "_该会话暂无可展示的引用详情。_"
        )
        downloads = self.locator.export(references) if references else None
        last = records[-1] if records else {}
        summary = (
            f"已切换到历史会话（{len(records)} 轮）｜ 最近更新 {last.get('created_at', '')}"
            f" ｜ 范围：{last.get('kb_names') or '全部'} ｜ 模式：{last.get('mode') or MODE_NORMAL}"
        )
        return self._session_view(session_id, refs_md, downloads, summary)

    def handle_delete_session(self, session_id: Optional[str]) -> Tuple[Any, ...]:
        """删除当前会话，并切回最近的会话。"""
        session_id = (session_id or "").strip()
        if not session_id:
            return self._session_noop()

        removed = self.metadata.delete_session(session_id)
        next_id = self.metadata.latest_session_id()
        view = self._session_view(next_id, summary=f"已删除该会话（{removed} 轮）。")
        return view

    @staticmethod
    def _decode_refs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            refs = json.loads(record.get("refs_json") or "[]")
        except Exception:
            return []
        return [ref for ref in refs if isinstance(ref, dict)]

    def handle_history_clear(
        self,
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Tuple[Any, ...]:
        """清空全部问答历史、会话与复用缓存。"""
        count = self.metadata.clear_queries()
        view = self._session_view("", summary=f"已清空 {count} 条问答历史与会话。")
        return self._pack(
            self._list_state(selected, scope),
            status_md=f"已清空 {count} 条问答历史与缓存，下次提问将重新检索。",
        ) + view

    @staticmethod
    def _fingerprint(
        question: str, kbs: List[str], scope: str, mode: str, strict: bool = True
    ) -> str:
        """问题 + 范围 + 模式 + 脱敏档位一致，才视为同一个问题。

        范围与模式参与指纹，避免"单篇文档"的答案被"全库提问"复用；
        脱敏档位参与指纹，避免"保留占位符"的答案被"还原真实信息"复用。
        """
        raw = "|".join(
            [
                " ".join((question or "").split()),
                "、".join(sorted(kbs or [])),
                scope or SCOPE_ALL,
                mode or "",
                "strict" if strict else "restore",
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _list_state(
        self,
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Dict[str, Any]:
        """计算所有列表型组件的最新取值。

        :param selected: 当前勾选的问答知识库，同时用于筛选「问答范围」的可选项
        :param scope: 当前问答范围，若因知识库筛选变化而失效会自动回退
        """
        names = self.metadata.kb_names()
        kb_choices = [(name, name) for name in names]
        keep = [name for name in (selected or []) if name in names]
        default_kb = names[0] if names else None
        return {
            "kb_checkbox": _update(choices=kb_choices, value=keep),
            "upload_kb": _update(choices=kb_choices, value=default_kb),
            "rebuild_kb": _update(choices=kb_choices, value=default_kb),
            "delete_kb_dropdown": _update(choices=kb_choices, value=None),
            "scope_dropdown": _update(
                choices=self._scope_choices(keep),
                value=self._keep_scope(scope, keep),
            ),
            "doc_table": self._doc_rows(),
            "delete_doc_dropdown": _update(choices=self._doc_choices(), value=None),
            "kb_overview": self._kb_overview(),
            "config_md": self.config_markdown(),
        }

    def handle_kb_change(
        self,
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Any:
        """知识库勾选变化时刷新「问答范围」：未勾选知识库的文档不再出现。"""
        names = self.metadata.kb_names()
        keep = [name for name in (selected or []) if name in names]
        return _update(
            choices=self._scope_choices(keep),
            value=self._keep_scope(scope, keep),
        )

    def _pack(self, state: Dict[str, Any], **overrides: Any) -> Tuple[Any, ...]:
        """按 :data:`CANON_KEYS` 的固定顺序拼装输出元组。"""
        merged = dict(state)
        merged.update(overrides)
        return tuple(merged.get(key, _update()) for key in CANON_KEYS)

    def refresh_ui(self, selected: Optional[List[str]] = None) -> Tuple[Any, ...]:
        """刷新所有列表型组件（页面加载时调用）。"""
        return self._pack(self._list_state(selected))

    @staticmethod
    def _scope_label(scoped_doc: Optional[Dict[str, Any]], kbs: List[str]) -> str:
        if scoped_doc:
            return f"{scoped_doc.get('filename', '未知文档')}（单篇文档）"
        if kbs:
            return "、".join(kbs)
        return "全部知识库"

    def _resolve_mode(self, kbs: List[str], scoped_doc: Optional[Dict[str, Any]] = None) -> str:
        """问答模式跟随知识库：涉及的知识库里有隐私库，就按隐私模式处理。"""
        names = [scoped_doc.get("kb_name") or ""] if scoped_doc else list(kbs or [])
        modes = [self.metadata.kb_mode(name) for name in names if name]
        return (
            MODE_PRIVACY_VALUE
            if MODE_PRIVACY_VALUE in modes
            else MODE_NORMAL_VALUE
        )

    # ==================================================================
    # 知识库管理
    # ==================================================================
    def handle_create_kb(
        self,
        name: str,
        kb_mode: str = MODE_NORMAL_VALUE,
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Tuple[Any, ...]:
        name = (name or "").strip()
        if not name:
            return self._pack(self._list_state(selected, scope), status_md="知识库名称不能为空。")

        mode = kb_mode if kb_mode in (MODE_NORMAL_VALUE, MODE_PRIVACY_VALUE) else MODE_NORMAL_VALUE
        try:
            self.metadata.ensure_kb(
                name, collection=collection_name(name), mode=mode
            )
            self.vector.get_or_create(name)
            label = MODE_PRIVACY if mode == MODE_PRIVACY_VALUE else MODE_NORMAL
            message = f"知识库 `{name}` 已就绪（模式：{label}）。库内文档将一律按此模式处理。"
        except Exception as exc:  # noqa: BLE001
            message = f"创建知识库失败：{exc}"

        keep = list(dict.fromkeys(list(selected or []) + [name]))
        return self._pack(self._list_state(keep, scope), status_md=message, kb_name_box="")

    def handle_delete_kb(
        self,
        name: Optional[str],
        selected: Optional[List[str]] = None,
        confirmed: bool = False,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Tuple[Any, ...]:
        log = get_logger()
        state = self._list_state(selected, scope)
        name = (name or "").strip()
        if not name:
            log.warning("删除知识库 被拦截(未选择知识库) confirmed=%s", confirmed)
            return self._pack(state, status_md="请先选择要删除的知识库。", confirm_checkbox=False)
        if not confirmed:
            log.warning("删除知识库 被拦截(未勾确认框) kb=%s", name)
            return self._pack(state, status_md=CONFIRM_REQUIRED_MESSAGE, confirm_checkbox=False)

        try:
            docs = self.metadata.list_documents(name)
            for doc in docs:
                self.documents.delete_by_path(doc["file_path"])
                self.metadata.delete_chunk_texts(doc["doc_id"])
            self.vector.delete_collection(name)
            self.metadata.delete_kb(name)
            # 索引变了，历史答案可能已过期，缓存整体失效
            self.metadata.clear_cached_answers()
        except Exception as exc:  # noqa: BLE001
            # 真实删除流程里任何一步抛异常都会被记下，并在状态栏给出明确提示，
            # 而不是让 Gradio 默默吞掉——这正是「删除不成功但不知为何」的根因排查点。
            log.error("删除知识库 失败 kb=%s 含文档数=%d 错误=%s", name, len(docs), exc)
            return self._pack(
                state,
                status_md=f"删除知识库 `{name}` 失败：{exc}。详情见 data/logs/smardock.log。",
                confirm_checkbox=False,
            )
        log.info("删除知识库 成功 kb=%s 含文档数=%d", name, len(docs))
        message = (
            f"已删除知识库 `{name}`（含 {len(docs)} 个文档）。"
            "脱敏映射表保留，以保证其它文档的还原不受影响。"
        )
        keep = [kb for kb in (selected or []) if kb != name]
        return self._pack(self._list_state(keep, scope), status_md=message, confirm_checkbox=False)

    # ==================================================================
    # 文档上传与索引
    # ==================================================================
    def handle_upload(
        self,
        files: Optional[List[str]],
        kb_name: Optional[str],
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Generator[Tuple[Any, ...], None, None]:
        """上传文档并建立索引。处理模式由目标知识库决定，上传时不再单独选择。

        执行期间可随时点「停止」：取消是协作式的（向量化每批检查一次信号），
        已经处理完的文档保留索引，正在处理的那篇不会落库。
        """
        kb_name = (kb_name or "").strip()
        state = self._list_state(selected, scope)
        if self._index_task:
            yield self._pack(state, **self._busy_message("status_md"))
            return
        if not kb_name:
            yield self._pack(state, status_md="请先选择上传目标知识库。")
            return
        if not files:
            yield self._pack(state, status_md="请先选择要上传的 PDF / Word 文件。")
            return
        if not self.embedder.is_available():
            yield self._pack(
                state,
                status_md="向量化器未配置，无法建立索引。请在 `.env` 中设置 `DASHSCOPE_API_KEY`。",
            )
            return

        # 处理模式跟随知识库：同一个库里的文档都按建库时选定的模式处理
        privacy = self.metadata.kb_mode(kb_name) == MODE_PRIVACY_VALUE
        self.metadata.ensure_kb(kb_name, collection=collection_name(kb_name))
        if privacy:
            self.redactor.load_ner()

        logs: List[str] = [
            f"目标知识库：`{kb_name}` ｜ 模式：{MODE_PRIVACY if privacy else MODE_NORMAL}",
            "",
        ]
        total = len(files)
        done = 0

        self._begin_index_task("upload")
        try:
            for index, file_path in enumerate(files, start=1):
                if self._index_cancel.is_set():
                    break

                name = Path(file_path).name
                logs.append(f"[{index}/{total}] ⏳ {name} —— 处理中…")
                yield self._pack(
                    self._list_state(selected, scope),
                    status_md=self._render_logs(logs, index - 1, total),
                    **self._indexing_state("upload"),
                )

                try:
                    detail = self._ingest_one(file_path, kb_name, privacy)
                    logs[-1] = f"[{index}/{total}] ✅ {name} —— {detail}"
                except EmbeddingCancelled:
                    logs[-1] = f"[{index}/{total}] ⏹ {name} —— 已停止，未入库"
                    break
                except UnsupportedDocumentError as exc:
                    logs[-1] = f"[{index}/{total}] ⚠️ {name} —— {exc}"
                except EmbeddingError as exc:
                    logs[-1] = f"[{index}/{total}] ❌ {name} —— 向量化失败：{exc}"
                except Exception as exc:  # noqa: BLE001
                    logs[-1] = f"[{index}/{total}] ❌ {name} —— {type(exc).__name__}: {exc}"

                done = index
                yield self._pack(
                    self._list_state(selected, scope),
                    status_md=self._render_logs(logs, index, total),
                )

            if self._index_cancel.is_set():
                logs.append("已停止：未处理完的文档不会入库，已完成的文档保持索引。")
            else:
                if privacy:
                    status = self.redactor.status()
                    if status["error"]:
                        logs.append(f"> {status['error']}")
                logs.append("索引完成。可在右侧「问答范围」中选择具体文档做单篇问答。")
                # 新增了内容，历史答案可能已过期，缓存整体失效
                self.metadata.clear_cached_answers()

            yield self._pack(
                self._list_state(selected, scope),
                status_md=self._render_logs(logs, done, total),
                **self._indexing_state(""),
            )
        finally:
            self._end_index_task()

    @staticmethod
    def _render_logs(logs: List[str], done: int, total: int) -> str:
        """把进度渲染成状态区里的一行文本进度条。

        说明：这里刻意不用 ``gr.Progress()`` —— 它会把进度条盖在该事件的
        **所有**输出组件上，而上传事件的输出几乎覆盖整个左侧栏，会导致文字重叠。
        """
        width = max(1, min(int(total), 20))
        filled = int(round(done * width / max(1, total)))
        bar = "▓" * filled + "░" * (width - filled)
        return f"{bar} {done}/{total}\n\n" + "\n\n".join(logs)

    def _prepare_index(
        self,
        path: Path,
        doc_id: str,
        kb_name: str,
        filename: str,
        privacy: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[List[float]], int]:
        """解析 → 图表识别 → 脱敏 → 分块 → 向量化，返回全部待落库数据。

        上传与重建索引共用这段流程，保证两条路径产出的索引完全一致。
        """
        blocks = self.parser.parse(path)
        if not blocks:
            raise ValueError("未能从文档中解析出任何内容")

        blocks = self.figures.detect(blocks)

        # 隐私模式下先脱敏：此后任何一次云端调用都只见占位符
        redaction_count = 0
        if privacy:
            blocks, _mapping, redaction_count = self.redactor.redact_blocks(
                blocks, kb_name=kb_name, doc_id=doc_id
            )

        # 图表 AI 描述必须排在脱敏之后：该请求会把图表上下文发往云端
        if config.FIGURE_AI_DESCRIBE:
            blocks = self.figures.describe_figures(blocks)

        chunks = self.parser.build_chunks(blocks, doc_id, filename, kb_name)
        if not chunks:
            raise ValueError("未解析到可索引的文本内容")

        contents = [chunk["content"] for chunk in chunks]
        # 向量化是最耗时的一段，也是唯一方便插入取消检查点的地方：
        # 传进运行中的取消信号，用户点「停止」后最多再等一批请求。
        embeddings = self.embedder.embed_texts(
            contents, cancel_check=self._index_cancel.is_set
        )
        return blocks, chunks, contents, embeddings, redaction_count

    def _ingest_one(self, file_path: str, kb_name: str, privacy: bool) -> str:
        doc_id = uuid.uuid4().hex[:12]

        # 1) 归档原件
        saved_path = self.documents.save(file_path, kb_name)

        # 2~6) 解析 → 脱敏 → 分块 → 向量化
        try:
            blocks, chunks, contents, embeddings, redaction_count = self._prepare_index(
                saved_path, doc_id, kb_name, saved_path.name, privacy
            )
        except Exception:
            # 中途失败或被「停止」：归档的原件一起回滚，
            # 否则会留下一个没有任何索引、界面也看不到的孤儿文件。
            self.documents.delete_by_path(saved_path)
            raise

        # 7) 落库：先文档、再块索引、再向量
        self.metadata.add_document(
            doc_id=doc_id,
            filename=saved_path.name,
            kb_name=kb_name,
            file_path=str(saved_path),
            mode="privacy" if privacy else "normal",
            chunk_count=len(chunks),
            redaction_count=redaction_count,
        )
        self.metadata.add_chunks(chunks)
        # 关键词索引（BM25），与向量检索互补
        self.metadata.index_chunk_texts(chunks)

        metadatas = [self._chunk_metadata(chunk, saved_path) for chunk in chunks]
        self.vector.add_documents(
            kb_name,
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        detail = f"解析 {len(blocks)} 个内容块 → 索引 {len(chunks)} 个向量块"
        if privacy:
            detail += f"；脱敏 {redaction_count} 处"
        return detail

    # ------------------------------------------------------------------
    # 重建索引
    # ------------------------------------------------------------------
    def _reindex_one(self, doc: Dict[str, Any]) -> str:
        """重建单篇文档的索引。

        顺序上**先把新索引算完，再替换旧的**，这样中途失败不会让文档变成"没有索引"
        的空壳。文档 id 与处理模式沿用原记录，隐私文档的脱敏占位符也会复用原映射。
        """
        path = Path(str(doc.get("file_path") or ""))
        if not path.exists():
            raise FileNotFoundError(f"归档原件不存在：{path}")

        doc_id = doc["doc_id"]
        kb_name = doc["kb_name"]
        privacy = doc.get("mode") == "privacy"

        blocks, chunks, contents, embeddings, redaction_count = self._prepare_index(
            path, doc_id, kb_name, doc["filename"], privacy
        )

        # 替换旧索引（向量 / 关键词 / 块索引三处一起清）
        self.vector.delete_by_doc(kb_name, doc_id)
        self.metadata.delete_chunk_texts(doc_id)
        self.metadata.delete_chunks_by_doc(doc_id)

        metadatas = [self._chunk_metadata(chunk, path) for chunk in chunks]
        self.vector.add_documents(
            kb_name,
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self.metadata.add_chunks(chunks)
        self.metadata.index_chunk_texts(chunks)
        self.metadata.update_document_stats(doc_id, len(chunks), redaction_count)

        detail = f"解析 {len(blocks)} 个内容块 → 索引 {len(chunks)} 个向量块"
        if privacy:
            detail += f"；脱敏 {redaction_count} 处"
        return detail

    def handle_rebuild(
        self,
        kb_name: Optional[str],
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Generator[Tuple[Any, ...], None, None]:
        """按知识库重建索引（解析引擎或分块策略更新后使用）。

        与上传共用一套取消机制：执行期间可点「停止」，已重建的文档保留新索引，
        正在重建的那篇沿用原索引（新索引算完才替换旧的）。
        """
        state = self._list_state(selected, scope)
        kb_name = (kb_name or "").strip()
        if self._index_task:
            yield self._pack(state, **self._busy_message("doc_status_md"))
            return
        if not kb_name:
            yield self._pack(state, doc_status_md="请先选择要重建索引的知识库。")
            return

        docs = self.metadata.list_documents(kb_name)
        if not docs:
            yield self._pack(state, doc_status_md=f"知识库 `{kb_name}` 下没有文档。")
            return
        if not self.embedder.is_available():
            yield self._pack(
                state, doc_status_md="向量化器未配置，无法重建索引。请设置 `DASHSCOPE_API_KEY`。"
            )
            return

        logs: List[str] = [f"重建知识库：`{kb_name}` ｜ 共 {len(docs)} 篇文档", ""]
        total = len(docs)
        done = 0

        self._begin_index_task("rebuild")
        try:
            for index, doc in enumerate(docs, start=1):
                if self._index_cancel.is_set():
                    break

                name = doc["filename"]
                logs.append(f"[{index}/{total}] ⏳ {name} —— 重新解析并向量化…")
                yield self._pack(
                    self._list_state(selected, scope),
                    doc_status_md=self._render_logs(logs, index - 1, total),
                    **self._indexing_state("rebuild"),
                )

                try:
                    detail = self._reindex_one(doc)
                    logs[-1] = f"[{index}/{total}] ✅ {name} —— {detail}"
                except EmbeddingCancelled:
                    logs[-1] = f"[{index}/{total}] ⏹ {name} —— 已停止，沿用原索引"
                    break
                except Exception as exc:  # noqa: BLE001
                    logs[-1] = f"[{index}/{total}] ❌ {name} —— {type(exc).__name__}: {exc}"

                done = index
                yield self._pack(
                    self._list_state(selected, scope),
                    doc_status_md=self._render_logs(logs, index, total),
                )

            if self._index_cancel.is_set():
                logs.append("已停止：已重建的文档用新索引，其余保持原索引。")
            else:
                logs.append("重建完成，新的解析与分块策略已生效。")
                # 索引变了，历史答案可能已过期，缓存整体失效
                self.metadata.clear_cached_answers()

            yield self._pack(
                self._list_state(selected, scope),
                doc_status_md=self._render_logs(logs, done, total),
                **self._indexing_state(""),
            )
        finally:
            self._end_index_task()

    @staticmethod
    def _chunk_metadata(chunk: Dict[str, Any], saved_path: Path) -> Dict[str, Any]:
        page = chunk.get("page")
        try:
            page_int = int(page) if page not in (None, "") else 0
        except (TypeError, ValueError):
            page_int = 0
        return {
            "doc_id": chunk["doc_id"],
            "kb_name": chunk["kb_name"],
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "file_path": str(saved_path),
            "page": page_int,
            "type": chunk.get("type", "text"),
            "position": json.dumps(chunk.get("position") or {}, ensure_ascii=False),
            "context": (chunk.get("context") or "")[:500],
            # 章节信息用于检索后的「整节展开」，让列表/步骤不会被切成半段喂给模型
            "section": chunk.get("section") or "",
            "section_id": chunk.get("section_id") or "",
        }

    def handle_delete_document(
        self,
        doc_id: Optional[str],
        confirmed: bool = False,
        selected: Optional[List[str]] = None,
        scope: Optional[str] = SCOPE_ALL,
    ) -> Tuple[Any, ...]:
        log = get_logger()
        state = self._list_state(selected, scope)
        if not doc_id:
            return self._pack(state, status_md="请先选择要删除的文档。", confirm_checkbox=False)
        if not confirmed:
            log.warning("删除文档 被拦截(未勾确认框) doc_id=%s", doc_id)
            return self._pack(state, status_md=CONFIRM_REQUIRED_MESSAGE, confirm_checkbox=False)

        try:
            doc = self.metadata.delete_document(doc_id)
            if not doc:
                log.warning("删除文档 未命中(文档不存在或已删) doc_id=%s", doc_id)
                return self._pack(state, status_md="文档不存在或已被删除。", confirm_checkbox=False)
            self.vector.delete_by_doc(doc["kb_name"], doc_id)
            self.metadata.delete_chunk_texts(doc_id)
            self.metadata.clear_cached_answers()
            self.documents.delete_by_path(doc["file_path"])
        except Exception as exc:  # noqa: BLE001
            log.error("删除文档 失败 doc_id=%s 错误=%s", doc_id, exc)
            return self._pack(
                state,
                status_md=f"删除文档失败：{exc}。详情见 data/logs/smardock.log。",
                confirm_checkbox=False,
            )
        log.info("删除文档 成功 doc_id=%s kb=%s filename=%s", doc_id, doc["kb_name"], doc["filename"])
        message = f"已删除文档 `{doc['filename']}`（知识库 `{doc['kb_name']}`）。"
        return self._pack(
            self._list_state(selected, scope), status_md=message, confirm_checkbox=False
        )

    # ==================================================================
    # 问答（流式）
    # ==================================================================
    def _render_answer(
        self,
        raw: str,
        mapping: Dict[str, str],
        ref_meta: Dict[int, str],
    ) -> str:
        """把模型输出渲染成聊天区可用的 HTML：转义 → 占位符还原 → 引用编号转上标。

        * 转义：文档与模型输出原样展示，避免内容里的 HTML 被当作标签执行；
          Markdown 语法（加粗、列表、表格、代码块）不受影响。
        * 引用上标带 ``title``，hover 才显示来源，正文保持干净。
        """
        text = html.escape(raw or "", quote=False)
        if mapping:
            safe_mapping = {key: html.escape(value, quote=False) for key, value in mapping.items()}
            text = self.restorer.restore(text, safe_mapping)

        def replace_citation(match: "re.Match[str]") -> str:
            index = int(match.group(1))
            tip = ref_meta.get(index)
            if not tip:
                return match.group(0)
            return (
                f'<sup class="kv-cite" title="{html.escape(tip, quote=True)}">{index}</sup>'
            )

        return CITATION_RE.sub(replace_citation, text)

    def stop_query(self, _session_hash: str = "") -> Tuple[Any, ...]:
        """「停止」按钮：请求中止当前回答，并立刻把输入区恢复可用。

        为什么不能只靠前端取消：Gradio 收到取消后会等生成器**停在 yield 上**才关得掉
        （``aclose`` 每 50ms 重试），而检索阶段中间没有 yield，输入区会一直锁着。
        所以这里自带一个后端处理器，先把界面解冻；真正在跑的检索/生成靠
        :attr:`_query_cancel` 信号在下一个检查点收手。
        """
        self._query_cancel.set()
        return _pack_query(
            {
                "question_box": _update(value="", interactive=True),
                "send_btn": _update(visible=True),
                "stop_btn": _update(visible=False),
                "summary_md": STOPPED_SUMMARY,
            }
        )

    def handle_query(
        self,
        question: str,
        kb_names: Optional[List[str]],
        history: Optional[List[Dict[str, str]]],
        scope: Optional[str] = None,
        reuse: bool = True,
        session_id: str = "",
    ) -> Generator[Tuple[Any, ...], None, None]:
        """生成器：检索后流式输出回答（实现见 :meth:`_query_stream`）。

        这一层只做一件事：把未预期的异常收敛成对话区里的一条提示。
        提问事件的输出横跨整个侧边栏（会话、引用、下载、状态栏…），而 Gradio 在
        事件抛异常时会给**该事件的所有输出组件**打上错误标志——多个面板会同时出现
        需要手动关闭的红叉。异常在这里就地消化，错误只出现在回答区。
        """
        #: 让内层把"当前对话展示内容"挂出来，出错时才有正确的上下文可拼提示
        live: Dict[str, Any] = {}
        try:
            yield from self._query_stream(
                question, kb_names, history, scope, reuse, session_id, live
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            yield _pack_query(
                {
                    "chatbot": self._error_history(live.get("history") or history, exc),
                    "question_box": _update(value="", interactive=True),
                    "send_btn": _update(visible=True),
                    "stop_btn": _update(visible=False),
                    "summary_md": f"回答中断：{type(exc).__name__}",
                }
            )

    @staticmethod
    def _error_history(
        history: Optional[List[Dict[str, Any]]], exc: BaseException
    ) -> List[Dict[str, Any]]:
        """把异常写进对话区：接到未完成的那条回答上，没有占位才新起一条。"""
        items = [dict(item) for item in (history or [])]
        message = f"⚠️ 回答生成中断：{type(exc).__name__}: {exc}"
        if items and (items[-1].get("role") or "") == "assistant":
            text = _message_text(items[-1].get("content")).strip()
            if not text or text == RETRIEVING_HINT:
                items[-1] = {"role": "assistant", "content": message}
                return items
        items.append({"role": "assistant", "content": message})
        return items

    def _query_stream(
        self,
        question: str,
        kb_names: Optional[List[str]],
        history: Optional[List[Dict[str, str]]],
        scope: Optional[str] = None,
        reuse: bool = True,
        session_id: str = "",
        live: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[Any, ...], None, None]:
        """生成器：检索后流式输出回答。

        :param scope: ``SCOPE_ALL`` 表示按勾选的知识库检索；传文档 ID 表示单篇文档问答。
        :param reuse: 命中存档记录时直接复用答案，不再调用模型（索引变更后缓存会自动失效）。
        :param session_id: 当前会话；为空时在首次提问后自动创建。
        :param live: 可变的运行状态容器，随时写入当前对话展示内容，
            供 :meth:`handle_query` 在出错时拼错误提示用（见 ``live["history"]``）。

        脱敏口径由**所涉及的知识库模式**决定（模式在建库时确定）：
            隐私库 —— 入库前脱敏（向量库只见占位符）、提问脱敏后才发往云端、
                      回答与引用也保留占位符；
            普通库 —— 不脱敏、不还原，原文原样。
        """
        history = [dict(item) for item in (history or [])]
        question = (question or "").strip()
        # 新一轮提问作废上一轮的停止信号（用户又发问了，说明要接着用）
        self._query_cancel.clear()
        #: 会话 id 会随着本轮提问被创建 / 复用，用可变容器让 emit 始终取到最新值
        current = {"session_id": (session_id or "").strip()}
        # 实际模式在路由确定范围后计算（见下方 _resolve_mode）
        current["privacy"] = False

        def composer(running: bool) -> Dict[str, Any]:
            """输入区状态：回答进行中时锁定输入，只留「停止」按钮可点。"""
            return {
                "question_box": _update(value="", interactive=not running),
                "send_btn": _update(visible=not running),
                "stop_btn": _update(visible=running),
            }

        def emit(
            hist: List[Dict[str, str]],
            refs_md: str = REFS_EMPTY,
            downloads: Optional[List[str]] = None,
            summary: str = "",
            streaming: bool = False,
            running: Optional[bool] = None,
        ) -> Tuple[Any, ...]:
            """按 :data:`QUERY_KEYS` 顺序拼装输出。

            ``streaming=True`` 时只有对话区被更新，其余组件一律保持原值——
            否则每一段增量都会给侧边栏所有面板打上"进行中"状态。

            ``running=True`` 表示本轮回答尚未结束：输入区被锁住、只留「停止」，
            这样回答期间连点「发送」或在输入框回车都不会再触发新一轮问答。
            缺省跟随 ``streaming``——流式阶段视为进行中，其余视为已结束。
            """
            if running is None:
                running = streaming
            values: Dict[str, Any] = {"chatbot": hist, **composer(running)}
            if streaming:
                return _pack_query(values)

            session_state, session_radio, session_md = self._session_updates(
                current["session_id"]
            )
            values.update(
                {
                    "session_state": session_state,
                    "session_radio": session_radio,
                    "session_md": session_md,
                    "references_md": refs_md,
                    "downloads": _update(
                        value=downloads or None, visible=bool(downloads)
                    ),
                    "summary_md": summary,
                }
            )
            return _pack_query(values)

        if not question:
            yield emit(history, summary="请输入问题。")
            return

        notes: List[str] = []

        # 0) 解析问答范围：选定单篇文档时忽略知识库多选框
        scoped_doc: Optional[Dict[str, Any]] = None
        if scope and scope != SCOPE_ALL:
            scoped_doc = self.metadata.get_document(scope)
            if not scoped_doc:
                notes.append("所选文档已不存在，本次已自动回退为按知识库检索。")

        # 1) 路由决策：没有可用范围时不调用大模型
        route = self.router.route(kb_names, scoped_doc=scoped_doc)
        if not route.ok:
            history.append({"role": "assistant", "content": route.message})
            yield emit(history, summary=route.message)
            return

        history.append({"role": "user", "content": question})
        if live is not None:
            # 从这里开始，出错时至少能把用户的问题一起展示出来
            live["history"] = history
        if route.message:
            notes.append(route.message)
        scope_label = self._scope_label(scoped_doc, route.kbs)

        # 1.1) 模式跟随知识库：涉及的知识库里只要有隐私库，就按隐私模式处理
        mode_value = self._resolve_mode(route.kbs, scoped_doc)
        privacy = mode_value == MODE_PRIVACY_VALUE
        mode = MODE_PRIVACY if privacy else MODE_NORMAL
        # 隐私模式默认严格：回答里也不出现真实信息
        strict_mode = privacy and bool(getattr(config, "PRIVACY_STRICT_REDACTION", True))
        # 只有隐私模式且在非严格档位下才做本地还原，普通模式一律不动原文
        restore_enabled = privacy and not strict_mode
        current["privacy"] = privacy

        # 1.2) 隐私模式：提问本身也先脱敏，避免用户把真实身份信息带进云端请求
        query_text = question
        if privacy:
            question_result = self.redactor.redact_text(question, kb_name="", doc_id="")
            if question_result.get("has_sensitive"):
                query_text = question_result["redacted_content"]
                notes.append("提问中的敏感信息已在本地脱敏后才发往云端")

        # 1.5) 命中存档记录：直接复用，省掉一次检索与模型调用
        fingerprint = self._fingerprint(
            question, route.kbs, scope or SCOPE_ALL, mode, strict_mode
        )
        if reuse:
            cached = self.metadata.find_cached_answer(fingerprint)
            if cached and cached.get("answer"):
                history.append({"role": "assistant", "content": cached["answer"]})
                yield emit(
                    history,
                    cached.get("references_md") or REFS_EMPTY,
                    cached.get("downloads") or None,
                    f"命中历史记录（{cached.get('created_at', '')}），未重复调用模型 ｜ 范围：{scope_label}",
                )
                return

        if not self.embedder.is_available():
            message = "向量化器未配置，无法检索。请在 `.env` 中设置 `DASHSCOPE_API_KEY`。"
            yield emit(history + [{"role": "assistant", "content": message}], summary=message)
            return
        if not self.qa.is_available():
            message = "问答模型未配置，无法生成回答。请在 `.env` 中设置 `DEEPSEEK_API_KEY`。"
            yield emit(history + [{"role": "assistant", "content": message}], summary=message)
            return

        # 2) 检索：向量 + 关键词融合 → 可选精排 → 整节展开
        #    （单篇文档范围时按 doc_id 过滤）
        #    检索可能持续数秒，先在对话区给出可见反馈——进度只出现在对话区
        history.append({"role": "assistant", "content": ""})
        display = [dict(item) for item in history]
        display[-1]["content"] = RETRIEVING_HINT
        if live is not None:
            # display 会被流式循环原地改写，这里存一份引用即可拿到最新展示内容
            live["history"] = display
        yield emit(
            [dict(item) for item in display], summary="正在检索相关片段…", running=True
        )

        doc_ids = [scoped_doc["doc_id"]] if scoped_doc else None
        # 隐私模式下用脱敏后的提问做检索：与库内占位符一致，召回反而更准
        hits = self.retriever.retrieve(
            route.kbs, query_text, doc_ids=doc_ids, cancel_check=self._query_cancel.is_set
        )
        if self._query_cancel.is_set():
            # 检索期间点了「停止」：不再拼上下文、不调用模型，直接把占位换成停止提示
            history[-1]["content"] = STOPPED_HINT
            yield emit(history, summary=STOPPED_SUMMARY)
            return
        if not hits:
            message = f"未在 {scope_label} 中检索到相关内容，请确认文档已成功上传并索引。"
            yield emit(history[:-1] + [{"role": "assistant", "content": message}], summary=message)
            return

        contexts = [
            {
                "index": index,
                "content": hit.get("content", ""),
                "source_kb": hit.get("source_kb", ""),
                "file_name": (hit.get("metadata") or {}).get("source_file", ""),
                "page": (hit.get("metadata") or {}).get("page") or None,
            }
            for index, hit in enumerate(hits, start=1)
        ]
        # 引用定位提前到这里：上标 hover 与引用面板、下载副本共用同一份打码映射
        all_references = self.locator.locate_many(hits)
        masked = (
            self.locator.masked_name_map(all_references)
            if privacy and getattr(config, "PRIVACY_MASK_SOURCE", True)
            else {}
        )
        # 上标按「知识片段序号」索引，与 contexts 顺序一一对应
        ref_meta = {
            context["index"]: self.locator.source_label(
                all_references[context["index"] - 1], masked
            )
            for context in contexts
        }
        mapping = self.metadata.get_redaction_map()

        # 3) 流式生成：每段增量都基于完整累积文本重算，内容只增不减
        #    （占位回答已在检索前压入 history，这里直接复用同一个位置）
        display = [dict(item) for item in history]
        if live is not None:
            # 这里重新绑定过 display，必须同步一次，错误提示才会接在已流出的正文后面
            live["history"] = display
        # 隐私模式（严格档位）不给还原映射：回答与引用全程保持占位符，真实信息不出本地库
        view_mapping = mapping if restore_enabled else {}
        raw = ""

        stopped = False
        try:
            for chunk in self.qa.answer_stream(
                query_text, contexts, mode="privacy" if privacy else "normal"
            ):
                if self._query_cancel.is_set():
                    stopped = True
                    break
                raw += chunk
                display[-1]["content"] = self._render_answer(raw, view_mapping, ref_meta)
                # 只推对话区：侧边栏面板保持原值，不会整屏显示"进行中"
                yield emit([dict(item) for item in display], streaming=True)
        except LLMError as exc:
            message = f"生成回答失败：{exc}"
            yield emit(
                history[:-1] + [{"role": "assistant", "content": message}], summary=message
            )
            return

        if stopped:
            # 中途停止：保留已生成的部分，但不写会话存档、不进复用缓存
            partial = (display[-1].get("content") or "").strip()
            display[-1]["content"] = (
                f"{partial}\n\n> {STOPPED_HINT.strip('_')}" if partial else STOPPED_HINT
            )
            yield emit([dict(item) for item in display], summary=STOPPED_SUMMARY)
            return

        # 4) 引用选择（引用对象已在生成前定位好）
        cited = self.qa.extract_citations(raw)
        references = [all_references[i - 1] for i in cited if 1 <= i <= len(all_references)]
        if not references:
            references = all_references[:3]

        # 5) 占位符还原：仅隐私模式 + 非严格档位才做；普通模式不动原文
        restored = (
            restore_enabled and bool(mapping) and self.restorer.contains_known(raw, mapping)
        )
        if mapping:
            if restore_enabled:
                references = self.restorer.restore_references(references, mapping)
            unresolved = self.restorer.find_unresolved(raw, mapping)
            if unresolved:
                notes.append(
                    "以下占位符未在脱敏映射表中找到对应原文，可能是模型自行生成的："
                    + "、".join(sorted(set(unresolved)))
                )

        if self.qa.notes:
            notes.extend(note for note in self.qa.notes if note not in notes)

        # 6) 引用展示与原件导出（隐私模式下文件名、路径、下载副本统一打码）
        downloads = self.locator.export(references, masked=masked or None)
        references_md = self.locator.render_markdown(references, masked=masked or None)

        # 对话页底部那条状态栏：只放简要信息
        summary = [
            f"引用 {len(references)} 条",
            f"范围：{scope_label}",
            f"模式：{mode}",
        ]
        if privacy:
            summary.append("已脱敏（回答保留占位符）" if strict_mode else "已在本地还原真实信息")
        summary.extend(notes)

        # 7) 会话归属：首次提问时创建会话，标题取首个问题
        current["session_id"] = current["session_id"] or self.metadata.create_session(
            title=self._shorten(question, 24), kb_names=route.kbs
        )
        self.metadata.touch_session(current["session_id"], kb_names=route.kbs)

        # 8) 存档：历史记录（含展示文本与引用详情）+ 复用缓存
        history[-1]["content"] = display[-1]["content"]
        answer_text = history[-1]["content"]
        summary_text = " ｜ ".join(summary)
        self.metadata.log_query(
            route.kbs,
            question,
            answer_text,
            mode,
            session_id=current["session_id"],
            references=references,
        )
        self.metadata.save_cached_answer(
            fingerprint,
            question,
            route.kbs,
            mode,
            {
                "answer": answer_text,
                "references_md": references_md,
                "downloads": downloads,
                "summary": summary_text,
            },
        )

        yield emit(history, references_md, downloads, summary_text)

    def finish_query(self, history: Optional[List[Dict[str, str]]]) -> Tuple[Any, ...]:
        """提问事件收尾：恢复输入区（挂在提问事件之后的 ``then`` 上）。

        正常结束与被「停止」取消都会走到这里——被取消时收尾输出没有机会发回前端，
        输入区会一直锁在"回答中"，所以必须由这个独立事件兜底解锁。

        若本轮在检索阶段就被停止（对话区只剩"正在检索…"占位），顺手换成停止提示。

        与 :meth:`handle_query` 同理：它同样不能把异常抛给 Gradio，
        否则错误标志会盖到本次输出的每个组件上。兜底只做最要紧的一件事——解锁输入区。
        """
        try:
            history = [dict(item) for item in (history or [])]
            stopped = False
            if history and (history[-1].get("role") or "") == "assistant":
                # content 可能是字符串，也可能是 Gradio 的分块结构，先统一取成文本再判断
                text = _message_text(history[-1].get("content")).strip()
                if not text or text == RETRIEVING_HINT:
                    history[-1]["content"] = STOPPED_HINT
                    stopped = True
            return _pack_query(
                {
                    "chatbot": history if stopped else _update(),
                    "question_box": _update(value="", interactive=True),
                    "send_btn": _update(visible=True),
                    "stop_btn": _update(visible=False),
                }
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return _pack_query(
                {
                    "question_box": _update(value="", interactive=True),
                    "send_btn": _update(visible=True),
                    "stop_btn": _update(visible=False),
                }
            )

    # ==================================================================
    def shutdown(self) -> None:
        try:
            self.metadata.close()
        except Exception:
            pass
