"""引用定位器（纯代码）。

职责：根据检索结果中的 chunk_id 和元数据，生成可追溯的引用信息。

输入：检索结果（含 chunk_id、页码、位置、来源文件）
输出：引用对象::

    {
        "text": "引用片段",
        "link": "file:///E:/.../doc.pdf#page=3",
        "source_kb": "知识库名称",
        "file_name": "doc.pdf",
        "page": 3,
        "local_path": "E:\\...\\doc.pdf"
    }

关于 ``link`` 的说明：浏览器出于安全限制，不允许从 http 页面跳转到 ``file:///``，
因此 UI 层不会把该链接当作可点击跳转，而是同时展示「文件名 + 页码 + 本地绝对路径」，
并可通过 :meth:`export` 把原件复制到导出目录供浏览器下载。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.storage.document_store import DocumentStore

# 强调与代码标记：片段里的这些符号会污染外层 Markdown 结构
_EMPHASIS_RE = re.compile(r"\*{1,3}|_{2,}")
_BACKTICK_RE = re.compile(r"`+")
# 只断开「](」这个链接形态，方括号本身必须保留：
# 片段里的 [姓名1] 等占位符要靠它被还原器识别。
_LINK_RE = re.compile(r"\]\(")


class ReferenceLocator:
    """把检索命中转换成人类可读、可追溯的引用。"""

    def __init__(self, document_store: Optional[DocumentStore] = None) -> None:
        self.documents = document_store or DocumentStore()

    # ------------------------------------------------------------------
    def locate(self, hit: Dict[str, Any], source_kb: str = "") -> Dict[str, Any]:
        metadata = dict(hit.get("metadata") or {})
        kb_name = source_kb or hit.get("source_kb") or metadata.get("kb_name") or ""
        file_path = str(metadata.get("file_path") or "")
        file_name = str(metadata.get("source_file") or (Path(file_path).name if file_path else "未知来源"))
        page = metadata.get("page")
        try:
            page = int(page) if page not in (None, "", "None") else None
        except (TypeError, ValueError):
            page = None

        local_path = self._resolve_local_path(kb_name, file_name, file_path)
        return {
            "text": self._snippet(hit.get("content") or ""),
            "link": self._build_uri(local_path, page),
            "source_kb": kb_name,
            "file_name": file_name,
            "page": page,
            "local_path": local_path or "",
            "chunk_id": metadata.get("chunk_id") or "",
            "type": metadata.get("type") or "text",
            "similarity": hit.get("similarity"),
        }

    def locate_many(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.locate(hit) for hit in hits]

    # ------------------------------------------------------------------
    @staticmethod
    def masked_name_map(references: List[Dict[str, Any]]) -> Dict[str, str]:
        """给每个原件按首次出现顺序分配打码名（文档1.pdf）。

        引用标注、本地路径、下载副本共用这份映射，保证三处一致：
        文件名常含人名（如「健康保险合同-石井然.pdf」），隐私模式下不应出现在界面上。
        """
        mapping: Dict[str, str] = {}
        for reference in references:
            key = reference.get("local_path") or reference.get("file_name") or ""
            if not key or key in mapping:
                continue
            suffix = Path(str(reference.get("file_name") or "")).suffix
            mapping[key] = f"文档{len(mapping) + 1}{suffix}"
        return mapping

    @staticmethod
    def display_name(reference: Dict[str, Any], masked: Optional[Dict[str, str]] = None) -> str:
        """展示用的文件名：给了打码映射就用打码名。"""
        name = reference.get("file_name") or "未知文件"
        if not masked:
            return name
        return masked.get(reference.get("local_path") or "", "") or masked.get(name, name)

    @classmethod
    def source_label(
        cls, reference: Dict[str, Any], masked: Optional[Dict[str, str]] = None
    ) -> str:
        """「文档1.pdf · 第 3 页」这类来源标注（引用上标的 hover 提示）。"""
        name = cls.display_name(reference, masked)
        page = reference.get("page")
        return name + (f" · 第 {page} 页" if page else "")

    # ------------------------------------------------------------------
    def export(
        self,
        references: List[Dict[str, Any]],
        masked: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """把引用涉及的原件复制到导出目录，返回可下载的绝对路径列表。"""
        exported: List[str] = []
        seen: set[str] = set()
        for reference in references:
            local_path = reference.get("local_path") or ""
            if not local_path or local_path in seen:
                continue
            seen.add(local_path)
            name = (masked or {}).get(local_path, "")
            target = self.documents.export_copy(local_path, name=name)
            if target:
                exported.append(str(target))
        return exported

    # ------------------------------------------------------------------
    def render_markdown(
        self,
        references: List[Dict[str, Any]],
        masked: Optional[Dict[str, str]] = None,
    ) -> str:
        if not references:
            return "_暂无引用来源_"

        lines: List[str] = []
        for index, reference in enumerate(references, start=1):
            page = reference.get("page")
            location = f"第 {page} 页" if page else "（无页码信息）"
            lines.append(
                f"**{index}. {self.display_name(reference, masked)}** · {location} · "
                f"知识库：{reference.get('source_kb') or '未标注'}"
            )
            snippet = self._plain_text(reference.get("text") or "")
            if snippet:
                # 再规范化一次：片段可能已被还原器填回真实信息，同样不能污染 Markdown 结构
                lines.append(f"> {snippet}")
            local_path = reference.get("local_path") or ""
            if local_path:
                names = masked or {}
                if local_path in names:
                    # 路径里的文件名同样打码，保留目录层级以便定位
                    original = reference.get("file_name") or ""
                    if original and original in local_path:
                        local_path = local_path.replace(original, names[local_path])
                lines.append(f"`{local_path}`")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _plain_text(content: str) -> str:
        """把片段转成"渲染后与原文一致"的安全纯文本。

        片段取自文档解析结果，本身可能带 Markdown 标记（``**粗体**``、`` `代码` ``）
        与尖括号内容。若直接塞进 gr.Markdown，会出现两类结构性问题：

        1. 片段被截断时可能把一个 ``**`` 从中间切断，标记不闭合后，
           **后面所有引用条目都会被吞进同一段粗体**（表现为"只显示第 1 条"）；
        2. 未配对的尖括号会被当作 HTML 标签。

        因此统一做：折叠空白 → 去掉强调/代码标记 → 断开链接形态 → 转义尖括号。

        注意：**不要转义方括号**。片段里的 ``[姓名1]`` 是脱敏占位符，
        转义后还原器就匹配不到了（表现为引用区显示 ``\\[姓名1]`` 而不是真实信息）。
        """
        text = " ".join((content or "").split())
        if not text:
            return ""
        text = _EMPHASIS_RE.sub("", text)     # **粗体** / _斜体_ / *斜体*
        text = _BACKTICK_RE.sub("", text)     # `代码`
        text = _LINK_RE.sub(r"]\\(", text)    # [x](y) 不当链接
        return text.replace("<", "&lt;").replace(">", "&gt;")

    @classmethod
    def _snippet(cls, content: str, limit: int = 300) -> str:
        """先规范化再截断，避免截断把标记切成不闭合状态。"""
        text = cls._plain_text(content)
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"

    @staticmethod
    def _build_uri(path: str, page: Optional[int]) -> str:
        if not path:
            return ""
        try:
            uri = Path(path).as_uri()
        except Exception:
            uri = "file:///" + path.replace("\\", "/")
        if page:
            uri = f"{uri}#page={page}"
        return uri

    def _resolve_local_path(self, kb_name: str, file_name: str, stored_path: str) -> str:
        """优先使用元数据里记录的路径，失效时回退到知识库目录推断。"""
        if stored_path and Path(stored_path).exists():
            return str(Path(stored_path))
        if stored_path:
            return stored_path
        if kb_name and file_name:
            candidate = self.documents.base_dir / kb_name / file_name
            if candidate.exists():
                return str(candidate)
        return ""

    @staticmethod
    def dump_position(position: Any) -> str:
        if isinstance(position, str):
            return position
        try:
            return json.dumps(position or {}, ensure_ascii=False)
        except Exception:
            return "{}"
