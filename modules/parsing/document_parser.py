"""文档解析器（纯代码）。

职责：解析 PDF / Word 文档，提取文字、表格、位置信息，输出结构化文本块。
技术选型：PyMuPDF（PDF）、python-docx（Word）

输出契约（每个文本块）::

    {
        "content": "文本内容",
        "type": "text | table | figure",
        "page": 3,                        # 1 起始；Word 无页码概念时为 None
        "position": {"bbox": [...] | None, "paragraph_index": 5},
        "context": "图表周围的文字描述（仅图表类型）"
    }

PDF 的 ``position`` 还会带上 ``size``（字号）与 ``bold``，用于识别章节标题。

分块策略（v2，修复「列表被切碎、标题与内容分离」的问题）
--------------------------------------------------------
v1 是「一个文本块 = 一个 chunk」，实测在数据手册这类文档上会出现三个问题：
    1. 章节标题自己成为一个只有十几个字的 chunk，检索时因为它与提问最"形似"
       而霸占 top-k 名额，却提供不了任何信息；
    2. 项目符号列表被切成一串 200 字左右的碎片，每片都丢失了所属章节；
    3. 列表碎片因为不含"MIO 外设"这类章节级关键词，排名被其它零星提及挤掉。
v2 改为**章节感知聚合**：
    先按字号/样式识别标题并维护标题层级（面包屑），再把同一章节内的相邻内容块
    合并到 ``CHUNK_SIZE``，并把面包屑作为前缀写进 chunk 正文。
    这样每个 chunk 自带"我在哪一节"，列表也保持完整。

不负责：语义理解、向量化、脱敏。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

# 按句末标点切分，保留标点
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")
_DOCX_HEADING_RE = re.compile(r"(?:heading|标题)\s*([1-9])", re.IGNORECASE)
# 图/表题注：虽然常常加粗放大会被字号规则误判为标题，但它不是章节，不能进面包屑
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table|chart|图|表)\s*[\d一二三四五六七八九十]", re.IGNORECASE)
# 正文段落长度阈值：用于统计正文字号（太短的块不足以代表正文）
_BODY_SAMPLE_MIN_CHARS = 40
# 面包屑最多保留层级，避免前缀喧宾夺主
_MAX_BREADCRUMB = 3
# 页眉页脚判定的边距比例（页面上下各 7%）
_MARGIN_RATIO = 0.07
# 页眉页脚至少重复出现在这么多页上才判定为噪声
_REPEAT_PAGE_MIN = 3


def _normalize_repeated(text: str) -> str:
    """归一化用于识别重复页眉页脚：数字统一替换，避免页码不同导致判不出重复。"""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", (text or "").strip().lower()))


class UnsupportedDocumentError(Exception):
    """不支持的文件类型。"""


class DocumentParser:
    """PDF / Word 解析器，输出结构化文本块，并按章节聚合生成可向量化的 chunk。"""

    #: 字号超过正文字号该比例即视为标题
    heading_size_ratio = 1.12
    #: 超过该长度的块不可能是标题
    heading_max_chars = 90
    #: 是否过滤重复出现的页眉页脚
    drop_repeated_margins = True

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def parse(self, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf(path)
        if suffix == ".docx":
            return self.parse_docx(path)
        raise UnsupportedDocumentError(f"暂不支持的文件类型：{suffix or '未知'}（仅支持 .pdf / .docx）")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    def parse_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """按 ``config.PDF_ENGINE`` 选择解析引擎。"""
        if config.PDF_ENGINE == "pymupdf4llm":
            return self._parse_pdf_markdown(path)
        return self._parse_pdf_blocks(path)

    def _parse_pdf_blocks(self, path: Path) -> List[Dict[str, Any]]:
        import pymupdf as fitz  # PyMuPDF（fitz 别名已废弃）

        blocks: List[Dict[str, Any]] = []
        paragraph_index = 0

        document = fitz.open(str(path))
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_no = page_index + 1

                # 1) 表格：先提取，避免同一区域被当成普通文本重复收录
                table_rects: List[Any] = []
                try:
                    finder = page.find_tables()
                    for table in getattr(finder, "tables", []) or []:
                        rows = table.extract() or []
                        markdown = self._rows_to_markdown(rows)
                        if not markdown.strip():
                            continue
                        blocks.append(
                            {
                                "content": markdown,
                                "type": "table",
                                "page": page_no,
                                "position": {
                                    "bbox": self._rect_to_list(table.bbox),
                                    "paragraph_index": paragraph_index,
                                },
                                "context": "",
                            }
                        )
                        paragraph_index += 1
                        table_rects.append(fitz.Rect(table.bbox))
                except Exception:
                    table_rects = []

                # 2) 文本块：走 dict 模式，带上字号与粗体标记用于识别标题
                page_dict = page.get_text("dict")
                raw_blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
                for raw in raw_blocks:
                    if raw.get("type") != 0:
                        continue
                    content, size, bold = self._block_text_and_style(raw)
                    if not content:
                        continue
                    rect = fitz.Rect(raw.get("bbox") or (0, 0, 0, 0))
                    if any(rect.intersects(table_rect) for table_rect in table_rects):
                        continue
                    blocks.append(
                        {
                            "content": content,
                            "type": "text",
                            "page": page_no,
                            "position": {
                                "bbox": self._rect_to_list(rect),
                                "paragraph_index": paragraph_index,
                                "size": round(float(size), 1),
                                "bold": bool(bold),
                            },
                            "context": "",
                            # 临时标记：位于页面上/下边距，用于后续识别重复页眉页脚
                            "_margin": bool(
                                rect.height
                                and page.rect.height
                                and (
                                    rect.y0 < page.rect.height * _MARGIN_RATIO
                                    or rect.y1 > page.rect.height * (1 - _MARGIN_RATIO)
                                )
                            ),
                        }
                    )
                    paragraph_index += 1

                # 3) 图片 / 图表位置（只标注"这里有图表"，不识别图片内容）
                try:
                    for image in page.get_images(full=True):
                        xref = image[0]
                        try:
                            rects = page.get_image_rects(xref)
                        except Exception:
                            rects = []
                        for rect in rects or []:
                            blocks.append(
                                {
                                    "content": "",
                                    "type": "figure",
                                    "page": page_no,
                                    "position": {
                                        "bbox": self._rect_to_list(rect),
                                        "paragraph_index": paragraph_index,
                                        "xref": int(xref),
                                    },
                                    "context": "",
                                }
                            )
                            paragraph_index += 1
                except Exception:
                    pass
        finally:
            document.close()

        return self._drop_repeated_margins(blocks)

    def _drop_repeated_margins(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """剔除重复出现的页眉页脚。

        这类文本位于页面上下边距且每页几乎相同（只有页码不同），留在正文里
        既白占一个 chunk 又会干扰检索，需要先做数字归一化再判重复。
        """
        if not self.drop_repeated_margins:
            for block in blocks:
                block.pop("_margin", None)
            return blocks

        counts: Dict[str, int] = {}
        for block in blocks:
            if block.pop("_margin", False):
                key = _normalize_repeated(block.get("content", ""))
                if key:
                    counts[key] = counts.get(key, 0) + 1

        noisy = {key for key, count in counts.items() if count >= _REPEAT_PAGE_MIN}
        if not noisy:
            return blocks

        kept: List[Dict[str, Any]] = []
        for block in blocks:
            if _normalize_repeated(block.get("content", "")) in noisy:
                continue
            kept.append(block)
        return kept

    # ------------------------------------------------------------------
    # PDF 引擎二：pymupdf4llm（版式模型 + Markdown 输出）
    # ------------------------------------------------------------------
    def _parse_pdf_markdown(self, path: Path) -> List[Dict[str, Any]]:
        """用版式模型解析，输出自带 ``#`` 标题层级、``-`` 列表与 Markdown 表格。

        相比原生块方案的优势：标题层级由版式模型给出，而不是靠字号猜；
        列表与表格结构在 Markdown 里天然完整。图表位置仍用 PyMuPDF 几何信息补充。
        """
        import pymupdf4llm

        entries = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        figures_by_page = self._figure_blocks_by_page(path)
        # 页眉页脚：跨页重复的整行。pymupdf4llm 常把页脚直接粘在正文段落后，
        # 所以按"行"而不是按"块"来剔除。
        noisy_lines = self._repeated_lines([entry.get("text") or "" for entry in entries])

        blocks: List[Dict[str, Any]] = []
        paragraph_index = 0

        for entry in entries:
            page_no = int((entry.get("metadata") or {}).get("page_number") or 0) or None
            text = entry.get("text") or ""
            if noisy_lines:
                text = "\n".join(
                    line
                    for line in text.splitlines()
                    if _normalize_repeated(line.strip()) not in noisy_lines
                )
            page_blocks = self._blocks_from_markdown(text, page_no)

            for block in page_blocks + list(figures_by_page.get(page_no, [])):
                block["position"]["paragraph_index"] = paragraph_index
                paragraph_index += 1
                blocks.append(block)

        return blocks

    def _blocks_from_markdown(
        self, markdown: str, page_no: Optional[int]
    ) -> List[Dict[str, Any]]:
        """把单页 Markdown 拆成 标题 / 表格 / 正文 三类块。"""
        blocks: List[Dict[str, Any]] = []
        buffer: List[str] = []
        buffer_kind = "text"

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            content = "\n".join(buffer).strip()
            buffer = []
            if content:
                blocks.append(self._markdown_block(content, buffer_kind, page_no))

        for line in (markdown or "").splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
                buffer_kind = "text"
                continue
            if stripped.startswith("|"):
                if buffer_kind != "table":
                    flush()
                    buffer_kind = "table"
                buffer.append(stripped)
                continue
            if stripped.startswith("#"):
                flush()
                blocks.append(self._markdown_block(stripped, "heading", page_no))
                buffer_kind = "text"
                continue
            # 列表行与正文行归为同类，保持列表整体性
            if buffer_kind in ("heading", "table"):
                flush()
                buffer_kind = "text"
            buffer.append(stripped)

        flush()
        return blocks

    @classmethod
    def _markdown_block(
        cls, content: str, kind: str, page_no: Optional[int]
    ) -> Dict[str, Any]:
        position: Dict[str, Any] = {"bbox": None, "paragraph_index": 0}
        if kind == "heading":
            level = len(content) - len(content.lstrip("#"))
            title = cls._clean_markdown(content)
            position["heading"] = True
            position["heading_level"] = max(1, min(level, 6))
            return {
                "content": title,
                "type": "text",
                "page": page_no,
                "position": position,
                "context": "",
            }
        return {
            "content": content,
            "type": "table" if kind == "table" else "text",
            "page": page_no,
            "position": position,
            "context": "",
        }

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """去掉 Markdown 标记，得到干净的章节标题。"""
        text = (text or "").strip().strip("#").strip()
        for marker in ("**", "__", "`", "*", "_"):
            text = text.replace(marker, "")
        return text.strip()

    @staticmethod
    def _figure_blocks_by_page(path: Path) -> Dict[int, List[Dict[str, Any]]]:
        """按页收集图片位置（Markdown 里没有坐标，仍需 PyMuPDF 几何信息）。"""
        import pymupdf as fitz

        result: Dict[int, List[Dict[str, Any]]] = {}
        document = fitz.open(str(path))
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                page_no = page_index + 1
                try:
                    images = page.get_images(full=True)
                except Exception:
                    images = []
                for image in images:
                    try:
                        rects = page.get_image_rects(image[0])
                    except Exception:
                        rects = []
                    for rect in rects or []:
                        result.setdefault(page_no, []).append(
                            {
                                "content": "",
                                "type": "figure",
                                "page": page_no,
                                "position": {
                                    "bbox": DocumentParser._rect_to_list(rect),
                                    "paragraph_index": 0,
                                    "xref": int(image[0]),
                                },
                                "context": "",
                            }
                        )
        finally:
            document.close()
        return result

    def _repeated_lines(self, page_texts: List[str]) -> set:
        """找出跨页重复出现的行（页眉页脚）。

        只考察每页开头 2 行与结尾 4 行，避免把正文里恰好重复的句子误删。
        数字先归一化，这样"第 8 页""第 9 页"能判成同一行。
        """
        if not self.drop_repeated_margins:
            return set()

        pages_by_key: Dict[str, set] = {}
        for page_index, text in enumerate(page_texts):
            lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
            if not lines:
                continue
            for line in lines[:2] + lines[-4:]:
                key = _normalize_repeated(line)
                if key:
                    pages_by_key.setdefault(key, set()).add(page_index)

        return {key for key, pages in pages_by_key.items() if len(pages) >= _REPEAT_PAGE_MIN}

    @staticmethod
    def _block_text_and_style(block: Dict[str, Any]) -> Tuple[str, float, bool]:
        """从 PyMuPDF 的 dict 块里取出纯文本、最大字号、是否粗体。"""
        parts: List[str] = []
        sizes: List[float] = []
        bold = False

        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(str(span.get("text", "")) for span in spans)
            if text.strip():
                parts.append(text)
            for span in spans:
                try:
                    sizes.append(float(span.get("size") or 0.0))
                except (TypeError, ValueError):
                    pass
                font = str(span.get("font") or "").lower()
                try:
                    flags = int(span.get("flags") or 0)
                except (TypeError, ValueError):
                    flags = 0
                # flags 第 4 位（16）在 PyMuPDF 中表示粗体
                if "bold" in font or "black" in font or flags & 16:
                    bold = True

        text = "\n".join(part for part in parts if part.strip()).strip()
        return text, (max(sizes) if sizes else 0.0), bold

    # ------------------------------------------------------------------
    # Word
    # ------------------------------------------------------------------
    def parse_docx(self, path: Path) -> List[Dict[str, Any]]:
        from docx import Document as DocxDocument
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = DocxDocument(str(path))
        blocks: List[Dict[str, Any]] = []
        paragraph_index = 0

        for item in self._iter_block_items(document, Paragraph, Table, qn):
            if isinstance(item, Paragraph):
                text = (item.text or "").strip()
                has_image = bool(item._p.findall(".//" + qn("w:drawing"))) or bool(
                    item._p.findall(".//" + qn("w:pict"))
                )
                style_name = ""
                try:
                    style = item.style
                    style_name = (style.name if style is not None else "") or ""
                except Exception:
                    style_name = ""

                if text:
                    blocks.append(
                        {
                            "content": text,
                            "type": "text",
                            "page": None,
                            "position": {
                                "bbox": None,
                                "paragraph_index": paragraph_index,
                                "style": style_name,
                            },
                            "context": "",
                        }
                    )
                    paragraph_index += 1

                if has_image:
                    blocks.append(
                        {
                            "content": "",
                            "type": "figure",
                            "page": None,
                            "position": {"bbox": None, "paragraph_index": paragraph_index},
                            # Word 无坐标，用所在段落文字作为上下文
                            "context": text,
                        }
                    )
                    paragraph_index += 1
                continue

            # 表格
            try:
                rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            except Exception:
                rows = []
            markdown = self._rows_to_markdown(rows)
            if markdown.strip():
                blocks.append(
                    {
                        "content": markdown,
                        "type": "table",
                        "page": None,
                        "position": {"bbox": None, "paragraph_index": paragraph_index},
                        "context": "",
                    }
                )
                paragraph_index += 1

        return blocks

    @staticmethod
    def _iter_block_items(document, paragraph_cls, table_cls, qn):
        """按文档流顺序迭代段落与表格。"""
        body = document.element.body
        for child in body.iterchildren():
            if child.tag == qn("w:p"):
                yield paragraph_cls(child, document)
            elif child.tag == qn("w:tbl"):
                yield table_cls(child, document)

    # ==================================================================
    # 分块：章节感知聚合
    # ==================================================================
    def build_chunks(
        self,
        blocks: List[Dict[str, Any]],
        doc_id: str,
        source_file: str,
        kb_name: str,
    ) -> List[Dict[str, Any]]:
        """把解析出的块聚合为带章节上下文的 chunk。

        chunk 契约在 v1 基础上增加：
            ``section``     章节面包屑文本（如 "MIO Peripherals"）
            ``section_id``  章节稳定标识，供检索时做"整节展开"
            ``order``       文档内顺序号
        """
        if not blocks:
            return []

        body_size = self._estimate_body_size(blocks)
        levels = self._heading_levels(blocks, body_size)

        chunks: List[Dict[str, Any]] = []
        breadcrumb: List[Tuple[int, str]] = []      # [(level, title)]
        section_index = 0
        current_section_id = ""

        buffer: List[str] = []
        buffer_len = 0
        buffer_type = "text"
        buffer_page: Optional[int] = None
        buffer_position: Dict[str, Any] = {}
        buffer_context = ""

        def section_text() -> str:
            return " > ".join(title for _level, title in breadcrumb)

        def flush() -> None:
            """把当前缓冲区落成一个或多个 chunk，并为每片都带上章节前缀。"""
            nonlocal buffer, buffer_len
            if not buffer:
                return
            body = "\n".join(buffer).strip()
            buffer = []
            buffer_len = 0
            if len(body) < config.MIN_CHUNK_CHARS:
                return

            heading = section_text()
            for piece in self._split_text(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                # 前缀写给每一片，避免长内容的后续片段丢失章节上下文
                content = f"章节：{heading}\n{piece}" if heading else piece
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}-{len(chunks):04d}",
                        "doc_id": doc_id,
                        "kb_name": kb_name,
                        "source_file": source_file,
                        "content": content,
                        "type": buffer_type,
                        "page": buffer_page,
                        "position": buffer_position,
                        "context": buffer_context,
                        "section": heading,
                        "section_id": current_section_id,
                        "order": len(chunks),
                    }
                )

        for block in blocks:
            content = (block.get("content") or "").strip()
            block_type = block.get("type", "text")

            # 标题：只更新面包屑，自身不产生 chunk（它会成为后续内容的章节前缀）
            if block_type == "text" and self._is_heading(block, body_size, levels):
                flush()
                level = self._heading_level(block, levels)
                title = content.splitlines()[0].strip()
                while breadcrumb and breadcrumb[-1][0] >= level:
                    breadcrumb.pop()
                breadcrumb.append((level, title))
                del breadcrumb[:-_MAX_BREADCRUMB]
                section_index += 1
                current_section_id = f"{doc_id}#s{section_index:03d}"
                continue

            # 图表块：无可读文字时用上下文构造可检索内容
            if block_type == "figure":
                flush()
                context = (block.get("context") or "").strip()
                if not context:
                    continue
                content = f"（图表内容）{context}"
            elif block_type == "table":
                # 表格保持原子性，不与正文混在一起
                flush()

            # 换页或换类型都断开，避免 chunk 的页码/类型元数据含混
            if buffer and (block.get("page") != buffer_page or buffer_type != block_type):
                flush()
            if buffer and buffer_len + len(content) > config.CHUNK_SIZE:
                flush()

            if not buffer:
                buffer_type = block_type
                buffer_page = block.get("page")
                buffer_position = block.get("position") or {}
                buffer_context = block.get("context") or ""

            buffer.append(content)
            buffer_len += len(content)

            if buffer_len >= config.CHUNK_SIZE:
                flush()

        flush()
        return chunks

    # ------------------------------------------------------------------
    # 标题识别辅助
    # ------------------------------------------------------------------
    def _estimate_body_size(self, blocks: List[Dict[str, Any]]) -> float:
        """用「按字数加权」的众数估计正文字号。"""
        counter: Dict[float, int] = {}
        for block in blocks:
            if block.get("type") != "text":
                continue
            content = (block.get("content") or "").strip()
            size = (block.get("position") or {}).get("size")
            if not size or len(content) < _BODY_SAMPLE_MIN_CHARS:
                continue
            counter[round(float(size), 1)] = counter.get(round(float(size), 1), 0) + len(content)
        if not counter:
            return 0.0
        return max(counter.items(), key=lambda item: item[1])[0]

    def _is_heading(
        self,
        block: Dict[str, Any],
        body_size: float,
        levels: Dict[float, int],
    ) -> bool:
        text = (block.get("content") or "").strip()
        if not text or len(text) > self.heading_max_chars:
            return False

        # Markdown 引擎已把标题层级写在块里，直接采信（图注/表注仍排除）
        if (block.get("position") or {}).get("heading"):
            return not bool(_CAPTION_RE.match(text))

        # 图注/表注不是章节：让它作为普通内容跟随当前章节，而不是成为新的父节点
        if _CAPTION_RE.match(text):
            return False
        position = block.get("position") or {}

        # Word：靠样式名判断
        style = str(position.get("style") or "")
        if style and _DOCX_HEADING_RE.search(style):
            return True

        # PDF：靠字号 + 粗体判断
        try:
            size = float(position.get("size") or 0.0)
        except (TypeError, ValueError):
            size = 0.0
        if not size or not body_size:
            return False
        if size >= body_size * self.heading_size_ratio and size in levels:
            return True
        # 字号与正文相同但整块加粗、且很短，也按小标题处理
        return bool(position.get("bold")) and len(text) <= 40 and size >= body_size

    def _heading_levels(
        self, blocks: List[Dict[str, Any]], body_size: float
    ) -> Dict[float, int]:
        """把出现的标题字号从大到小映射为 1、2、3… 级。"""
        sizes = set()
        for block in blocks:
            if block.get("type") != "text":
                continue
            text = (block.get("content") or "").strip()
            if not text or len(text) > self.heading_max_chars:
                continue
            style = str((block.get("position") or {}).get("style") or "")
            if style and _DOCX_HEADING_RE.search(style):
                continue
            try:
                size = float((block.get("position") or {}).get("size") or 0.0)
            except (TypeError, ValueError):
                continue
            if body_size and size >= body_size * self.heading_size_ratio:
                sizes.add(round(size, 1))
        return {size: index + 1 for index, size in enumerate(sorted(sizes, reverse=True))}

    def _heading_level(self, block: Dict[str, Any], levels: Dict[float, int]) -> int:
        # Markdown 引擎：直接用 "#" 的个数
        level = (block.get("position") or {}).get("heading_level")
        if level:
            try:
                return max(1, min(int(level), 6))
            except (TypeError, ValueError):
                pass
        style = str((block.get("position") or {}).get("style") or "")
        match = _DOCX_HEADING_RE.search(style)
        if match:
            return int(match.group(1))
        try:
            size = round(float((block.get("position") or {}).get("size") or 0.0), 1)
        except (TypeError, ValueError):
            return 1
        return levels.get(size, 1)

    # ------------------------------------------------------------------
    # 文本切分
    # ------------------------------------------------------------------
    @staticmethod
    def _split_text(text: str, size: int, overlap: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        size = max(100, int(size))
        overlap = max(0, min(int(overlap), size // 2))
        if len(text) <= size:
            return [text]

        sentences = [s for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
        pieces: List[str] = []
        buffer = ""

        for sentence in sentences:
            if len(sentence) > size:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                step = max(1, size - overlap)
                for start in range(0, len(sentence), step):
                    pieces.append(sentence[start:start + size])
                continue

            if len(buffer) + len(sentence) <= size:
                buffer += sentence
            else:
                pieces.append(buffer)
                tail = buffer[-overlap:] if overlap else ""
                buffer = tail + sentence

        if buffer.strip():
            pieces.append(buffer)

        return [p.strip() for p in pieces if p.strip()]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _rect_to_list(rect) -> Optional[List[float]]:
        if rect is None:
            return None
        try:
            return [round(float(v), 2) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]
        except Exception:
            try:
                return [round(float(v), 2) for v in tuple(rect)[:4]]
            except Exception:
                return None

    @staticmethod
    def _rows_to_markdown(rows: List[List[Any]]) -> str:
        """把二维表格转成 Markdown 表格文本。"""
        cleaned: List[List[str]] = []
        width = 0
        for row in rows or []:
            cells = [("" if cell is None else str(cell)).replace("\n", " ").strip() for cell in row]
            if not any(cells):
                continue
            cleaned.append(cells)
            width = max(width, len(cells))
        if not cleaned:
            return ""

        for row in cleaned:
            while len(row) < width:
                row.append("")

        header = cleaned[0]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
