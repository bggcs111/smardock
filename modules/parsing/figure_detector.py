"""图表识别器（纯代码为主 + AI 辅助）。

职责：
    - 纯代码部分：版面分析，识别"这里有图表/表格/图片"，并用邻近文字补齐上下文
    - AI 辅助部分：调用云端 LLM 从上下文文字中提取图表标题与语义描述（默认关闭）

输出：图表块的位置信息 + 标题 + 上下文文字。
说明：不识别图片内容本身。
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

import config

# 图/表题注的常见写法
_CAPTION_RE = re.compile(r"^\s*(图|表|图表|Figure|Fig\.?|Table)\s*[\d一二三四五六七八九十]+")


class FigureDetector:
    """版面分析 + 上下文补齐 + 可选的 AI 语义描述。"""

    def __init__(
        self,
        describe_fn: Optional[Callable[[str], str]] = None,
        max_context_blocks: int = 3,
    ) -> None:
        # describe_fn: 由外部注入的 LLM 调用（签名 str -> str），用于图表语义描述
        self.describe_fn = describe_fn
        self.max_context_blocks = max(1, int(max_context_blocks))

    # ------------------------------------------------------------------
    # 纯代码：版面分析
    # ------------------------------------------------------------------
    def detect(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为 figure 块补齐上下文文字与题注；就地返回同一列表。"""
        if not blocks:
            return blocks

        for index, block in enumerate(blocks):
            if block.get("type") != "figure":
                continue

            if not block.get("context"):
                block["context"] = self._collect_context(blocks, index)

            block["caption"] = self._find_caption(blocks, index)

        return blocks

    def _collect_context(self, blocks: List[Dict[str, Any]], figure_index: int) -> str:
        figure = blocks[figure_index]
        page = figure.get("page")
        bbox = (figure.get("position") or {}).get("bbox")

        candidates: List[tuple] = []
        for index, block in enumerate(blocks):
            if index == figure_index or block.get("type") != "text":
                continue
            if page is not None and block.get("page") != page:
                continue

            distance = self._distance(bbox, (block.get("position") or {}).get("bbox"))
            if distance is None:
                # Word 等无坐标场景：用段落顺序的接近程度近似
                distance = abs(index - figure_index) * 1000.0
            candidates.append((distance, index, block.get("content", "")))

        candidates.sort(key=lambda item: (item[0], item[1]))
        picked = [content.strip() for _, _, content in candidates[: self.max_context_blocks] if content]
        return " ".join(picked)

    @staticmethod
    def _distance(bbox_a, bbox_b) -> Optional[float]:
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return None
        ax0, ay0, ax1, ay1 = bbox_a[:4]
        bx0, by0, bx1, by1 = bbox_b[:4]
        dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
        dy = max(0.0, max(by0 - ay1, ay0 - by1))
        return (dx * dx + dy * dy) ** 0.5

    def _find_caption(self, blocks: List[Dict[str, Any]], figure_index: int) -> str:
        figure = blocks[figure_index]
        page = figure.get("page")
        bbox = (figure.get("position") or {}).get("bbox")

        candidates: List[tuple] = []
        for index, block in enumerate(blocks):
            if index == figure_index or block.get("type") != "text":
                continue
            content = (block.get("content") or "").strip()
            if not _CAPTION_RE.match(content):
                continue
            if page is not None and block.get("page") != page:
                continue
            distance = self._distance(bbox, (block.get("position") or {}).get("bbox"))
            candidates.append((distance if distance is not None else 0.0, index, content))

        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    # ------------------------------------------------------------------
    # AI 辅助：图表语义描述（可选）
    # ------------------------------------------------------------------
    def describe_figures(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为图表生成语义描述并写入 content，使其可被向量化检索。"""
        if not config.FIGURE_AI_DESCRIBE or self.describe_fn is None:
            return blocks

        for block in blocks:
            if block.get("type") != "figure" or block.get("content"):
                continue
            context = (block.get("context") or "").strip()
            if not context:
                continue
            try:
                description = self.describe_fn(
                    "下面是一份文档中某张图表附近的文字。请用一两句话概括这张图表可能表达的内容，"
                    "只输出描述本身：\n" + context[:1500]
                )
            except Exception:
                continue
            if description:
                block["content"] = f"（图表语义描述）{description.strip()}"
        return blocks
