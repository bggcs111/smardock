"""占位符还原器（纯代码 · 全程本地）。

职责：隐私模式下，把大模型回答中的占位符还原为真实信息。
输入：大模型生成的回答（含 [姓名1]、[电话1] 等占位符）+ 脱敏映射表
输出：还原后的回答

说明：全程本地字符串替换，不调用云端 API。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# 占位符形如 [姓名1]、[电话1]：方括号内是「实体类型 + 编号」。
# 该模式刻意收紧，用于把引用标记 [1]、Markdown 链接 [文字](url) 等排除在外。
_PLACEHOLDER_RE = re.compile(r"\[[^\W\d][\w·]{0,19}\d{1,4}\]")


class PlaceholderRestorer:
    """占位符 -> 原文 的本地还原。"""

    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self._mapping: Dict[str, str] = {}
        if mapping:
            self.update_mapping(mapping)

    # ------------------------------------------------------------------
    def update_mapping(self, mapping: Dict[str, str]) -> None:
        self._mapping.update(mapping or {})

    def set_mapping(self, mapping: Dict[str, str]) -> None:
        self._mapping = dict(mapping or {})

    @property
    def mapping(self) -> Dict[str, str]:
        return dict(self._mapping)

    # ------------------------------------------------------------------
    def has_placeholder(self, text: str) -> bool:
        """文本中是否出现形如 [姓名1] 的占位符。"""
        return bool(_PLACEHOLDER_RE.search(text or ""))

    def contains_known(self, text: str, mapping: Optional[Dict[str, str]] = None) -> bool:
        """文本中是否出现映射表里已登记的占位符（决定是否需要还原）。"""
        table = self._mapping if mapping is None else (mapping or {})
        content = text or ""
        return any(placeholder and placeholder in content for placeholder in table)

    def restore(self, text: str, mapping: Optional[Dict[str, str]] = None) -> str:
        """把文本中的占位符替换回原文；无映射时原样返回。"""
        if not text:
            return text
        table = self._mapping if mapping is None else dict(mapping)
        if not table:
            return text

        # 长占位符优先，避免 [姓名1] 误匹配 [姓名11] 的前缀
        for placeholder in sorted(table.keys(), key=len, reverse=True):
            if placeholder and placeholder in text:
                text = text.replace(placeholder, str(table[placeholder]))
        return text

    def restore_references(
        self,
        references: List[Dict[str, Any]],
        mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """引用片段、文件名中的占位符同样需要还原。"""
        table = self._mapping if mapping is None else dict(mapping)
        if not table:
            return references

        restored: List[Dict[str, Any]] = []
        for reference in references:
            item = dict(reference)
            for field in ("text", "file_name", "source_kb", "link", "local_path"):
                if isinstance(item.get(field), str):
                    item[field] = self.restore(item[field], table)
            restored.append(item)
        return restored

    @staticmethod
    def find_unresolved(text: str, mapping: Dict[str, str]) -> List[str]:
        """返回文本中残留、未在映射表里登记的占位符。"""
        table = mapping or {}
        return [token for token in _PLACEHOLDER_RE.findall(text or "") if token not in table]
