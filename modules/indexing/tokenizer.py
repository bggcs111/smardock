"""检索分词器（纯代码）。

用途：为 BM25 关键词检索生成词元。
策略：
    * 英文/数字按连续字母数字切词（如 "MIO12" -> "mio12"）；
    * 中文按**字二元组**（bigram，如 "外设" -> "外设"，"可以用于" -> 可以/以用/用于）。

中文不引入分词模型（jieba 之类）的原因：目标是关键词匹配而非语义，
二元组在没有额外依赖的前提下覆盖率足够，且对"MIO 外设"这类跨语言混合查询稳定。
"""
from __future__ import annotations

import re

# 连续的 ASCII 字母/数字
_ASCII_RE = re.compile(r"[a-zA-Z0-9]+")
# CJK 统一表意文字（含扩展 A）
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> list:
    """把文本切成关键词词元列表（用于建索引与查询，两端必须用同一套规则）。"""
    if not text:
        return []

    lowered = str(text).lower()
    tokens = list(_ASCII_RE.findall(lowered))

    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))

    return tokens


def to_index_text(text: str) -> str:
    """把文本转成可写入 FTS 索引的空格分隔词元串。"""
    return " ".join(tokenize(text))
