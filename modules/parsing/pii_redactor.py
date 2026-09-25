"""PII 脱敏器（本地 NER 模型 + 纯代码正则补充）。

职责：
    - 本地 AI 部分：使用 ModelScope RaNER 模型识别人名 / 地名 / 机构名 / 地缘政治实体
    - 纯代码部分：正则补充识别电话、身份证、邮箱、银行卡；执行替换并维护脱敏映射表

触发条件：仅隐私模式下启用。
输出契约::

    {
        "redacted_content": "脱敏后文本",
        "redaction_map": {"[姓名1]": "张三", "[电话1]": "138xxxx"},
        "has_sensitive": true
    }

设计要点（对原方案的修正）：
    原方案中编号计数器是「每次 redact_text 调用重置」的，会导致
    「文档A 的 [姓名1]=张三、文档B 的 [姓名1]=李四」这类跨文档歧义，
    多知识库混合检索时还原结果会错乱。
    因此这里改为**占位符全局唯一**：编号从元数据库的 redaction_map 表统一分配，
    同一实体在全库范围内复用同一个占位符，跨文档 / 跨知识库还原都不会歧义。

依赖降级：未安装 modelscope 或模型加载失败时，自动降级为"仅正则脱敏"，
系统其余功能不受影响，并通过 :attr:`load_error` / :meth:`status` 对外暴露原因。
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import config
from modules.logutil import get_logger

# ---------------------------------------------------------------------------
# 实体类型映射：模型输出标签 -> 中文占位符前缀
# ---------------------------------------------------------------------------
ENTITY_TYPE_MAP: Dict[str, str] = {
    "PER": "姓名",
    "PERSON": "姓名",
    "NAME": "姓名",
    "LOC": "地址",
    "LOCATION": "地址",
    "GPE": "地区",
    "ORG": "机构",
    "ORGANIZATION": "机构",
}

# ---------------------------------------------------------------------------
# 正则规则：(占位符前缀, 模式, 优先级，越大越优先)
#
# 数值型 PII 的形态很固定，用正则最可靠。踩过的坑：
#   1. 身份证须在银行卡之前，否则 18 位身份证会被 16~19 位的银行卡规则抢先；
#   2. 号码常带分隔符（138-0000-0001 / 400-000-0000 / 6222 0212 ...），
#      只写连续数字会整段漏掉。
# ---------------------------------------------------------------------------
REGEX_RULES: List[Tuple[str, str, int]] = [
    # 身份证（18 位，末位可为 X）
    ("身份证", r"(?<![0-9A-Za-z])\d{17}[\dXx](?![0-9A-Za-z])", 100),
    # 统一社会信用代码（18 位，字符集受限；与身份证同长时身份证优先）
    ("信用代码", r"(?<![0-9A-Za-z])[0-9A-HJ-NPQRTUWXY]{18}(?![0-9A-Za-z])", 95),
    ("邮箱", r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?![A-Za-z])", 90),
    # 手机号：允许 +86 前缀，允许 - 或空格分组
    ("电话", r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d(?:[-\s]?\d{4}){2}(?!\d)", 86),
    # 服务热线：400 / 800
    ("电话", r"(?<!\d)(?:400|800)[-\s]?\d{3}[-\s]?\d{4}(?!\d)", 84),
    # 银行卡：连续 16~19 位，或 4 位一组带分隔符
    ("银行卡", r"(?<!\d)\d{16,19}(?!\d)", 82),
    ("银行卡", r"(?<!\d)\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{3,4}(?!\d)", 81),
    # 座机：区号（可带括号）+ 7~8 位号码
    ("座机", r"(?<!\d)\(?0\d{2,3}\)?[-\s]?\d{7,8}(?!\d)", 70),
]

# ---------------------------------------------------------------------------
# 标签锚定规则（纯代码，不依赖模型）
#
# 合同、保单、表单、简历这类文档里，中文 PII 几乎都跟在字段名后面
# （「投保人（甲方）：石井然」「联系地址：上海市…」）。用字段名定位比通用
# NER 更准，而且零依赖、可解释。没有字段名的自由文本（如会议纪要里的人名）
# 仍需要 NER 模型补充。
# ---------------------------------------------------------------------------
# 取值方式：冒号后取到「下一个字段名 + 冒号」或标点 / 换行 / 行尾为止，
# 这样同一行里「身份证号码：X 联系地址：Y 联系电话：Z」也能正确切开。
_VALUE_TAIL = r"(?P<v>.{1,60}?)(?=\s*[\u4e00-\u9fa5A-Za-z]{2,10}\s*[：:]|[，,。；;、|｜\n\r]|$)"
_FIELD_SEP = r"\s*(?:[（(][^）)]{0,12}[）)])?\s*[：:]\s*(?:指|为|系|是)?\s*"

_NAME_ANCHOR = r"(?:姓\s*名|投保人|被保险人|受益人|客户姓名|联系人|申请人|户主|业主|家长)"
_ADDRESS_ANCHOR = (
    r"(?:联系地址|通讯地址|通信地址|居住地址|家庭住址|户籍地址|送达地址|"
    r"住\s*址|地\s*址|住所地|工作地址|现居地址)"
)
_ORG_ANCHOR = (
    r"(?:保险人|投保单位|工作单位|单位名称|公司名称|开户行|开户银行|"
    r"就诊医院|就读学校|学校名称|发卡银行)"
)
_CODE_ANCHOR = (
    r"(?:合同编号|保单号|保单号码|保险单号|证件号码|证件号|身份证号|身份证号码|"
    r"统一社会信用代码|纳税人识别号|社保号|医保号|员工编号|工\s*号|学\s*号)"
)
_BIRTH_ANCHOR = r"(?:出生日期|出生年月日|出生年月|生\s*日)"

#: (占位符前缀, 完整模式, 校验器名, 优先级)
CONTEXT_RULES: List[Tuple[str, str, str, int]] = [
    ("姓名", _NAME_ANCHOR + _FIELD_SEP + _VALUE_TAIL, "person", 96),
    ("地址", _ADDRESS_ANCHOR + _FIELD_SEP + _VALUE_TAIL, "address", 96),
    ("机构", _ORG_ANCHOR + _FIELD_SEP + _VALUE_TAIL, "org", 96),
    ("编号", _CODE_ANCHOR + _FIELD_SEP + _VALUE_TAIL, "code", 96),
    ("日期", _BIRTH_ANCHOR + _FIELD_SEP + _VALUE_TAIL, "birthday", 96),
]

#: 表格形态：「|字段名|值|」——Markdown 表格里的字段没有冒号，需要单独一套模式
_TABLE_PREFIX = r"\|\s*"
_TABLE_MID = r"\s*\|\s*"
_TABLE_VALUE_TAIL = r"(?P<v>[^|\n\r]{1,60}?)\s*(?=\|)"

_ANCHOR_RULES: List[Tuple[str, str, str, int]] = [
    ("姓名", _NAME_ANCHOR, "person", 96),
    ("地址", _ADDRESS_ANCHOR, "address", 96),
    ("机构", _ORG_ANCHOR, "org", 96),
    ("编号", _CODE_ANCHOR, "code", 96),
    ("日期", _BIRTH_ANCHOR, "birthday", 96),
]


def build_context_rules(extra_fields: Optional[List[str]] = None) -> List[Tuple[str, str, str, int]]:
    """生成标签锚定规则：每个字段同时支持「字段名：值」与表格「|字段名|值|」两种形态。

    :param extra_fields: 额外要脱敏的字段名（如 ["性别", "职业类别"]），
                         用通用校验器，占位符前缀即字段名本身。
    """
    rules: List[Tuple[str, str, str, int]] = []
    for label, anchor, validator, priority in _ANCHOR_RULES:
        rules.append((label, anchor + _FIELD_SEP + _VALUE_TAIL, validator, priority))
        rules.append(
            (label, _TABLE_PREFIX + anchor + _TABLE_MID + _TABLE_VALUE_TAIL, validator, priority + 1)
        )

    for field in extra_fields or []:
        safe = re.escape(field.strip())
        if not safe:
            continue
        rules.append((field, safe + _FIELD_SEP + _VALUE_TAIL, "generic", 94))
        rules.append(
            (field, _TABLE_PREFIX + safe + _TABLE_MID + _TABLE_VALUE_TAIL, "generic", 95)
        )
    return rules

#: 机构特征词：用于把「机构」和「人名 / 地址」区分开
_ORG_WORDS = (
    "公司", "银行", "医院", "学校", "大学", "学院", "中心", "事务所", "委员会",
    "集团", "保险", "支行", "分行", "诊所", "药房", "研究院", "管理局", "合作社",
)
#: 地域特征词：地址必须命中其一，也用于排除把人名当地址
_REGION_WORDS = (
    "省", "市", "区", "县", "镇", "乡", "村", "路", "街", "道", "巷", "号", "栋",
    "幢", "楼", "室", "大厦", "花园", "小区", "广场", "公寓", "苑", "里", "弄",
)
#: 不是人名的常见占位词（法律角色、指代、空白值）
_NAME_STOP = {
    "甲方", "乙方", "丙方", "本人", "其本人", "同一人", "上述", "投保人", "被保险人",
    "受益人", "以下简称", "无", "不详", "未填写", "身份证", "户口簿",
    "法定继承人", "继承人", "法定代理人", "代理人", "监护人", "配偶", "子女",
    "父母", "法定受益人", "指定受益人", "全体继承人", "其他", "其他人员",
}
#: 以这些字结尾的几乎都是角色 / 指代，不是姓名
_NAME_ROLE_SUFFIX = ("人", "者", "方", "们", "等")

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")
_USELESS_RE = re.compile(r"^[\s\W_]+$")
_HAN_NAME_RE = re.compile(r"^[\u4e00-\u9fa5·]{2,6}$")
_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-*/_]{5,39}$")
_BIRTH_RE = re.compile(r"^\d{4}\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])\s*\d{1,2}\s*日?$")


def _is_person(value: str) -> bool:
    if value in _NAME_STOP or value.endswith(_NAME_ROLE_SUFFIX):
        return False
    if not _HAN_NAME_RE.match(value):
        return False
    # 含机构词或地域词的一定不是人名
    return not any(word in value for word in _ORG_WORDS + _REGION_WORDS)


def _is_address(value: str) -> bool:
    if not 4 <= len(value) <= 60:
        return False
    return any(word in value for word in _REGION_WORDS)


def _is_org(value: str) -> bool:
    if not 4 <= len(value) <= 40:
        return False
    return any(word in value for word in _ORG_WORDS)


def _is_code(value: str) -> bool:
    return bool(_CODE_RE.match(value))


def _is_birthday(value: str) -> bool:
    return bool(_BIRTH_RE.match(value))


def _is_generic(value: str) -> bool:
    """额外字段（性别、职业等）的宽松校验：非空、不过长。"""
    return 1 <= len(value) <= 40


_VALIDATORS = {
    "person": _is_person,
    "address": _is_address,
    "org": _is_org,
    "code": _is_code,
    "birthday": _is_birthday,
    "generic": _is_generic,
}

#: 编译后的标签锚定规则缓存（模式串 -> 编译结果）
_CONTEXT_COMPILED: Dict[str, Any] = {}


class PIIRedactor:
    """敏感信息识别与替换。"""

    def __init__(
        self,
        metadata_store=None,
        model_id: Optional[str] = None,
        enable_ner: Optional[bool] = None,
        enable_regex: Optional[bool] = None,
        ner_max_segment: int = 300,
    ) -> None:
        self.metadata = metadata_store
        self.model_id = model_id or config.PII_NER_MODEL_ID
        self.enable_ner = config.PII_NER_ENABLED if enable_ner is None else bool(enable_ner)
        self.enable_regex = config.PII_REGEX_ENABLED if enable_regex is None else bool(enable_regex)
        self.enable_context = bool(getattr(config, "PII_CONTEXT_ENABLED", True))
        self.extra_fields = list(getattr(config, "PII_EXTRA_FIELDS", []) or [])
        self._context_rules = build_context_rules(self.extra_fields)
        self.ner_max_segment = max(50, int(ner_max_segment))

        self._pipeline = None
        self._ner_loaded = False
        self._ner_failed = False
        self.load_error: Optional[str] = None
        #: 加载锁：并发调用（预热 / 上传 / 提问）只构建一次管道
        self._ner_lock = threading.Lock()

    # ==================================================================
    # NER 模型（懒加载）
    # ==================================================================
    @property
    def ner_available(self) -> bool:
        """NER 是否已成功加载（调用前请先 :meth:`load_ner`）。"""
        return self._pipeline is not None

    def _limit_cpu_threads(self) -> None:
        """按指定线程数限制 NER 推理的 CPU 占用（PII_NER_CPU_THREADS）。

        数值即 torch 线程数，超过逻辑核数时按核数封顶；
        设为 0 时自动取逻辑核数的一半（保守档，至少 1）。
        线程数就是占用上限的来源——避免索引期间跑满所有核、影响同时用电脑。
        """
        try:
            import torch

            total = os.cpu_count() or 4
            configured = int(config.PII_NER_CPU_THREADS)
            threads = configured if configured > 0 else max(1, total // 2)
            threads = max(1, min(threads, total))
            torch.set_num_threads(threads)
            get_logger().info(
                "NER CPU 限制 torch 线程=%d / 逻辑核数=%d", threads, total
            )
        except Exception as exc:  # noqa: BLE001
            get_logger().warning("NER CPU 线程限制未生效：%s", exc)

    def load_ner(self) -> bool:
        """加载本地 NER 管道；失败时返回 False 并记录原因（不抛异常）。

        线程安全：加锁后只构建一次。启动预热 / 上传 / 提问脱敏并发调用时，
        后来的调用会等第一次加载完成，不会重复构建，也不会拿到"加载中"的空结果。
        """
        if not self.enable_ner:
            self.load_error = "已通过 PII_NER_ENABLED=false 关闭 NER 识别"
            return False
        with self._ner_lock:
            if self._ner_loaded:
                return self._pipeline is not None
            self._ner_loaded = True

            try:
                import modelscope  # noqa: F401
            except Exception as exc:  # pragma: no cover - 依赖缺失
                self._ner_failed = True
                self.load_error = (
                    f"未安装 modelscope（{exc}）。已降级为仅正则脱敏；"
                    "如需识别人名/地名/机构名，请执行：pip install modelscope torch"
                )
                return False

            try:
                self._limit_cpu_threads()
                from modelscope.pipelines import pipeline

                try:
                    self._pipeline = pipeline(
                        task="named-entity-recognition", model=self.model_id
                    )
                except TypeError:
                    from modelscope.utils.constant import Tasks

                    self._pipeline = pipeline(Tasks.named_entity_recognition, self.model_id)
            except Exception as exc:  # pragma: no cover - 模型加载失败
                self._ner_failed = True
                self.load_error = f"本地 NER 模型加载失败（{exc}）。已降级为仅正则脱敏。"
                return False

            self.load_error = None
            return True

    def _ner_detect(self, text: str) -> List[Dict[str, Any]]:
        """调用 NER 模型，长文本按句切段后偏移合并。"""
        if not self.load_ner():
            return []

        entities: List[Dict[str, Any]] = []
        for segment, offset in self._segments(text):
            try:
                result = self._pipeline(segment)
            except Exception:
                continue
            for item in self._parse_ner_output(result):
                start = int(item.get("start", 0)) + offset
                end = int(item.get("end", 0)) + offset
                span = item.get("span") or text[start:end]
                if not span or start >= end:
                    continue
                entities.append(
                    {
                        "start": start,
                        "end": end,
                        "text": span,
                        "label": str(item.get("type", "")).upper(),
                        "source": "ner",
                        "priority": 50,
                    }
                )
        return entities

    @staticmethod
    def _parse_ner_output(result: Any) -> List[Dict[str, Any]]:
        if not result:
            return []
        if isinstance(result, dict):
            output = result.get("output", result)
            if isinstance(output, dict):
                output = output.get("output", [])
        else:
            output = result
        if not isinstance(output, list):
            return []
        return [item for item in output if isinstance(item, dict)]

    def _segments(self, text: str) -> List[Tuple[str, int]]:
        """把长文本切成模型可处理的小段，返回 (文本段, 在原文中的偏移)。"""
        if len(text) <= self.ner_max_segment:
            return [(text, 0)]

        sentences = [s for s in _SENT_SPLIT_RE.split(text) if s]
        segments: List[Tuple[str, int]] = []
        buffer = ""
        offset = 0

        for sentence in sentences:
            if len(sentence) > self.ner_max_segment:
                if buffer:
                    segments.append((buffer, offset))
                    offset += len(buffer)
                    buffer = ""
                for start in range(0, len(sentence), self.ner_max_segment):
                    piece = sentence[start:start + self.ner_max_segment]
                    segments.append((piece, offset + start))
                offset += len(sentence)
                continue

            if len(buffer) + len(sentence) <= self.ner_max_segment:
                buffer += sentence
            else:
                segments.append((buffer, offset))
                offset += len(buffer)
                buffer = sentence

        if buffer:
            segments.append((buffer, offset))
        return segments

    # ==================================================================
    # 正则补充识别
    # ==================================================================
    def _regex_detect(self, text: str) -> List[Dict[str, Any]]:
        if not self.enable_regex:
            return []
        entities: List[Dict[str, Any]] = []
        for label, pattern, priority in REGEX_RULES:
            try:
                for match in re.finditer(pattern, text):
                    value = match.group()
                    if not value.strip():
                        continue
                    entities.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "text": value,
                            "label": label,
                            "source": "regex",
                            "priority": priority,
                        }
                    )
            except re.error:
                continue
        return entities

    # ==================================================================
    # 标签锚定识别（结构化文档，不依赖模型）
    # ==================================================================
    def _context_detect(self, text: str) -> List[Dict[str, Any]]:
        """按「字段名 + 冒号」定位中文 PII，并用校验器过滤误报。"""
        if not self.enable_context:
            return []

        entities: List[Dict[str, Any]] = []
        for label, pattern, validator, priority in self._context_rules:
            rule = _CONTEXT_COMPILED.get(pattern)
            if rule is None:
                try:
                    rule = re.compile(pattern)
                except re.error:
                    continue
                _CONTEXT_COMPILED[pattern] = rule

            checker = _VALIDATORS.get(validator)
            for match in rule.finditer(text):
                value = (match.group("v") or "").strip()
                if not value or (checker and not checker(value)):
                    continue
                start = match.start("v")
                entities.append(
                    {
                        "start": start,
                        "end": start + len(value),
                        "text": value,
                        "label": label,
                        "source": "context",
                        "priority": priority,
                    }
                )
        return entities

    # ==================================================================
    # 识别与合并
    # ==================================================================
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """识别文本中的全部敏感实体，返回按位置升序、互不重叠的实体列表。"""
        text = text or ""
        if not text.strip():
            return []

        raw: List[Dict[str, Any]] = []
        raw.extend(self._context_detect(text))
        raw.extend(self._regex_detect(text))
        raw.extend(self._ner_detect(text))

        # 优先级高者优先；同优先级下取更长的匹配
        raw.sort(key=lambda e: (-int(e["priority"]), e["start"], -(e["end"] - e["start"])))

        accepted: List[Dict[str, Any]] = []
        occupied: List[Tuple[int, int]] = []
        for entity in raw:
            start, end = int(entity["start"]), int(entity["end"])
            if start >= end or end > len(text):
                continue
            value = entity["text"]
            if not value.strip() or _USELESS_RE.match(value):
                continue
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            entity["type"] = self._to_placeholder_label(entity.get("label", ""))
            accepted.append(entity)

        accepted.sort(key=lambda e: e["start"])
        return accepted

    @staticmethod
    def _to_placeholder_label(label: str) -> str:
        key = (label or "").strip().upper()
        if not key:
            return "敏感信息"
        return ENTITY_TYPE_MAP.get(key, key)

    # ==================================================================
    # 替换与映射分配
    # ==================================================================
    def redact_text(
        self,
        text: str,
        kb_name: str = "",
        doc_id: str = "",
        pending: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> Dict[str, Any]:
        """对单段文本脱敏。

        :param pending: 本次批量处理中已分配的 (实体类型, 原文) -> 占位符 缓存，
                        用于保证同一批内相同实体复用同一占位符。
        """
        entities = self.detect(text)
        if not entities:
            return {"redacted_content": text, "redaction_map": {}, "has_sensitive": False}

        pending = pending if pending is not None else {}
        records: List[Dict[str, Any]] = []
        mapping: Dict[str, str] = {}

        # 按位置倒序替换，避免索引偏移
        for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
            entity_type = entity["type"]
            original = entity["text"]
            key = (entity_type, original)

            placeholder = pending.get(key)
            if placeholder is None:
                placeholder = self._allocate(entity_type, original, kb_name, doc_id, records)
                pending[key] = placeholder

            mapping[placeholder] = original
            text = text[: entity["start"]] + placeholder + text[entity["end"]:]

        return {"redacted_content": text, "redaction_map": mapping, "has_sensitive": True}

    def _allocate(
        self,
        entity_type: str,
        original: str,
        kb_name: str,
        doc_id: str,
        records: List[Dict[str, Any]],
    ) -> str:
        """分配（或复用）全局唯一的占位符，并登记到脱敏映射表。"""
        if self.metadata is not None:
            existing = self.metadata.find_placeholder(entity_type, original)
            if existing:
                return existing
            index = self.metadata.next_placeholder_index(entity_type)
        else:
            index = 1

        placeholder = f"[{entity_type}{index}]"

        if self.metadata is not None:
            record = {
                "kb_name": kb_name or "",
                "doc_id": doc_id or "",
                "placeholder": placeholder,
                "original": original,
                "entity_type": entity_type,
            }
            records.append(record)
            # 立即落库，保证后续批次不会分配到相同编号
            self.metadata.add_redactions([record])

        return placeholder

    # ==================================================================
    # 批处理：对解析出的文本块整体脱敏
    # ==================================================================
    def redact_blocks(
        self,
        blocks: List[Dict[str, Any]],
        kb_name: str = "",
        doc_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], int]:
        """对块列表脱敏，返回 (脱敏后的块, 累积映射表, 命中实体数)。"""
        pending: Dict[Tuple[str, str], str] = {}
        accumulated: Dict[str, str] = {}
        hit_count = 0
        total_blocks = len(blocks)
        log = get_logger()

        for block_index, block in enumerate(blocks, start=1):
            for field in ("content", "context"):
                value = block.get(field)
                if not value:
                    continue
                result = self.redact_text(value, kb_name=kb_name, doc_id=doc_id, pending=pending)
                if result["has_sensitive"]:
                    block[field] = result["redacted_content"]
                    accumulated.update(result["redaction_map"])
                    hit_count += len(result["redaction_map"])
            # 只记进度与统计量，不记录文档内容
            if doc_id and total_blocks >= 40 and (
                block_index % 20 == 0 or block_index == total_blocks
            ):
                log.info(
                    "脱敏进度 doc=%s %d/%d块 已命中=%d",
                    doc_id, block_index, total_blocks, hit_count,
                )

        return blocks, accumulated, hit_count

    # ==================================================================
    # 状态
    # ==================================================================
    @property
    def ner_loaded(self) -> bool:
        """是否已经尝试加载过模型（不代表加载成功）。"""
        return self._ner_loaded

    def status(self) -> Dict[str, Any]:
        """返回脱敏器状态（不会触发模型加载）。"""
        return {
            "enabled": True,
            "ner_enabled": self.enable_ner,
            "ner_loaded": self._ner_loaded,
            "ner_available": self.ner_available,
            "regex_enabled": self.enable_regex,
            "context_enabled": self.enable_context,
            "model_id": self.model_id,
            "error": self.load_error,
        }
