"""问答 Agent（AI 能力 · 云端 LLM）。

职责：整合检索和生成。
工作流程：
    1. 接收用户问题和检索到的知识片段
    2. 构建包含知识片段的 Prompt
    3. 调用大模型生成回答
    4. 解析回答中的引用编号，回传引用索引

输出契约::

    {
        "answer": "回答内容（隐私模式下可能含占位符）",
        "cite_indices": [1, 3]
    }

说明：本单元不负责检索，也不负责生成引用链接；检索由调用方完成后注入 contexts。
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterator, List, Optional

import config
from modules.logutil import get_logger, next_seq

_CITE_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = """你是一名严谨的文档知识库问答助手。请严格遵循以下规则：
1. 只依据【知识片段】中的内容作答，不得编造、不得引入片段之外的知识。
2. 每条结论后必须用 [编号] 标注来源，编号对应知识片段的序号，例如 [1]、[2]。
3. 如果知识片段不足以回答，请直接说明"根据现有资料无法回答"，并指出还缺少什么信息。
4. 使用简体中文，条理清晰；涉及数字、日期、条款时保持与原文一致。
5. 若片段中出现 [姓名1]、[电话1] 这类方括号占位符，请原样保留，不要推测或改写它们。"""

PRIVACY_HINT = (
    "注意：本次知识片段中的部分敏感信息已被替换为 [姓名1]、[机构1]、[电话1] 等占位符。"
    "请把这些占位符当作完整、正常的信息直接引用输出（例如「联系电话为 [电话1]」），"
    "不要说明它们是占位符，不要声称资料不足或无法提供该信息，也不要尝试猜测其真实内容。"
)


class LLMError(RuntimeError):
    """大模型调用失败。"""


class QAAgent:
    """DeepSeek（OpenAI 兼容协议）问答 Agent。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else float(temperature)
        self.max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else int(max_tokens)
        self.timeout = config.LLM_TIMEOUT if timeout is None else int(timeout)
        self._client = None
        # 部分模型（如 DashScope 上的 kimi-k3）不支持 temperature / max_tokens，
        # 这里记录探测结果，避免每次请求都白跑一次失败调用。
        self._temperature_supported = True
        self._max_tokens_supported = True
        self.notes: List[str] = []

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}"

    # ------------------------------------------------------------------
    # 通用文本生成（供图表识别器等单元复用）
    # ------------------------------------------------------------------
    def complete(self, prompt: str, system: str = "你是一名严谨的中文文档分析助手。") -> str:
        return self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        mode: str = "normal",
    ) -> Dict[str, Any]:
        """基于知识片段回答问题。

        :param contexts: [{"index": 1, "content": "...", "source_kb": "kb", "file_name": "...", "page": 3}]
        """
        if not self.is_available():
            raise LLMError("问答模型未配置：请设置 DEEPSEEK_API_KEY")

        question = (question or "").strip()
        if not question:
            raise LLMError("问题不能为空")

        system = SYSTEM_PROMPT if mode != "privacy" else f"{SYSTEM_PROMPT}\n\n{PRIVACY_HINT}"
        user_prompt = self._build_prompt(question, contexts, privacy=mode == "privacy")
        text = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return {"answer": text, "cite_indices": self.extract_citations(text)}

    # ------------------------------------------------------------------
    def _build_prompt(
        self, question: str, contexts: List[Dict[str, Any]], privacy: bool = False
    ) -> str:
        """拼装知识片段 + 问题。

        :param privacy: 隐私模式下**不发送知识库名与文件名**——文件名常含人名
            （如「健康保险合同-石井然.pdf」），带上去等于把 PII 直接发往云端。
            模型只需要片段内容即可作答，来源标注由引用编号在本地完成。
        """
        if not contexts:
            return f"【知识片段】\n（无）\n\n【问题】\n{question}"

        blocks: List[str] = []
        used_chars = 0
        for context in contexts:
            content = (context.get("content") or "").strip()
            if not content:
                continue
            if used_chars + len(content) > config.MAX_CONTEXT_CHARS:
                content = content[: max(0, config.MAX_CONTEXT_CHARS - used_chars)]
            if not content:
                break
            used_chars += len(content)

            index = context.get("index")
            if privacy:
                page = context.get("page")
                source = f"来源：第 {page} 页" if page else "来源：本地片段"
            else:
                location = f"，第 {context['page']} 页" if context.get("page") else ""
                source = (
                    f"来源：{context.get('source_kb', '未知知识库')} / "
                    f"{context.get('file_name', '未知文件')}{location}"
                )
            blocks.append(f"[{index}] {source}\n{content}")

        joined = "\n\n".join(blocks) if blocks else "（无）"
        return (
            f"【知识片段】\n{joined}\n\n"
            f"【问题】\n{question}\n\n"
            "【回答要求】\n给出结论，并在每条结论后用 [编号] 标注来源；"
            "若资料不足请直接说明。"
        )

    def answer_stream(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        mode: str = "normal",
    ) -> Iterator[str]:
        """流式生成回答，逐段产出文本增量（供 UI 做渐入输出）。"""
        if not self.is_available():
            raise LLMError("问答模型未配置：请设置 DEEPSEEK_API_KEY")
        question = (question or "").strip()
        if not question:
            raise LLMError("问题不能为空")

        system = SYSTEM_PROMPT if mode != "privacy" else f"{SYSTEM_PROMPT}\n\n{PRIVACY_HINT}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._build_prompt(question, contexts, privacy=mode == "privacy")},
        ]

        client = self._get_client()
        kwargs = self._base_kwargs(messages, stream=True)

        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        seq = next_seq("LLM")
        log = get_logger()
        started = time.perf_counter()
        log.info(
            "LLM调用 #%d 流式开始 model=%s prompt_chars=%d",
            seq, self.model, prompt_chars,
        )

        last_error: Optional[Exception] = None
        stream = None
        for _ in range(3):
            try:
                stream = client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not self._adapt_kwargs(kwargs, exc):
                    log.error("LLM调用 #%d 建立流式失败：%s", seq, exc)
                    raise LLMError(f"调用大模型失败：{exc}") from exc
        if stream is None:
            log.error("LLM调用 #%d 建立流式失败：%s", seq, last_error)
            raise LLMError(f"调用大模型失败：{last_error}")

        output_chars = 0
        finish_reason = ""
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                reason = getattr(choices[0], "finish_reason", None)
                if reason:
                    finish_reason = str(reason)
                delta = getattr(choices[0].delta, "content", None)
                if delta:
                    output_chars += len(delta)
                    yield delta
        except GeneratorExit:
            # 用户点「停止」时这个生成器会在 yield 处被关掉（GeneratorExit），
            # 主动关掉 HTTP 流，云端才能立刻察觉连接断开并停止继续生成。
            # 注意：**已经吐出来的 token 仍会照常计费**，这里省下的是后面没生成的部分。
            log.info(
                "LLM调用 #%d 流式被停止 已输出chars=%d 耗时=%.1fs",
                seq, output_chars, time.perf_counter() - started,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            log.error(
                "LLM调用 #%d 流式中断 已输出chars=%d 耗时=%.1fs 错误=%s",
                seq, output_chars, time.perf_counter() - started, exc,
            )
            raise LLMError(f"读取流式响应失败：{exc}") from exc
        finally:
            self._close_stream(stream)

        elapsed = time.perf_counter() - started
        if not finish_reason:
            # 整个流没带任何 finish_reason 就结束了：连接多半被提前掐断，
            # 回答很可能停在半句——排查「回答显示一半」类问题的直接证据。
            log.warning(
                "LLM调用 #%d 流式结束(未收到finish_reason，流可能被提前断开) "
                "model=%s prompt_chars=%d 已输出chars=%d 耗时=%.1fs",
                seq, self.model, prompt_chars, output_chars, elapsed,
            )
        elif finish_reason != "stop":
            # length：命中 max_tokens；其余值：模型侧提前收尾
            log.warning(
                "LLM调用 #%d 流式结束(未正常完结) finish_reason=%s model=%s "
                "prompt_chars=%d 已输出chars=%d 耗时=%.1fs",
                seq, finish_reason, self.model, prompt_chars, output_chars, elapsed,
            )
        else:
            log.info(
                "LLM调用 #%d 流式完成 model=%s prompt_chars=%d output_chars=%d 耗时=%.1fs",
                seq, self.model, prompt_chars, output_chars, elapsed,
            )

    @staticmethod
    def _close_stream(stream: Any) -> None:
        """尽力关闭流式响应（不同 SDK 的关闭方法名不一致，缺失就跳过）。"""
        close = getattr(stream, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def _base_kwargs(self, messages: List[Dict[str, str]], stream: bool) -> Dict[str, Any]:
        """组装请求参数，只带上当前模型支持的选项。"""
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if self._temperature_supported and self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self._max_tokens_supported and self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def _adapt_kwargs(self, kwargs: Dict[str, Any], exc: Exception) -> bool:
        """遇到「参数不支持」类错误时去掉该参数并记住，返回是否做了调整。

        不同模型对可选参数的支持不同（例如 DashScope 上的 kimi-k3 不接受 temperature），
        适配结果会被记住，后续请求不再白跑一次失败调用。
        """
        lowered = str(exc).lower()
        adjusted = False

        if "temperature" in kwargs and "temperature" in lowered:
            kwargs.pop("temperature")
            self._temperature_supported = False
            self._note(f"模型 {self.model} 不支持 temperature，已自动省略该参数")
            adjusted = True
        if "max_tokens" in kwargs and "max_tokens" in lowered:
            kwargs.pop("max_tokens")
            self._max_tokens_supported = False
            self._note(f"模型 {self.model} 不支持 max_tokens，已自动省略该参数")
            adjusted = True

        return adjusted

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """非流式调用，返回完整回答。"""
        client = self._get_client()
        kwargs = self._base_kwargs(messages, stream=False)

        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        seq = next_seq("LLM")
        started = time.perf_counter()

        last_error: Optional[Exception] = None
        for _ in range(3):
            try:
                content = self._extract_content(client.chat.completions.create(**kwargs))
                get_logger().info(
                    "LLM调用 #%d 非流式 model=%s prompt_chars=%d output_chars=%d 耗时=%.1fs",
                    seq, self.model, prompt_chars, len(content), time.perf_counter() - started,
                )
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not self._adapt_kwargs(kwargs, exc):
                    break

        get_logger().error(
            "LLM调用 #%d 非流式失败 model=%s prompt_chars=%d 错误=%s",
            seq, self.model, prompt_chars, last_error,
        )
        raise LLMError(f"调用大模型失败：{last_error}")

    def _note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"解析大模型响应失败：{exc}") from exc
        return str(content).strip()

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url or None,
                timeout=self.timeout,
            )
        return self._client

    # ------------------------------------------------------------------
    @staticmethod
    def extract_citations(text: str) -> List[int]:
        """从回答中提取引用编号（去重并保持出现顺序）。"""
        seen: List[int] = []
        for match in _CITE_RE.finditer(text or ""):
            number = int(match.group(1))
            if number not in seen:
                seen.append(number)
        return seen
