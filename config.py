"""全局配置模块。

所有可调参数集中在此，可通过环境变量或项目根目录下的 .env 文件覆盖。
其余模块只读取本模块的常量，不直接读取环境变量，保证配置来源单一。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# .env 加载（优先使用 python-dotenv，缺失时用内置极简解析器兜底）
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=str(path), override=False)
        return
    except Exception:
        pass

    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data"))).expanduser().resolve()
CHROMA_DIR = Path(_env("CHROMA_DIR", str(DATA_DIR / "chroma_db"))).expanduser().resolve()
METADATA_DB = Path(_env("METADATA_DB", str(DATA_DIR / "metadata.db"))).expanduser().resolve()
UPLOAD_DIR = Path(_env("UPLOAD_DIR", str(DATA_DIR / "uploaded_docs"))).expanduser().resolve()
EXPORT_DIR = Path(_env("EXPORT_DIR", str(DATA_DIR / "exports"))).expanduser().resolve()
#: 运行时临时目录：Gradio 的上传中转文件等都放这里，退出服务时整体清理
TEMP_DIR = Path(_env("TEMP_DIR", str(DATA_DIR / "tmp"))).expanduser().resolve()
#: 运行日志目录：关键事件（API 调用、检索、问答）写入这里的 smardock.log，不随退出清理
LOG_DIR = Path(_env("LOG_DIR", str(DATA_DIR / "logs"))).expanduser().resolve()

#: ModelScope 模型缓存目录：NER 模型下载到这里（不放系统盘的 ~/.cache）
MODELSCOPE_CACHE_DIR = Path(
    _env("MODELSCOPE_CACHE", str(BASE_DIR / ".cache" / "modelscope"))
).expanduser().resolve()
#: modelscope 在 import 时读取该环境变量，必须在此之前写回（本模块最先被 import）
os.environ["MODELSCOPE_CACHE"] = str(MODELSCOPE_CACHE_DIR)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# ---------------------------------------------------------------------------
# PDF 解析引擎
# ---------------------------------------------------------------------------
# pymupdf4llm：走版式模型，输出自带标题层级、列表与 Markdown 表格（推荐）
# pymupdf    ：原生块 + 字号启发式识别标题（无额外依赖）
PDF_ENGINE = _env("PDF_ENGINE", "pymupdf4llm").lower()

# ---------------------------------------------------------------------------
# 文本切片
# ---------------------------------------------------------------------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)
MIN_CHUNK_CHARS = _env_int("MIN_CHUNK_CHARS", 4)

# ---------------------------------------------------------------------------
# 向量化器（云端 Embedding API）
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "dashscope").lower()  # dashscope | openai
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 1024)
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 10)
EMBEDDING_MAX_CHARS = _env_int("EMBEDDING_MAX_CHARS", 3000)
EMBEDDING_MAX_RETRY = _env_int("EMBEDDING_MAX_RETRY", 3)

DASHSCOPE_API_KEY = _env("DASHSCOPE_API_KEY", "")
EMBEDDING_API_KEY = _env("EMBEDDING_API_KEY", DASHSCOPE_API_KEY)
# 仅 provider=openai 时使用（可指向 OpenAI / SiliconFlow / 本地 vLLM / Ollama 等兼容端点）
EMBEDDING_BASE_URL = _env("EMBEDDING_BASE_URL", "")

# ---------------------------------------------------------------------------
# 大语言模型（问答 Agent / 图表语义描述）
# ---------------------------------------------------------------------------
LLM_API_KEY = _env("DEEPSEEK_API_KEY", _env("LLM_API_KEY", ""))
LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = _env("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 2048)
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 120)

# 图表语义描述：调用云端 LLM 为图表生成语义描述（默认关闭，只做本地版面识别）
FIGURE_AI_DESCRIBE = _env_bool("FIGURE_AI_DESCRIBE", False)

# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
# 提高候选量：先把足够多的相关块捞出来，再靠「章节展开」补全上下文。
TOP_K_PER_KB = _env_int("TOP_K_PER_KB", 6)
TOP_K_TOTAL = _env_int("TOP_K_TOTAL", 10)
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 12000)
#: 最终交给大模型的片段数量上限
MAX_CONTEXTS = _env_int("MAX_CONTEXTS", 6)

# 命中块 → 所在章节完整内容（small-to-big 检索）。
# 数据手册里的列表/步骤常被切成多个小块，只喂命中块容易只看到一半。
SECTION_EXPAND = _env_bool("SECTION_EXPAND", True)
#: 单个章节最多拼接多少个块
SECTION_EXPAND_MAX_CHUNKS = _env_int("SECTION_EXPAND_MAX_CHUNKS", 8)
#: 单个章节拼接后的字符上限
MAX_SECTION_CHARS = _env_int("MAX_SECTION_CHARS", 3000)

# 混合检索：向量（语义）＋ BM25 关键词（精确词面），各自先取候选再融合。
# 向量擅长"意思相近"，关键词擅长 MIO12 / DS987 这类精确标识符，两者互补。
KEYWORD_SEARCH = _env_bool("KEYWORD_SEARCH", True)
CANDIDATE_VECTOR = _env_int("CANDIDATE_VECTOR", 20)
CANDIDATE_KEYWORD = _env_int("CANDIDATE_KEYWORD", 20)
#: RRF（倒数排名融合）平滑系数
RRF_K = _env_int("RRF_K", 60)

# 重排序：把融合后的候选交给重排序模型精排，再取前 N 条喂给大模型
RERANK_ENABLED = _env_bool("RERANK_ENABLED", True)
RERANK_MODEL = _env("RERANK_MODEL", "gte-rerank-v2")
RERANK_TOP_N = _env_int("RERANK_TOP_N", 8)

# ---------------------------------------------------------------------------
# PII 脱敏（本地 NER + 正则）
# ---------------------------------------------------------------------------
PII_NER_ENABLED = _env_bool("PII_NER_ENABLED", True)
# 标签锚定识别（姓名/地址/机构/编号/出生日期），纯代码、不依赖模型，默认开启
PII_CONTEXT_ENABLED = _env_bool("PII_CONTEXT_ENABLED", True)
# 隐私模式是否严格脱敏：开启后回答与引用也保持占位符，不在本地还原真实信息。
# 普通模式不受此开关影响（普通模式全程不脱敏、不还原）。
PRIVACY_STRICT_REDACTION = _env_bool("PRIVACY_STRICT_REDACTION", True)
# 隐私模式下是否给引用来源打码：文件名 / 本地路径 / 下载副本名统一换成「文档N」
PRIVACY_MASK_SOURCE = _env_bool("PRIVACY_MASK_SOURCE", True)
# 额外要脱敏的字段名（逗号分隔），例如：性别,职业类别,婚姻状况
PII_EXTRA_FIELDS = [item.strip() for item in _env("PII_EXTRA_FIELDS", "").split(",") if item.strip()]
PII_NER_MODEL_ID = _env(
    "PII_NER_MODEL_ID",
    "damo/nlp_raner_named-entity-recognition_chinese-base-generic",
)
PII_REGEX_ENABLED = _env_bool("PII_REGEX_ENABLED", True)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SERVER_NAME = _env("SERVER_NAME", "127.0.0.1")
SERVER_PORT = _env_int("SERVER_PORT", 7860)
SHARE = _env_bool("SHARE", False)
INBROWSER = _env_bool("INBROWSER", True)


def ensure_dirs() -> None:
    """确保所有运行时目录存在。"""
    for directory in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR, EXPORT_DIR, TEMP_DIR, LOG_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)


def cleanup_temp_files() -> List[str]:
    """清理临时缓存文件（退出服务时调用），返回已清理项的说明。

    只删可再生的东西：
        * 导出目录里的原件副本（浏览器下载用的临时拷贝）
        * Gradio 上传中转目录
        * 项目内的 __pycache__
    归档原件 `uploaded_docs/` 与向量库 `chroma_db/` 一概不动。
    """
    import shutil

    cleaned: List[str] = []

    for directory in (EXPORT_DIR, TEMP_DIR):
        path = Path(directory)
        if not path.exists():
            continue
        count = 0
        for child in path.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                count += 1
            except OSError:
                continue
        if count:
            cleaned.append(f"{path.name}/（{count} 项）")

    cache_dirs = [
        path
        for path in BASE_DIR.rglob("__pycache__")
        if ".venv" not in path.parts and path.is_dir()
    ]
    if cache_dirs:
        for path in cache_dirs:
            shutil.rmtree(path, ignore_errors=True)
        cleaned.append(f"__pycache__（{len(cache_dirs)} 个目录）")

    return cleaned


def config_status() -> List[Dict[str, Any]]:
    """配置自检，返回供 UI 展示的列表。"""
    items: List[Dict[str, Any]] = []

    if EMBEDDING_PROVIDER == "openai":
        embed_ok = bool(EMBEDDING_API_KEY and EMBEDDING_BASE_URL)
        embed_hint = "" if embed_ok else "需同时配置 EMBEDDING_API_KEY 与 EMBEDDING_BASE_URL"
    else:
        embed_ok = bool(EMBEDDING_API_KEY)
        embed_hint = "" if embed_ok else "未配置 DASHSCOPE_API_KEY，无法向量化文档"

    items.append(
        {
            "name": f"向量化器（{EMBEDDING_PROVIDER} / {EMBEDDING_MODEL}）",
            "ok": embed_ok,
            "hint": embed_hint,
        }
    )

    llm_ok = bool(LLM_API_KEY)
    items.append(
        {
            "name": f"问答模型（{LLM_MODEL} @ {LLM_BASE_URL}）",
            "ok": llm_ok,
            "hint": "" if llm_ok else "未配置 DEEPSEEK_API_KEY，无法生成回答",
        }
    )
    return items
