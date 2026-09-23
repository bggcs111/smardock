"""运行日志：把排查问题的关键事件写入 data/logs/smardock.log。

记录内容（一行一事，便于 grep）：
    * 云端 API 调用：LLM / Embedding / Rerank 的序号、耗时、输入输出字符数、成败。
      「调用几次 API」直接看每类调用的 #序号，序号在进程内累计。
    * 流式回答的结束方式：finish_reason=stop（正常完结）/ length（命中 max_tokens
      被截断）/ 空（连接被提前断开——回答显示一半就断的直接证据）。
    * 每次检索的命中数量与上下文规模。
    * 服务启动 / 退出。

隐私说明：日志只记字符数等统计量，不记录问题与文档的原文内容。

文件按大小轮转（默认 5MB × 6 个），不随服务退出清理，便于事后翻查。
"""
from __future__ import annotations

import logging
import logging.handlers
import threading
from pathlib import Path
from typing import Dict, Optional

import config

_LOGGER_NAME = "smardock"

#: 进程内累计的各类 API 调用次数（键：LLM / Embedding / Rerank）
_counters: Dict[str, int] = {}
_counters_lock = threading.Lock()
_logger: Optional[logging.Logger] = None


def setup_logging() -> logging.Logger:
    """初始化文件日志（幂等，重复调用返回同一个 logger）。"""
    global _logger
    if _logger is not None:
        return _logger

    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "smardock.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    # 不向 root logger 传播：控制台输出仍由原有 print/traceback 负责，避免重复
    logger.propagate = False

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """取运行日志 logger；未初始化时自动初始化（便于脚本直接使用）。"""
    return _logger or setup_logging()


def next_seq(kind: str) -> int:
    """返回该类调用的累计序号（从 1 开始），用于统计「调用了几次 API」。"""
    with _counters_lock:
        _counters[kind] = _counters.get(kind, 0) + 1
        return _counters[kind]


def call_totals() -> Dict[str, int]:
    """返回各类 API 调用的累计次数副本（进程内统计）。"""
    with _counters_lock:
        return dict(_counters)
