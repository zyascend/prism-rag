"""统一日志初始化 — structlog 配置

替换各入口脚本中散落的 logging.basicConfig()，
提供一处调用完成全项目日志配置。
"""
from __future__ import annotations

import logging
import structlog


def init_logging(
    level: str = "INFO",
    log_file: str | None = "logs/app.jsonl",
    console: bool = True,
) -> None:
    """配置 structlog：JSON 到文件 + 彩色到控制台

    Args:
        level: 日志级别 (DEBUG | INFO | WARNING | ERROR)
        log_file: JSON 日志输出路径，None 则不写文件
        console: 是否输出彩色控制台日志
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 共享处理器列表
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if console
        else structlog.processors.JSONRenderer(),
    ]

    # 文件输出: JSON
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )

    # 控制台输出: 彩色
    console_handler = None
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(),
            )
        )

    # 配置 root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    if log_file:
        root_logger.addHandler(file_handler)
    if console and console_handler:
        root_logger.addHandler(console_handler)

    # 配置 structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if console else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )