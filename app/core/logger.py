# -*- coding: utf-8 -*-
"""日志统一配置：分层应用日志 + 独立审计日志（JSON 行）。

- 应用日志按模块路径天然分层（app.api / app.services / app.agent / app.infra），
  级别约定：请求出入口 api 层 4xx=WARNING / 5xx=ERROR；性能字段统一 duration_ms。
- 审计日志使用独立 logger "app.audit"，输出结构化 JSON 行，与普通日志分流；
  当前阶段只落控制台不建审计表，生产可接 ELK/落表，接口不变。
- Windows 控制台默认 GBK，输出中文会乱码，启动时将 stdout/stderr 重配置为 UTF-8。
"""

import json
import logging
import sys
from datetime import datetime, timezone

# 审计 logger 名称（独立分流，不向 root 传播）
AUDIT_LOGGER_NAME = "app.audit"

# 审计记录字段（见方案 13.2）
_AUDIT_FIELDS = ("ts", "user_id", "username", "action", "target", "detail", "ip")


def _setup_console_utf8() -> None:
    """Windows 控制台 GBK 输出中文乱码，统一重配置为 UTF-8（幂等）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                # 已销毁的流或环境不支持时忽略
                pass


def setup_logging(level: str = "INFO") -> None:
    """初始化应用日志与审计日志（进程启动时调用一次）。"""
    _setup_console_utf8()
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # 应用根 logger：所有 app.* 子模块按层级输出（app.audit 除外，见下）
    app_logger = logging.getLogger("app")
    app_logger.setLevel(log_level)
    app_logger.addHandler(handler)
    app_logger.propagate = False

    # 审计 logger：独立输出，不进应用日志流
    audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(handler)
    audit_logger.propagate = False


def audit(
    action: str,
    *,
    user_id: str | None = None,
    username: str | None = None,
    target: str | None = None,
    detail: str | None = None,
    ip: str | None = None,
) -> None:
    """写一条审计记录（结构化 JSON 行）。

    红线（见方案 13.2）：不记密码与 token 全文（token 只记 jti 前 8 位）、不记文档正文。
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "username": username,
        "action": action,
        "target": target,
        "detail": detail,
        "ip": ip,
    }
    line = json.dumps(record, ensure_ascii=False)
    logging.getLogger(AUDIT_LOGGER_NAME).info(line)
