# -*- coding: utf-8 -*-
"""Checkpointer 工厂：按 CHECKPOINTER_BACKEND 返回对应实现。

阶段 4：memory（默认，重启即失）| mock_redis（磁盘持久化 + 预置快照，重启可回放）
阶段 8：redis / postgres（真实后端，业务代码零改动）
"""

from __future__ import annotations

import warnings
from typing import Union

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.config import get_settings
from app.infra.checkpointer.mock_redis import MockRedisSaver

warnings.filterwarnings("ignore", message="LangChainPendingDeprecationWarning")

_cached: BaseCheckpointSaver | None = None


def get_checkpointer() -> BaseCheckpointSaver:
    """进程级 checkpointer 单例。

    读取 settings.checkpointer_backend 决定实现：
    - memory: MemorySaver（langgraph 官方，进程内字典，重启即失）
    - mock_redis: MockRedisSaver（磁盘持久化 + 预置快照，重启可回放）
    - redis / postgres: 阶段 8 接入真实后端时在此分支
    """
    global _cached
    if _cached is not None:
        return _cached

    backend = get_settings().checkpointer_backend
    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        _cached = MemorySaver()
    elif backend == "mock_redis":
        _cached = MockRedisSaver()
    else:
        # redis / postgres 阶段 8 实现；当前阶段 fallback 到 mock_redis 保证可用
        from app.core.logger import setup_logging  # noqa: F401

        import logging

        logging.getLogger("app.infra.checkpointer").warning(
            "checkpointer_backend=%s 暂未实现，fallback 到 mock_redis", backend
        )
        _cached = MockRedisSaver()
    return _cached
