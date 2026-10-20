# -*- coding: utf-8 -*-
"""Store 工厂：按 STORE_BACKEND 返回对应实现。

阶段 5：in_memory（默认，进程内）| mock_postgres（预置画像 + hash 语义检索）
阶段 8：postgres（AsyncPostgresStore，业务代码零改动）
"""

from __future__ import annotations

import warnings
from typing import Union

from langgraph.store.base import BaseStore

from app.core.config import get_settings

warnings.filterwarnings("ignore", message="LangChainPendingDeprecationWarning")

_cached: BaseStore | None = None


def get_store() -> BaseStore:
    """进程级 Store 单例。"""
    global _cached
    if _cached is not None:
        return _cached

    backend = get_settings().store_backend
    if backend == "in_memory":
        from langgraph.store.memory import InMemoryStore

        _cached = InMemoryStore()
    elif backend == "mock_postgres":
        from app.infra.store.mock_postgres import MockPostgresStore

        _cached = MockPostgresStore()
    else:
        # postgres 阶段 8 实现；当前 fallback 到 mock_postgres
        import logging

        logging.getLogger("app.infra.store").warning(
            "store_backend=%s 暂未实现，fallback 到 mock_postgres", backend
        )
        from app.infra.store.mock_postgres import MockPostgresStore

        _cached = MockPostgresStore()
    return _cached
