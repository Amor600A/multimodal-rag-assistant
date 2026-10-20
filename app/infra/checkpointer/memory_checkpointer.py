# -*- coding: utf-8 -*-
"""Checkpointer 入口（向后兼容）。

实际实现已迁移到 factory.py（按 CHECKPOINTER_BACKEND 分发）。
此模块保留 get_checkpointer() 作为统一入口，避免调用方改 import。
"""

from __future__ import annotations

from app.infra.checkpointer.factory import get_checkpointer  # noqa: F401

__all__ = ["get_checkpointer"]
