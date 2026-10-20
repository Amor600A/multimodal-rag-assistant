# -*- coding: utf-8 -*-
"""Mock 会话元数据仓储（内存模拟 sessions 表）。

注：会话历史消息存在 langgraph checkpointer 中，本仓储只管 thread 的
元信息（归属用户、标题、时间戳），支撑 /api/sessions 列表/删除等接口。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Session:
    thread_id: str
    user_id: str
    title: str
    created_at: float = field(default_factory=time.time)


class MockSessionRepository:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}  # thread_id -> Session

    async def create(self, *, thread_id: str, user_id: str, title: str) -> Session:
        # thread_id 通常由客户端生成（uuid4），幂等：已存在直接返回
        existing = self._sessions.get(thread_id)
        if existing is not None:
            return existing
        s = Session(thread_id=thread_id, user_id=user_id, title=title)
        self._sessions[thread_id] = s
        return s

    async def get(self, thread_id: str) -> Optional[Session]:
        return self._sessions.get(thread_id)

    async def list_by_user(self, user_id: str) -> list[Session]:
        """返回某用户的全部会话，按 created_at 降序。"""
        items = [s for s in self._sessions.values() if s.user_id == user_id]
        items.sort(key=lambda s: s.created_at, reverse=True)
        return items

    async def delete(self, thread_id: str) -> bool:
        return self._sessions.pop(thread_id, None) is not None


_session_repo = MockSessionRepository()


def get_session_repository() -> MockSessionRepository:
    return _session_repo
