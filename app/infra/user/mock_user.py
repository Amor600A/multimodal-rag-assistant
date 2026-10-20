# -*- coding: utf-8 -*-
"""Mock 用户仓储（内存模拟 Postgres users 表）。

关键设计：模块级单例（不是每次请求新建），保证 seed_data 在 lifespan
中注入的数据能被后续路由处理器看见。factory 层按 USER_DB_BACKEND 分发。

接口方法与生产 PostgresUserRepository 保持一致，切真实后端时 factory
分支即可，调用方代码零改动。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    """领域模型（与 create_db.sql 表结构对齐）。"""
    id: str
    username: str
    password_hash: str
    email: Optional[str] = None
    role: str = "user"            # user | admin
    dept_id: Optional[str] = None # 调岗后实时更新，不缓存到 JWT
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login_at: Optional[float] = None


class MockUserRepository:
    """内存字典模拟，dict[username] = User（同时维护 id 索引）。"""

    def __init__(self) -> None:
        self._by_username: dict[str, User] = {}
        self._by_id: dict[str, User] = {}

    # -- 写操作（供 seed / auth_service 使用）--

    async def create(
        self,
        *,
        username: str,
        password_hash: str,
        role: str = "user",
        dept_id: Optional[str] = None,
        email: Optional[str] = None,
    ) -> User:
        if username in self._by_username:
            raise ValueError(f"用户 {username!r} 已存在")
        u = User(
            id=uuid.uuid4().hex,
            username=username,
            password_hash=password_hash,
            role=role,
            dept_id=dept_id,
            email=email,
        )
        self._by_username[username] = u
        self._by_id[u.id] = u
        return u

    async def update_last_login(self, user_id: str) -> None:
        u = self._by_id.get(user_id)
        if u is not None:
            u.last_login_at = time.time()

    async def set_dept(self, user_id: str, dept_id: Optional[str]) -> None:
        u = self._by_id.get(user_id)
        if u is not None:
            u.dept_id = dept_id

    # -- 读操作 --

    async def get_by_username(self, username: str) -> Optional[User]:
        return self._by_username.get(username)

    async def get_by_id(self, user_id: str) -> Optional[User]:
        return self._by_id.get(user_id)

    async def all(self) -> list[User]:
        return list(self._by_username.values())


# 进程级单例（关键！保证 seed 数据对所有请求可见）
_user_repo = MockUserRepository()


def get_user_repository() -> MockUserRepository:
    """用户仓储工厂（阶段 2 只暴露 mock 单例；切 postgres 时在此加分支）。"""
    return _user_repo
