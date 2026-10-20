# -*- coding: utf-8 -*-
"""Mock 部门仓储（内存模拟 departments 表）。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Department:
    id: str
    code: str
    name: str
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


class MockDepartmentRepository:
    def __init__(self) -> None:
        self._by_code: dict[str, Department] = {}
        self._by_id: dict[str, Department] = {}

    async def create(self, *, code: str, name: str, parent_id: Optional[str] = None) -> Department:
        if code in self._by_code:
            raise ValueError(f"部门 {code!r} 已存在")
        d = Department(id=uuid.uuid4().hex, code=code, name=name, parent_id=parent_id)
        self._by_code[code] = d
        self._by_id[d.id] = d
        return d

    async def get_by_code(self, code: str) -> Optional[Department]:
        return self._by_code.get(code)

    async def get_by_id(self, dept_id: str) -> Optional[Department]:
        return self._by_id.get(dept_id)

    async def all(self) -> list[Department]:
        return list(self._by_code.values())


_dept_repo = MockDepartmentRepository()


def get_department_repository() -> MockDepartmentRepository:
    """部门仓储工厂（阶段 2 只暴露 mock 单例）。"""
    return _dept_repo
