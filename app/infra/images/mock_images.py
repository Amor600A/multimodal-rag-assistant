# -*- coding: utf-8 -*-
"""Mock 图片元数据仓储（内存字典，进程级单例）。

对应方案 5.4 images 表：image_id / user_id / mime / file_path / bound / created_at。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# 进程级单例：进程内所有请求共享同一份图片元数据
_store: dict[str, "ImageRecord"] = {}


@dataclass
class ImageRecord:
    image_id: str
    user_id: str
    mime: str
    file_path: str
    bound: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class MockImageRepository:
    """内存图片元数据仓储。"""

    def __init__(self) -> None:
        self._store = _store

    async def create(self, record: ImageRecord) -> ImageRecord:
        self._store[record.image_id] = record
        return record

    async def get(self, image_id: str) -> Optional[ImageRecord]:
        return self._store.get(image_id)

    async def mark_bound(self, image_id: str) -> None:
        """标记图片已被会话消息实际引用（孤儿清理判定依据）。"""
        rec = self._store.get(image_id)
        if rec:
            rec.bound = True


_instance: Optional[MockImageRepository] = None


def get_image_repository() -> MockImageRepository:
    """获取进程级单例仓储。"""
    global _instance
    if _instance is None:
        _instance = MockImageRepository()
    return _instance
