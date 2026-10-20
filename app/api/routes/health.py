# -*- coding: utf-8 -*-
"""健康检查路由。"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """存活探针：返回固定状态，不依赖任何后端组件。"""
    return {"status": "ok"}
