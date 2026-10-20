# -*- coding: utf-8 -*-
"""总路由聚合：所有子路由按 api_prefix 挂载。"""

from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.dept import router as dept_router
from app.api.routes.health import router as health_router
from app.api.routes.images import router as images_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.session import router as session_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(dept_router)
api_router.include_router(images_router)
api_router.include_router(knowledge_router)
api_router.include_router(session_router)
api_router.include_router(health_router)
