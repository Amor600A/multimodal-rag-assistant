# -*- coding: utf-8 -*-
"""FastAPI 应用入口。

阶段 2 增量：lifespan 注入种子数据 + 注册 auth/dept/session 路由。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logger import setup_logging

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期。

    - 启动：日志初始化 → 种子数据注入（Mock 后端必备，重启后数据不丢）
    - 关闭：记录日志（当前阶段无资源释放）
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    # 启动横幅 + 组件清单（方案阶段 1 验收点）
    logger.info("=" * 64)
    logger.info("%s 启动（%s 模式，全部后端为 Mock/内存实现）", settings.app_name, settings.llm_backend)
    for name, backend in settings.backend_summary().items():
        logger.info("  %-22s %s", name, backend)
    tools = settings.enabled_tools_list
    logger.info("  %-22s %s", "工具调用", ",".join(tools) if tools else "未启用（纯 RAG 链路）")
    logger.info("  %-22s %s", "输出护栏", "开" if settings.output_guard_enabled else "关")
    logger.info("  %-22s %s", "记忆入向量库", "开" if settings.memory_to_vector_enabled else "关")

    # 种子数据注入（阶段 2 增量）
    if settings.user_db_backend == "mock_postgres":
        from scripts.seed_data import seed_all

        await seed_all()

    logger.info("=" * 64)

    yield

    logger.info("%s 已关闭", settings.app_name)


def create_app() -> FastAPI:
    """应用工厂：路由挂载点 + 前端静态资源托管。"""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(api_router, prefix=settings.api_prefix)

    # 同域托管前端（web/），uploads/ 不挂静态（仅经代理端点访问）
    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app


app = create_app()
