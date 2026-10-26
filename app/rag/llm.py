# -*- coding: utf-8 -*-
"""LLM 工厂：按 LLM_BACKEND 返回对应语言模型实例。

LLM_BACKEND=mock         ：返回 None，调用方走 mock_llm.pick_reply
LLM_BACKEND=tongyi       ：返回 ChatTongyi（DashScope，model_name 由 LLM_MODEL 指定）
LLM_BACKEND=openai_compatible：预留给 OpenAI 兼容端点实现（阶段 8）
"""

from __future__ import annotations

import logging

from app.core.config import get_settings

logger = logging.getLogger("app.rag.llm")


_llm = None


def get_llm():
    """进程级 LLM 单例。

    返回值约定：mock 模式返回 None（调用方自行处理关键词匹配兜底）；
    真实后端返回 LangChain BaseChatModel 实例，可直接 .invoke([SystemMessage, HumanMessage])。
    """
    global _llm
    if _llm is not None:
        return _llm

    settings = get_settings()
    backend = settings.llm_backend
    if backend == "tongyi":
        from langchain_community.chat_models.tongyi import ChatTongyi

        _llm = ChatTongyi(
            model_name=settings.llm_model,
            dashscope_api_key=settings.embedding_api_key,
        )
        logger.info(
            "LLM 已初始化：tongyi（model=%s）",
            settings.llm_model,
        )
    elif backend == "openai_compatible":
        # 阶段 8 实现：OpenAI Chat Completions 接口
        _llm = None
        logger.warning("LLM_BACKEND=openai_compatible 暂未实现，generate 节点将走 mock 路径")
    else:
        # mock：返回 None，调用方走 pick_reply
        _llm = None
        logger.info("LLM 已初始化：mock（使用关键词匹配兜底）")
    return _llm
