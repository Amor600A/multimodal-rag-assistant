# -*- coding: utf-8 -*-
"""retrieve 节点：调 rag_service 双路召回，结果写入 state.retrieved_chunks。"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.core.deps import CurrentUser
from app.services.rag_service import get_rag_service

logger = logging.getLogger("app.agent.nodes.retrieve")


async def retrieve_node(state: AgentState) -> dict:
    """检索节点：根据用户消息 + 当前身份做权限过滤双路召回。"""
    user_text = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content)
            break

    user = CurrentUser(
        user_id=state.get("user_id", ""),
        username="",
        role=state.get("role", "user"),
        dept_id=state.get("dept_id"),
    )

    # 含图片时把 vision 节点产出的图片描述并入查询词（方案 7：图与文一起进检索）
    query = user_text
    vision_desc = state.get("vision_desc")
    if vision_desc:
        query = f"{user_text} {vision_desc}"

    chunks = get_rag_service().search(user, query, limit=5)
    # 转成可序列化 dict 写入 state
    chunk_dicts = [
        {
            "index": c.index,
            "doc_id": c.doc_id,
            "text": c.text,
            "source": c.source,
            "scope": c.scope,
            "score": c.score,
        }
        for c in chunks
    ]
    logger.info("retrieve: query=%s hits=%d", query[:30], len(chunk_dicts))
    return {"retrieved_chunks": chunk_dicts}
