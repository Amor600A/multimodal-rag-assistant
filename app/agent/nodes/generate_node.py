# -*- coding: utf-8 -*-
"""generate 节点：片段注入 Prompt + 偏好注入 + 末尾来源引用块。

mock LLM 模式下：pick_reply 拿基础回复，命中召回片段则追加 `来源: [1]`。
真实 LLM 模式下（阶段 8）：把片段注入 system prompt，模型生成带引用的回复。
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.mock_llm import pick_reply
from app.agent.state import AgentState
from app.services.memory_service import get_memory_service

logger = logging.getLogger("app.agent.nodes.generate")


async def generate_node(state: AgentState) -> dict:
    """生成节点：基于召回片段 + 用户偏好生成回复。"""
    user_text = ""
    user_id = state.get("user_id", "")
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content)
            break

    reply = pick_reply(user_text)

    # 偏好注入
    pref = get_memory_service().get_reply_preference(user_id)
    if pref:
        reply = f"[按您的偏好：{pref}] {reply}"

    # 图片理解注入（vision 节点产出，方案 7）
    vision_desc = state.get("vision_desc")
    if vision_desc:
        reply = f"[图片内容：{vision_desc}] {reply}"

    # 末尾来源引用块：取第一个召回片段（mock 模式简化）
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        reply = f"{reply}\n\n来源: [1]"
    else:
        reply = f"{reply}\n\n来源: 无"

    return {"messages": [AIMessage(content=reply)]}
