# -*- coding: utf-8 -*-
"""Agent 状态定义。

LangGraph StateGraph 的每个 node 返回 dict，默认会覆盖 state 里的同名字段。
加 Annotated[..., operator.add] reducer 的字段才会累加（消息列表）。
后续 ReAct 工具调用阶段再加 tool_calls / tool_results 等字段。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Agent 图状态。

    字段说明：
    - messages: 累计的对话历史（Annotated[list, operator.add] 保证累加不覆盖）
    - user_id / dept_id / role: 从 JWT 注入的当前用户身份，供权限过滤
    - thread_id: LangGraph checkpoint 的 configurable.thread_id
    - image_id: 本次消息携带的图片 id（可选），决定入口是否先经 vision 节点
    - vision_desc: vision 节点产出的图片文字描述（供检索/生成共用）
    - retrieved_chunks: retrieve 节点的召回结果（短编号片段列表）
    - guard_warning: output_guard 产生的净化提示（SSE warning 帧用）
    """

    messages: Annotated[list[BaseMessage], operator.add]
    user_id: str
    dept_id: str | None
    role: str
    thread_id: str
    image_id: str | None
    vision_desc: str | None
    retrieved_chunks: list[dict]
    guard_warning: str | None
