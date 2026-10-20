# -*- coding: utf-8 -*-
"""Agent 图构建 + 编译单例。

链路：START →（含 image_id 时先经 vision）→ retrieve → generate → output_guard → END
阶段 7+ 接 ReAct 工具调用后，在此基础上加条件边，对外接口不变。
"""

from __future__ import annotations

import logging
import warnings

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.generate_node import generate_node
from app.agent.nodes.output_guard import output_guard_node
from app.agent.nodes.retrieve_node import retrieve_node
from app.agent.nodes.vision_node import vision_node
from app.agent.state import AgentState
from app.infra.checkpointer.memory_checkpointer import get_checkpointer

warnings.filterwarnings("ignore", message="LangChainPendingDeprecationWarning")

logger = logging.getLogger("app.agent.graph")

_compiled = None


def _route_entry(state: AgentState) -> str:
    """入口路由：消息携带 image_id 时先走 vision，否则直接 retrieve。"""
    return "vision" if state.get("image_id") else "retrieve"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("vision", vision_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("output_guard", output_guard_node)
    g.add_conditional_edges(
        START,
        _route_entry,
        {"vision": "vision", "retrieve": "retrieve"},
    )
    g.add_edge("vision", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "output_guard")
    g.add_edge("output_guard", END)
    return g


def get_compiled_graph():
    global _compiled
    if _compiled is None:
        graph = build_graph()
        _compiled = graph.compile(checkpointer=get_checkpointer())
        logger.info("Agent 图已编译（nodes=[vision, retrieve, generate, output_guard]）")
    return _compiled
