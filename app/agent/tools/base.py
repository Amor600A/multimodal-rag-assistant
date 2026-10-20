# -*- coding: utf-8 -*-
"""Tool 抽象接口。阶段 4 接 builtin/custom 工具时按此协议实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    """Agent 可调用工具的最小协议。

    后续接 LangGraph ToolNode 时，这里的 Tool 会包装成 langchain_core.tools.BaseTool。
    当前阶段只定义接口，不实现任何具体工具。
    """

    name: str
    description: str

    @abstractmethod
    async def run(self, **kwargs) -> str:
        """执行工具，返回文本结果（后续可扩展为 ToolMessage）。"""
        raise NotImplementedError
