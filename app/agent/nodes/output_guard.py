# -*- coding: utf-8 -*-
"""output_guard 节点：引用一致性校验 + 净化。

校验逻辑（方案 6.4）：
1. 提取回复末尾 `来源: [1] [3] ...` 行
2. 编号必须在本次召回集合内，否则移除（净化）
3. 移除后为空 → 整块删除
4. 格式缺失 → 视为无引用，跳过不阻断
5. 任何异常 → 降级跳过 + 审计，绝不把流变成 error
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage

from app.agent.state import AgentState

logger = logging.getLogger("app.agent.nodes.output_guard")

_CITATION_RE = re.compile(r"来源:\s*(\[[^\]]*\](?:\s*\[[^\]]*\])*)\s*$", re.MULTILINE)
_REF_RE = re.compile(r"\[(\d+)\]")


async def output_guard_node(state: AgentState) -> dict:
    """输出护栏：净化伪造引用编号，产生 warning。"""
    chunks = state.get("retrieved_chunks", [])
    valid_indexes = {c["index"] for c in chunks}

    # 取最后一条 AIMessage
    last_ai = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    if last_ai is None:
        return {}

    content = last_ai.content
    warning: str | None = None

    try:
        match = _CITATION_RE.search(content)
        if not match:
            # 格式缺失：视为无引用，跳过
            logger.info("output_guard: 无来源行，跳过校验（citation_format_miss）")
            return {}

        citation_block = match.group(1)
        cited = [int(n) for n in _REF_RE.findall(citation_block)]
        invalid = [n for n in cited if n not in valid_indexes]

        if invalid:
            # 净化：移除伪造编号
            kept = [n for n in cited if n in valid_indexes]
            if kept:
                new_block = "来源: " + " ".join(f"[{n}]" for n in kept)
            else:
                new_block = ""  # 整块删除
            content = _CITATION_RE.sub(new_block, content).rstrip() + "\n" if new_block else _CITATION_RE.sub("", content).rstrip() + "\n"
            warning = f"部分来源标注不可信已移除：编号 {invalid}"
            logger.info("output_guard: 净化引用 %s，保留 %s", invalid, kept)

        return {"messages": [AIMessage(content=content)], "guard_warning": warning}

    except Exception as e:
        # 护栏异常降级：跳过 + 审计，不阻断
        logger.warning("output_guard 异常，降级跳过: %s", e)
        return {}
