# -*- coding: utf-8 -*-
"""generate 节点：片段注入 Prompt + 偏好注入 + 末尾来源引用块。

mock LLM 模式下（LLM_BACKEND=mock）：pick_reply 拿基础回复，命中召回片段则
追加 `来源: [1]`。
真实 LLM 模式下（LLM_BACKEND=tongyi）：把片段注入 system prompt，模型生成
带引用的回复，output_guard 节点负责校验引用一致性。
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.mock_llm import pick_reply
from app.agent.state import AgentState
from app.core.config import get_settings
from app.rag.llm import get_llm
from app.services.memory_service import get_memory_service

logger = logging.getLogger("app.agent.nodes.generate")

# System Prompt 模板：召回片段 + 分数语义说明 + 输出规范
_SCORE_HINT_RERANK = "【关于召回分数】片段后面的分数是 Rerank 精排分数（relevance 0~5），越高表示与问题越相关。"
_SCORE_HINT_COSINE = "【关于召回分数】片段后面的分数是向量检索 COSINE 相似度（0~1），越高表示语义越接近。"

_SYSTEM_TEMPLATE = """你是多模态 RAG 助手，回答基于以下召回的知识片段。

    {score_hint}

    【召回片段】
    {chunks_block}

    【要求】
    1. 优先参考片段内容回答，正文引用片段时用 [编号] 标注
    2. 回答末尾单独一行，格式固定为「来源: [1] [2]」，列出回答实际用到的片段编号
    3. 片段不足以回答时可基于常识补充，但请说明
    {preference_block}"""


def _build_system_prompt(chunks: list[dict], reply_preference: str, rerank_on: bool) -> str:
    """把召回片段 + 用户偏好 + 分数语义说明组装成 system prompt。"""
    lines: list[str] = []
    for c in chunks:
        idx = c.get("index", "?")
        source = c.get("source", "-")
        ftype = c.get("file_type") or "text"
        score = c.get("score", 0.0)
        text = c.get("text", "")
        # Rerank 分数一般 0~5 或 0~100，COSINE 0~1，统一打印到两位小数
        lines.append(f"[{idx}]  (来源={source}, 类型={ftype}, 分数={score:.2f})\n{text}")
    chunks_block = "\n\n".join(lines) if lines else "（无召回片段）"
    pref = f"\n4. 回复风格偏好：{reply_preference}" if reply_preference else ""
    score_hint = _SCORE_HINT_RERANK if rerank_on else _SCORE_HINT_COSINE
    return _SYSTEM_TEMPLATE.format(
        score_hint=score_hint, chunks_block=chunks_block, preference_block=pref,
    )


async def generate_node(state: AgentState) -> dict:
    """生成节点：基于召回片段 + 用户偏好生成回复。"""
    user_text = ""
    user_id = state.get("user_id", "")
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content)
            break

    # 图片理解注入（vision 节点产出，方案 7）：与用户问题拼接成检索 query；
    # 生成侧把它作为额外上下文加入 system prompt
    vision_desc = state.get("vision_desc")
    query = user_text
    if vision_desc:
        query = f"{user_text}\n\n（图片内容：{vision_desc}）"

    # 用户偏好（风格注入）
    pref = get_memory_service().get_reply_preference(user_id)

    # 召回片段（按 index 排序，确保编号顺序）
    chunks = sorted(state.get("retrieved_chunks", []), key=lambda c: c.get("index", 0))

    llm = get_llm()
    if llm is None:
        # ---- mock 路径：关键词匹配兜底 ----
        reply = pick_reply(user_text)
        if pref:
            reply = f"[按您的偏好：{pref}] {reply}"
        if vision_desc:
            reply = f"[图片内容：{vision_desc}] {reply}"
        if chunks:
            reply = f"{reply}\n\n来源: [1]"
        else:
            reply = f"{reply}\n\n来源: 无"
        return {"messages": [AIMessage(content=reply)]}

    # ---- 真实 LLM 路径：RAG prompt 生成 ----
    rerank_on = get_settings().rerank_enabled
    system = SystemMessage(content=_build_system_prompt(chunks, pref, rerank_on))
    human = HumanMessage(content=query)
    logger.info("generate: calling real LLM (backend=%s, chunks=%d)", type(llm).__name__, len(chunks))
    resp = await llm.ainvoke([system, human])
    content = resp.content if isinstance(resp, AIMessage) else str(resp)
    return {"messages": [AIMessage(content=content)]}
