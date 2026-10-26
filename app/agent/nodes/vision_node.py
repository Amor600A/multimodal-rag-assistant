# -*- coding: utf-8 -*-
"""vision 节点：消息携带图片时先做多模态理解，产出图片描述写入 state.vision_desc。

链路位置（方案 4.3 图结构）：START →（含 image_id 时）vision → retrieve → generate →
output_guard → END。不含图片时入口直接走 retrieve，本节点不参与。

产出用途（方案 7 多模态设计）：图片描述与用户文本一起进入检索与生成。
三种后端取鉴权方式：
- mock: 返回固定描述文案，保证链路可测
- tongyi: 走 DashScope 原生多模态 generation API（qwen-vl-plus 在此端点可用）
- openai_compatible: 走 AsyncOpenAI Chat Completions
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from app.agent.state import AgentState
from app.core.config import get_settings
from app.infra.images import get_image_repository

logger = logging.getLogger("app.agent.nodes.vision")

_UPLOADS_ROOT = Path(__file__).resolve().parents[3] / "uploads"

# Mock 模式固定描述（方案 7：mock 下图片理解返回固定文案）
_MOCK_DESC = "一张含表格与流程示意的工作文档截图（Mock 模式固定描述）。"

_VISION_PROMPT = "请用一段话描述这张图片的关键信息，用于企业内部知识检索。"

# DashScope 原生多模态 generation 端点
_DASHSCOPE_VISION_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


async def vision_node(state: AgentState) -> dict:
    """图片理解节点：校验图片归属后生成描述，写入 vision_desc。"""
    image_id = state.get("image_id")
    if not image_id:
        return {}

    rec = await get_image_repository().get(image_id)
    if rec is None:
        logger.warning("vision: 图片不存在 image=%s", image_id)
        return {}
    if rec.user_id != state.get("user_id", ""):
        logger.warning("vision: 图片归属不符 image=%s user=%s", image_id, state.get("user_id"))
        return {}

    desc = await _describe_image(rec.file_path, rec.mime)
    logger.info("vision: image=%s desc_len=%d", image_id, len(desc))
    return {"vision_desc": desc}


async def _describe_image(file_path: str, mime: str) -> str:
    """生成图片描述：按 llm_backend 分支调用对应的多模态接口。"""
    settings = get_settings()
    if settings.llm_backend == "mock":
        return _MOCK_DESC

    abs_path = _UPLOADS_ROOT / file_path
    b64 = base64.b64encode(abs_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    if settings.llm_backend == "tongyi":
        return await _describe_via_dashscope(data_url, settings)

    # openai_compatible 或其他：走 AsyncOpenAI Chat Completions
    if not settings.llm_base_url or not settings.llm_api_key:
        logger.warning(
            "vision: 后端 %s 缺 llm_base_url/llm_api_key，返回 mock 描述",
            settings.llm_backend,
        )
        return _MOCK_DESC
    return await _describe_via_openai(data_url, mime, settings)


async def _describe_via_dashscope(data_url: str, settings) -> str:
    """DashScope 原生多模态 generation API（tongyi 模式下 vision 用）。

    qwen-vl-plus 在 DashScope OpenAI 兼容端点不可用（兼容端点只覆盖文本模型），
    必须走原生 multimodal-generation 端点。
    """
    if not settings.embedding_api_key:
        logger.warning("vision: tongyi 模式缺 embedding_api_key，返回 mock 描述")
        return _MOCK_DESC

    import httpx

    body = {
        "model": settings.llm_vision_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": data_url},
                        {"text": _VISION_PROMPT},
                    ],
                }
            ]
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _DASHSCOPE_VISION_URL,
            headers={
                "Authorization": f"Bearer {settings.embedding_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
    if resp.status_code != 200:
        logger.warning(
            "vision: DashScope 多模态 API 失败 status=%d body=%s",
            resp.status_code, resp.text[:200],
        )
        return _MOCK_DESC
    try:
        data = resp.json()
        # DashScope 返回结构：output.choices[0].message.content 是 list，取 text 类型项
        choices = (data.get("output") or {}).get("choices") or []
        content = choices[0].get("message", {}).get("content", []) if choices else []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                return str(item["text"]).strip()
    except Exception as exc:
        logger.warning("vision: DashScope 响应解析失败 %s", exc)
    return _MOCK_DESC


async def _describe_via_openai(data_url: str, mime: str, settings) -> str:
    """openai_compatible 后端：AsyncOpenAI Chat Completions。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    resp = await client.chat.completions.create(
        model=settings.llm_vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()
