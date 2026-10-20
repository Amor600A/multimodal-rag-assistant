# -*- coding: utf-8 -*-
"""vision 节点：消息携带图片时先做多模态理解，产出图片描述写入 state.vision_desc。

链路位置（方案 4.3 图结构）：START →（含 image_id 时）vision → retrieve → generate →
output_guard → END。不含图片时入口直接走 retrieve，本节点不参与。

产出用途（方案 7 多模态设计）：图片描述与用户文本一起进入检索与生成。
Mock 模式（LLM_BACKEND=mock）：返回固定描述文案，保证链路可测。
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
    """生成图片描述：mock 模式返回固定文案，真实模式调多模态 LLM。"""
    settings = get_settings()
    if settings.llm_backend == "mock":
        return _MOCK_DESC

    from openai import AsyncOpenAI

    abs_path = _UPLOADS_ROOT / file_path
    b64 = base64.b64encode(abs_path.read_bytes()).decode("ascii")
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    resp = await client.chat.completions.create(
        model=settings.llm_vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()
