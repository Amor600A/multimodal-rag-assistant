# -*- coding: utf-8 -*-
"""knowledge 路由：文档入库（打 visibility 权限标）。

权限打标规则（方案 6.3 默认拒绝）：
- admin：可选 public / dept / private
- 普通用户：强制 private（防越权上传到 public/dept）
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, get_current_user
from app.services.rag_service import get_rag_service

logger = logging.getLogger("app.api.knowledge")

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeIngestReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    visibility: str = Field("private", pattern="^(public|dept|private)$")
    dept_id: str | None = None
    title: str | None = None


@router.post("", status_code=201)
async def ingest_knowledge(
    req: KnowledgeIngestReq,
    current: CurrentUser = Depends(get_current_user),
) -> dict:
    """文档入库。自动打 owner_user_id / dept_id 标。"""
    visibility = req.visibility
    owner_user_id = current.user_id
    dept_id = req.dept_id

    # 权限打标：普通用户强制 private
    if current.role != "admin":
        if visibility != "private":
            raise HTTPException(status_code=403, detail="非 admin 用户仅可上传 private 文档")
        dept_id = None
    else:
        # admin 上传 dept 文档需要指定 dept_id
        if visibility == "dept" and not dept_id:
            raise HTTPException(status_code=400, detail="dept 文档需指定 dept_id")

    doc_id = f"doc-{uuid.uuid4().hex}"
    record = {
        "id": doc_id,
        "text": req.text,
        "scope": "personal" if visibility == "private" else "domain",
        "visibility": visibility,
        "owner_user_id": owner_user_id if visibility == "private" else None,
        "dept_id": dept_id if visibility == "dept" else None,
        "source": "upload",
    }
    get_rag_service().ingest([record])
    logger.info("知识入库: doc=%s vis=%s user=%s", doc_id, visibility, current.user_id)
    return {"id": doc_id, "visibility": visibility}
