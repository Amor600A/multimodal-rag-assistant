# -*- coding: utf-8 -*-
"""会话路由：列表 / 新建 / 删除（按用户隔离，仅本人可删）。"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.deps import CurrentUser, get_current_user
from app.infra.session.mock_session import get_session_repository

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionReq(BaseModel):
    thread_id: str | None = None     # 客户端可指定；不指定则服务端生成
    title: str | None = None         # 前端首句截断（见方案 9.2），不指定则用 thread_id 前缀


@router.get("")
async def list_sessions(current: CurrentUser = Depends(get_current_user)) -> list[dict]:
    repo = get_session_repository()
    sessions = await repo.list_by_user(current.user_id)
    return [
        {"thread_id": s.thread_id, "title": s.title, "created_at": s.created_at}
        for s in sessions
    ]


@router.post("", status_code=201)
async def create_session(req: CreateSessionReq, current: CurrentUser = Depends(get_current_user)) -> dict:
    thread_id = req.thread_id or uuid.uuid4().hex
    title = req.title or thread_id[:8]
    repo = get_session_repository()
    session = await repo.create(thread_id=thread_id, user_id=current.user_id, title=title)
    return {
        "thread_id": session.thread_id,
        "title": session.title,
        "created_at": session.created_at,
    }


@router.delete("/{thread_id}", status_code=204)
async def delete_session(thread_id: str, current: CurrentUser = Depends(get_current_user)) -> None:
    repo = get_session_repository()
    session = await repo.get(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session.user_id != current.user_id:
        # 跨用户删除：返回 403（方案 12.2 权限矩阵回归点）
        raise HTTPException(status_code=403, detail="无权删除此会话")
    await repo.delete(thread_id)
    # TODO: 后续阶段同步清理 checkpointer 中该 thread 的 checkpoint（langgraph 无原生删除，可忽略或加自定义表）
