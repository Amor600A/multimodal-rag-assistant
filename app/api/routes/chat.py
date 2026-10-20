# -*- coding: utf-8 -*-
"""Chat SSE 路由。

核心职责：
1. 鉴权（复用 get_current_user）
2. 单飞锁：同一用户同一会话同一时刻只允许一个请求
3. 调 LangGraph astream(stream_mode=["messages", "updates"]) 并转 SSE 推送
4. 审计 + 限流（复用 auth.py 的模式）
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_compiled_graph
from app.core.deps import CurrentUser, get_current_user
from app.core.logger import audit
from app.core.ratelimit import check_rate_limit
from app.infra.images import get_image_repository
from app.infra.session.mock_session import get_session_repository

logger = logging.getLogger("app.api.chat")

router = APIRouter()


class ChatReq(BaseModel):
    """聊天请求体。"""

    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(None, description="会话 thread_id，不传时自动创建默认")
    image_id: str | None = Field(
        None,
        description="已上传图片的 image_id（可选；实际携带并成功入会话时置 images.bound=TRUE）",
    )


# ---------- 单飞锁：per (user_id, thread_id) ----------
# 同一用户同一会话在处理请求时，后续请求排队等待，避免 checkpoint 并发写冲突
_locks: dict[tuple[str, str], asyncio.Lock] = {}
_locks_cleanup_counter: int = 0
_LOCK_CLEANUP_EVERY = 100  # 每 100 次 acquire 做一次空锁清理


def _get_or_create_lock(user_id: str, thread_id: str) -> asyncio.Lock:
    global _locks_cleanup_counter
    key = (user_id, thread_id)
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    # 轻量清理：每 100 次 acquire 扫一遍，删掉已被垃圾回收但 key 还在的条目
    _locks_cleanup_counter += 1
    if _locks_cleanup_counter >= _LOCK_CLEANUP_EVERY:
        _locks_cleanup_counter = 0
        stale = [k for k, l in _locks.items() if not isinstance(l, asyncio.Lock)]
        for k in stale:
            _locks.pop(k, None)
    return lock


def _release_lock(user_id: str, thread_id: str) -> None:
    """释放锁并清理空条目。"""
    key = (user_id, thread_id)
    lock = _locks.get(key)
    if lock is not None and not lock.locked():
        # 锁已释放过，删除 key（下次请求重建）
        _locks.pop(key, None)


async def _check_thread_ownership(user_id: str, thread_id: str) -> bool:
    """校验 thread_id 归属：用户只能访问自己的会话。"""
    sessions = get_session_repository()
    s = await sessions.get(thread_id)
    if s is None:
        return False
    return s.user_id == user_id


async def _mark_image_bound(image_id: str, user_id: str) -> None:
    """消息实际携带图片并成功入会话后，置 images.bound=TRUE（方案 7.1 孤儿清理判定依据）。

    仅当图片存在且归属当前用户时才置位，避免越权标记他人图片。
    """
    repo = get_image_repository()
    rec = await repo.get(image_id)
    if rec is None or rec.user_id != user_id:
        return
    await repo.mark_bound(image_id)


# ---------- SSE 事件构建 ----------

def _sse_token(content: str) -> dict[str, Any]:
    return {"event": "message", "data": json.dumps({"type": "token", "content": content}, ensure_ascii=False)}


def _sse_node(node_name: str) -> dict[str, Any]:
    return {"event": "node", "data": json.dumps({"type": "node", "node": node_name}, ensure_ascii=False)}


def _sse_done(thread_id: str) -> dict[str, Any]:
    return {"event": "done", "data": json.dumps({"type": "done", "thread_id": thread_id}, ensure_ascii=False)}


def _sse_error(msg: str) -> dict[str, Any]:
    return {"event": "error", "data": json.dumps({"type": "error", "detail": msg}, ensure_ascii=False)}


# ---------- 主路由 ----------

@router.post("/chat")
async def chat(
    req: ChatReq,
    request: Request,
    current: CurrentUser = Depends(get_current_user),
) -> EventSourceResponse:
    # 限流：按 IP 5/min（和 auth 路由同口径）
    check_rate_limit(f"chat:ip={request.client.host or 'unknown'}", limit=5, window=60)

    # 解析 thread_id：有 session 归属校验，无则自动创建默认会话
    thread_id = req.thread_id
    sessions = get_session_repository()
    if thread_id:
        if not await _check_thread_ownership(current.user_id, thread_id):
            # 跨用户或不存在，返回 403 格式 SSE 错误
            async def _err():
                yield _sse_error("会话不存在或无权访问")
            return EventSourceResponse(_err())
    else:
        # 自动创建一个默认会话（供首次对话）
        thread_id = f"auto-{current.user_id}-{int(time.time())}"
        await sessions.create(
            thread_id=thread_id,
            user_id=current.user_id,
            title="新会话",
        )

    # 单飞锁
    lock = _get_or_create_lock(current.user_id, thread_id)

    async def _stream():
        try:
            await lock.acquire()

            from langchain_core.messages import HumanMessage

            graph = get_compiled_graph()
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": current.user_id,
                    "dept_id": current.dept_id,
                    "role": current.role,
                },
                "recursion_limit": 25,
            }

            # 构造 LangGraph 输入 state
            # image_id / vision_desc 为逐轮字段：显式置空避免上一轮 checkpoint 里的值残留
            state_input = {
                "messages": [HumanMessage(content=req.message)],
                "user_id": current.user_id,
                "dept_id": current.dept_id,
                "role": current.role,
                "thread_id": thread_id,
                "image_id": req.image_id,
                "vision_desc": None,
                "metadata": {},
            }

            # astream：updates 流 = 节点转换 + 状态更新；messages 流 = 节点产出的消息
            # LangGraph 0.2.x list 形式 stream_mode 产出 tuple('mode', value)
            final_content: str | None = None
            guard_warning: str | None = None

            async for step in graph.astream(
                state_input,
                config,
                stream_mode=["messages", "updates"],
            ):
                if isinstance(step, tuple) and len(step) == 2:
                    mode, value = step
                else:
                    continue
                if mode == "updates":
                    # value = {"node_name": {state_field: value}}
                    for node_name, update in value.items():
                        yield _sse_node(node_name)
                        # output_guard 产出 guard_warning
                        if node_name == "output_guard" and isinstance(update, dict):
                            gw = update.get("guard_warning")
                            if gw:
                                guard_warning = gw
                elif mode == "messages":
                    # value = (AIMessage, metadata_dict)
                    full_msg, _meta = value
                    content = getattr(full_msg, "content", None)
                    if content:
                        # 只保留最后一条 AI 消息（output_guard 净化后的最终版）
                        final_content = str(content)

            # 图片引用置位：消息实际携带 image_id 且已成功入会话，置 images.bound=TRUE
            if req.image_id:
                await _mark_image_bound(req.image_id, current.user_id)

            # 输出最终回复（逐字符模拟 token 流）
            if final_content:
                for ch in final_content:
                    yield _sse_token(ch)

            # output_guard 净化提示
            if guard_warning:
                yield {"event": "warning", "data": json.dumps({"type": "warning", "detail": guard_warning}, ensure_ascii=False)}

            yield _sse_done(thread_id)

            # 长期记忆提取（异步，不阻塞 SSE 响应；提取失败只记日志）
            try:
                from app.services.memory_service import get_memory_service

                asyncio.create_task(
                    get_memory_service().extract_and_store(current.user_id, req.message)
                )
            except Exception as e:
                logger.warning("记忆提取任务启动失败: %s", e)

        except asyncio.CancelledError:
            logger.info("SSE 连接被客户端取消（thread_id=%s）", thread_id)
            yield _sse_error("连接已取消")
        except Exception as e:
            logger.exception("chat SSE 异常（thread_id=%s）", thread_id)
            yield _sse_error(str(e))
        finally:
            lock.release()
            _release_lock(current.user_id, thread_id)

    # 审计（异步，不阻塞 SSE 首块）
    from app.core.logger import audit

    audit(
        "chat_start",
        username=current.username,
        detail=f"thread_id={thread_id} len={len(req.message)}",
    )

    return EventSourceResponse(_stream())
