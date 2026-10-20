# -*- coding: utf-8 -*-
"""图片上传与代理访问路由（方案 7.1）。

存储：uploads/{user_id}/{image_id}.{ext}，按用户分目录物理隔离。
访问：唯一出口为本代理端点，uploads/ 不挂静态目录。
权限：仅本人或 admin 可访问，否则返回 404（不暴露资源存在性）。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import jwt
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.deps import CurrentUser, get_current_user
from app.core.security import decode_token
from app.infra.images import ImageRecord, get_image_repository
from app.infra.user.mock_user import get_user_repository

logger = logging.getLogger("app.api.images")

router = APIRouter(prefix="/knowledge/images", tags=["images"])

# MIME 白名单 → 扩展名映射
_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
}
_MAX_SIZE = 10 * 1024 * 1024  # 10MB
_UPLOADS_ROOT = Path(__file__).resolve().parents[3] / "uploads"

_bearer = HTTPBearer(auto_error=False)


async def _user_from_token(token: str) -> CurrentUser | None:
    """解析 token 并实时查库组装 CurrentUser（dept_id 不来自 JWT）。"""
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    user = await get_user_repository().get_by_id(user_id)
    if user is None or not user.is_active:
        return None
    return CurrentUser(
        user_id=user.id,
        username=user.username,
        role=payload.get("role", "user"),
        dept_id=user.dept_id,
    )


async def _resolve_access_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CurrentUser | None:
    """图片访问专用鉴权：Authorization 头优先，其次 ?token=（仅本端点支持，供 <img> 标签）。"""
    if creds is not None:
        u = await _user_from_token(creds.credentials)
        if u is not None:
            return u
    token = request.query_params.get("token")
    if token:
        return await _user_from_token(token)
    return None


@router.post("", status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    current: CurrentUser = Depends(get_current_user),
) -> dict:
    """上传图片（MIME 白名单 + 10MB 上限），返回 image_id 与代理 URL。"""
    mime = (file.content_type or "").lower()
    if mime not in _MIME_TO_EXT:
        raise HTTPException(status_code=415, detail="仅支持 png/jpg/jpeg/webp")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(content) > _MAX_SIZE:
        raise HTTPException(status_code=413, detail="图片大小不能超过 10MB")

    ext = _MIME_TO_EXT[mime]
    image_id = str(uuid.uuid4())
    user_dir = _UPLOADS_ROOT / current.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    abs_path = user_dir / f"{image_id}.{ext}"
    abs_path.write_bytes(content)

    rel_path = f"{current.user_id}/{image_id}.{ext}"
    rec = ImageRecord(
        image_id=image_id,
        user_id=current.user_id,
        mime=mime,
        file_path=rel_path,
    )
    await get_image_repository().create(rec)
    logger.info("图片上传: image=%s user=%s size=%d", image_id, current.user_id, len(content))
    return {
        "image_id": image_id,
        "url": f"/api/knowledge/images/{image_id}",
    }


@router.get("/{image_id}")
async def get_image(
    image_id: str,
    current: CurrentUser | None = Depends(_resolve_access_user),
) -> FileResponse:
    """图片代理访问（权限校验：仅本人/admin，否则 404）。"""
    if current is None:
        raise HTTPException(status_code=404, detail="未找到")

    rec = await get_image_repository().get(image_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="未找到")

    # 权限校验：本人或 admin
    if rec.user_id != current.user_id and current.role != "admin":
        raise HTTPException(status_code=404, detail="未找到")

    abs_path = _UPLOADS_ROOT / rec.file_path
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="未找到")

    return FileResponse(str(abs_path), media_type=rec.mime)
