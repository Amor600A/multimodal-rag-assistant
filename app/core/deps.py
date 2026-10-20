# -*- coding: utf-8 -*-
"""依赖注入：FastAPI Depends 使用。

核心：get_current_user 实时查库组装 CurrentUser，JWT 不缓存 dept_id
（用户调岗后旧 token 不会按旧部门越权，见方案 4.1）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_token
from app.infra.org.mock_org import get_department_repository
from app.infra.user.mock_user import User, get_user_repository

logger = logging.getLogger("app.deps")

security = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    """依赖注入产物：JWT 字段 + 实时查库补全 dept。"""
    user_id: str
    username: str
    role: str          # user | admin
    dept_id: Optional[str]  # 实时查库，调岗立即生效

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> CurrentUser:
    """从 Authorization: Bearer <JWT> 解析并校验，实时查库补 dept。

    SSE 流中途过期处理（方案 4.1）：仅请求进入时校验一次，astream 过程
    中不重复校验、不主动断流。401 由前端统一走 refresh 重试逻辑。
    """
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供认证")

    token = credentials.credentials
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token 已过期")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"token 无效: {e}")

    user_id = payload.get("sub")
    role = payload.get("role", "user")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token 缺少 sub")

    # 实时查库：用户存在 + is_active + dept_id（不来自 JWT）
    user_repo = get_user_repository()
    user: Optional[User] = await user_repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已停用")

    return CurrentUser(
        user_id=user.id,
        username=user.username,
        role=role,
        dept_id=user.dept_id,
    )
