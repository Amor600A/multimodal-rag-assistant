# -*- coding: utf-8 -*-
"""auth 路由：register / login / refresh / logout / me。

限流：login / register / refresh 按 IP 限 5/min（防暴力破解）。
审计：登录成功/失败/登出/刷新记审计（见方案 13.2）。
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.core.deps import CurrentUser, get_current_user
from app.core.logger import audit
from app.core.ratelimit import _check_rate_limit
from app.infra.org.mock_org import get_department_repository
from app.infra.user.mock_user import get_user_repository
from app.services.auth_service import AuthError, AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("app.api.auth")


# ---- 请求体 ----

class RegisterReq(BaseModel):
    username: str
    password: str
    role: str = "user"


class LoginReq(BaseModel):
    username: str
    password: str


class RefreshReq(BaseModel):
    refresh_token: str


class LogoutReq(BaseModel):
    refresh_token: Optional[str] = None


# ---- 限流 + IP 提取（限流实现已迁移至 app.core.ratelimit）----

def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _auth() -> AuthService:
    return AuthService()


# ---- 端点 ----

@router.post("/register", status_code=201)
async def register(req: RegisterReq, request: Request) -> dict:
    _check_rate_limit(_get_client_ip(request))
    auth = _auth()
    try:
        await auth.register(req.username, req.password, req.role)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    audit("register", username=req.username, detail=f"role={req.role}", ip=_get_client_ip(request))
    return {"ok": True}


@router.post("/login")
async def login(req: LoginReq, request: Request) -> dict:
    ip = _get_client_ip(request)
    _check_rate_limit(ip)
    auth = _auth()
    try:
        result = await auth.login(req.username, req.password)
    except AuthError as e:
        audit("login_failed", username=req.username, ip=ip, detail=str(e))
        raise HTTPException(status_code=401, detail=str(e))
    # 登录成功：查询 dept 信息一起返回（前端展示用）
    user = await get_user_repository().get_by_username(req.username)
    dept_repo = get_department_repository()
    dept_name = None
    if user and user.dept_id:
        dept = await dept_repo.get_by_id(user.dept_id)
        if dept:
            dept_name = dept.name
    audit("login_success", username=req.username, ip=ip, detail=f"role={user.role if user else '?'}")
    return {
        "access_token": result.access_token,
        "refresh_token": result.refresh_token,
        "token_type": result.token_type,
        "user": {
            "id": user.id if user else None,
            "username": req.username,
            "role": user.role if user else None,
            "dept_id": user.dept_id if user else None,
            "dept_name": dept_name,
        },
    }


@router.post("/refresh")
async def refresh(req: RefreshReq, request: Request) -> dict:
    _check_rate_limit(_get_client_ip(request))
    auth = _auth()
    try:
        result = await auth.refresh(req.refresh_token)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    audit("refresh", detail="rotation", ip=_get_client_ip(request))
    return {
        "access_token": result.access_token,
        "refresh_token": result.refresh_token,
        "token_type": result.token_type,
    }


@router.post("/logout")
async def logout(request: Request, current: CurrentUser = Depends(get_current_user), req: LogoutReq = LogoutReq()) -> dict:
    auth = _auth()
    await auth.logout(req.refresh_token)
    audit("logout", username=current.username, ip=_get_client_ip(request))
    return {"ok": True}


@router.get("/me")
async def me(current: CurrentUser = Depends(get_current_user)) -> dict:
    """当前用户信息（实时查库补 dept_id / dept_name）。"""
    dept_repo = get_department_repository()
    dept_name = None
    if current.dept_id:
        d = await dept_repo.get_by_id(current.dept_id)
        if d:
            dept_name = d.name
    return {
        "user_id": current.user_id,
        "username": current.username,
        "role": current.role,
        "dept_id": current.dept_id,
        "dept_name": dept_name,
    }
