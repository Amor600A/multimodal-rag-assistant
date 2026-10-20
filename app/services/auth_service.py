# -*- coding: utf-8 -*-
"""auth_service：注册 / 登录 / 双 token 刷新轮换 / 登出吊销。

依赖 MockUserRepository 与 MockTokenStore（单例），切真实后端时 factory
分支即可，调用方代码零改动。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import jwt

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.infra.user.mock_user import MockUserRepository, get_user_repository
from app.infra.user.token_store import MockTokenStore, get_token_store

logger = logging.getLogger("app.services.auth")


class AuthError(Exception):
    """业务层错误：路由应转换为对应 HTTPException。"""


@dataclass
class LoginResult:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AuthService:
    def __init__(
        self,
        user_repo: MockUserRepository | None = None,
        token_store: MockTokenStore | None = None,
    ) -> None:
        self._users = user_repo or get_user_repository()
        self._store = token_store or get_token_store()

    # -- 注册 --

    async def register(self, username: str, password: str, role: str = "user") -> None:
        # 用户名格式校验（简单规则：3-32 字母数字下划线）
        if not (3 <= len(username) <= 32 and username.replace("_", "").isalnum()):
            raise AuthError("用户名格式不合法（3-32 字母数字下划线）")
        if len(password) < 6:
            raise AuthError("密码至少 6 位")
        existing = await self._users.get_by_username(username)
        if existing is not None:
            raise AuthError("用户名已存在")
        await self._users.create(
            username=username,
            password_hash=hash_password(password),
            role=role,
        )

    # -- 登录 --

    async def login(self, username: str, password: str) -> LoginResult:
        user = await self._users.get_by_username(username)
        # 统一返回"用户名或密码错误"，不区分用户名不存在与密码错（防枚举）
        if user is None or not verify_password(password, user.password_hash):
            raise AuthError("用户名或密码错误")
        if not user.is_active:
            raise AuthError("账户已停用")

        access = create_access_token(user.id, user.role)
        refresh, jti, exp_ts = create_refresh_token(user.id, user.role)
        # login 时 refresh token 刚签发，不应吊销；让它保持有效等用户后续使用
        await self._users.update_last_login(user.id)
        return LoginResult(access_token=access, refresh_token=refresh)

    # -- 刷新（轮换：旧 refresh 作废，签新 refresh）--

    async def refresh(self, refresh_token: str) -> LoginResult:
        try:
            payload = decode_token(refresh_token)
        except jwt.ExpiredSignatureError:
            raise AuthError("refresh token 已过期")
        except jwt.PyJWTError as e:
            raise AuthError(f"refresh token 无效: {e}")

        if payload.get("type") != "refresh":
            raise AuthError("提供的不是 refresh token")

        jti = payload.get("jti")
        user_id = payload.get("sub")
        role = payload.get("role", "user")

        # 吊销校验：jti 已被登出/轮换
        if self._store.is_revoked(jti):
            raise AuthError("refresh token 已吊销")

        # 用户仍存在
        user = await self._users.get_by_id(user_id)
        if user is None or not user.is_active:
            raise AuthError("用户不存在或已停用")

        # 轮换：旧 refresh jti 作废（已用），签新 refresh 保持有效
        self._store.revoke(jti, int(payload["exp"]))
        new_access = create_access_token(user.id, role)
        new_refresh, new_jti, new_exp = create_refresh_token(user.id, role)
        return LoginResult(access_token=new_access, refresh_token=new_refresh)

    # -- 登出（吊销 refresh jti）--

    async def logout(self, refresh_token: str | None) -> None:
        if not refresh_token:
            return
        try:
            payload = decode_token(refresh_token)
        except Exception:
            return  # 已过期/无效则无需再吊销
        if payload.get("type") != "refresh":
            return
        jti = payload.get("jti")
        exp = int(payload.get("exp", 0))
        if jti and exp:
            self._store.revoke(jti, exp)
            logger.info("refresh jti=%s 已吊销", jti[:8])
