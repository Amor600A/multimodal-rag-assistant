# -*- coding: utf-8 -*-
"""JWT 与密码哈希工具。

bcrypt 哈希统一在此实现，auth_service 与 seed_data 共用，
保证种子密码与登录验证使用同一算法（见方案 4.1 JWT 设计）。
"""

import uuid
from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.core.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """bcrypt 哈希密码。"""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码与哈希是否匹配。"""
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


# ---- JWT ----

def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def create_access_token(user_id: str, role: str) -> str:
    """签发 access token（120 分钟，含 sub + role + exp）。"""
    s = get_settings()
    payload = {
        "sub": user_id,
        "role": role,
        "exp": _now_ts() + s.jwt_expire_minutes * 60,
        "iat": _now_ts(),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def create_refresh_token(user_id: str, role: str) -> tuple[str, str, int]:
    """签发 refresh token，返回 (token, jti, exp_ts)。

    jti 是 refresh token 的唯一标识，登出时存 TokenStore 用于吊销；
    exp_ts 同时写入 TokenStore 作为 TTL。
    """
    s = get_settings()
    jti = uuid.uuid4().hex
    exp_ts = _now_ts() + s.jwt_refresh_expire_days * 86400
    payload = {
        "sub": user_id,
        "role": role,
        "type": "refresh",
        "jti": jti,
        "exp": exp_ts,
        "iat": _now_ts(),
    }
    token = jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)
    return token, jti, exp_ts


def decode_token(token: str) -> dict:
    """解析 JWT，签名/过期校验失败抛 jwt.PyJWTError。

    仅解析，不校验 jti 是否被吊销——吊销由 TokenStore 在调用方单独检查。
    """
    s = get_settings()
    return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
