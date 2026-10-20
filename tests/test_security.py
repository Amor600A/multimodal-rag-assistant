# -*- coding: utf-8 -*-
"""test_security：JWT 签发/解析/过期 + bcrypt 哈希 + auth 全链路（单元 + 集成）。

单元部分直接调用 app.core.security，不经过 HTTP；
集成部分走真实路由（register / login / me / refresh 轮换 / logout 吊销），
复用 conftest 的 session 级 client（lifespan 注入种子数据）与登录辅助函数。

并发/限流说明：登录与刷新路由按 IP 限 5/min，conftest 的 autouse fixture
已在每个用例前清空计数桶，故单个用例内的调用次数控制在 5 次以内即可。
"""

from __future__ import annotations

import time

import jwt
import pytest

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from conftest import SEED_PASSWORDS, login

settings = get_settings()


# ============================================================
# 1. bcrypt 哈希
# ============================================================

def test_hash_is_salted_and_not_plaintext():
    """同一密码两次哈希不同（随机 salt），且哈希中不含明文。"""
    plain = "alice123"
    h1 = hash_password(plain)
    h2 = hash_password(plain)
    # 随机 salt：两次结果必然不同
    assert h1 != h2
    # 不存明文
    assert plain not in h1
    # bcrypt 特征前缀
    assert h1.startswith("$2")


def test_verify_password_correct_and_wrong():
    """verify 正确密码返回 True，错误密码返回 False。"""
    hashed = hash_password("secret-pass")
    assert verify_password("secret-pass", hashed) is True
    assert verify_password("wrong-pass", hashed) is False
    # 非法哈希不应抛异常，而是安全返回 False
    assert verify_password("secret-pass", "not-a-bcrypt-hash") is False


# ============================================================
# 2. JWT 签发与解析（单元）
# ============================================================

def test_access_token_roundtrip_sub_role():
    """create_access_token → decode_token 得到正确的 sub / role，且未过期。"""
    token = create_access_token("u-123", "admin")
    payload = decode_token(token)
    assert payload["sub"] == "u-123"
    assert payload["role"] == "admin"
    assert payload["exp"] > int(time.time())


def test_tampered_signature_rejected():
    """篡改签名后解析失败（签名校验不通过）。"""
    token = create_access_token("u-123", "user")
    header, body, sig = token.split(".")
    # 改动签名段首字符（base64url 有效字符），保证签名不再是原签名
    flipped = ("B" if sig[0] == "A" else "A") + sig[1:]
    tampered = f"{header}.{body}.{flipped}"
    with pytest.raises(jwt.PyJWTError):
        decode_token(tampered)


def test_wrong_secret_rejected():
    """用错误密钥签发的 token 解析失败。"""
    payload = {"sub": "u-123", "role": "user", "exp": int(time.time()) + 300}
    forged = jwt.encode(payload, "attacker-secret", algorithm=settings.jwt_algorithm)
    with pytest.raises(jwt.PyJWTError):
        decode_token(forged)


def test_expired_token_raises_expired_signature():
    """手工构造 exp 已过去的 token，解析抛 ExpiredSignatureError。"""
    payload = {
        "sub": "u-123",
        "role": "user",
        "exp": int(time.time()) - 10,
        "iat": int(time.time()) - 600,
    }
    expired = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token(expired)


def test_refresh_token_carries_jti_and_type():
    """create_refresh_token 返回 (token, jti, exp_ts)，解析含 type=refresh 与 jti。"""
    token, jti, exp_ts = create_refresh_token("u-123", "user")
    payload = decode_token(token)
    assert payload["type"] == "refresh"
    assert payload["jti"] == jti
    assert payload["sub"] == "u-123"
    assert payload["exp"] == exp_ts


# ============================================================
# 3. 全链路集成（走 HTTP）
# ============================================================

def test_register_login_me_flow(client):
    """register → login → GET /api/auth/me 全链路。"""
    username = "sec_new_user"
    # 注册（唯一用户名，返回 201）
    r = client.post("/api/auth/register", json={"username": username, "password": "pwd123456"})
    assert r.status_code == 201, r.text
    assert r.json()["ok"] is True

    # 登录拿到双 token
    body = login(client, username, "pwd123456")
    assert body["token_type"] == "bearer"
    access = body["access_token"]
    assert body["refresh_token"]

    # 用 access token 访问 me
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer " + access})
    assert r.status_code == 200, r.text
    me = r.json()
    assert me["username"] == username
    assert me["role"] == "user"


def test_refresh_rotation_invalidates_old_refresh(client):
    """refresh 轮换：新 access 可用、旧 refresh 作废、新 refresh 可用。"""
    first = login(client, "alice")
    old_refresh = first["refresh_token"]

    # 第一次刷新成功，返回新的 access + refresh
    r = client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
    assert r.status_code == 200, r.text
    rotated = r.json()
    new_access = rotated["access_token"]
    new_refresh = rotated["refresh_token"]
    # 新 refresh 的 jti 是 uuid，必然不同于旧 refresh
    # （access token 因 payload 含秒级 iat/exp，同一秒内签发可能字节相同，故不断言其不同）
    assert new_refresh != old_refresh

    # 新 access 可正常访问 me
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer " + new_access})
    assert r.status_code == 200, r.text
    assert r.json()["username"] == "alice"

    # 旧 refresh 已被轮换吊销，再次刷新返回 401
    r = client.post("/api/auth/refresh", json={"refresh_token": old_refresh})
    assert r.status_code == 401, r.text

    # 新 refresh 仍有效
    r = client.post("/api/auth/refresh", json={"refresh_token": new_refresh})
    assert r.status_code == 200, r.text


def test_logout_revokes_refresh(client):
    """logout 吊销 refresh 后，该 refresh 不能再用于刷新。"""
    body = login(client, "alice")
    access = body["access_token"]
    refresh = body["refresh_token"]

    # 登出需携带 access token，body 传 refresh token 用于吊销
    r = client.post(
        "/api/auth/logout",
        headers={"Authorization": "Bearer " + access},
        json={"refresh_token": refresh},
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    # 已吊销的 refresh 不能再刷新
    r = client.post("/api/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 401, r.text


def test_expired_refresh_rejected_by_api(client):
    """构造过期的 refresh token，refresh 端点返回 401。"""
    payload = {
        "sub": "nobody",
        "role": "user",
        "type": "refresh",
        "jti": "expired-jti-0001",
        "exp": int(time.time()) - 10,
        "iat": int(time.time()) - 600,
    }
    expired = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    r = client.post("/api/auth/refresh", json={"refresh_token": expired})
    assert r.status_code == 401, r.text


# ============================================================
# 4. 登录失败（HTTP）
# ============================================================

def test_login_wrong_password_returns_401(client):
    """已存在用户 + 错误密码 → 401。"""
    r = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "totally-wrong"},
    )
    assert r.status_code == 401, r.text


def test_login_unknown_user_returns_401(client):
    """不存在的用户 → 401（与错误密码同文案，防枚举）。"""
    r = client.post(
        "/api/auth/login",
        json={"username": "no_such_user", "password": SEED_PASSWORDS["alice"]},
    )
    assert r.status_code == 401, r.text
