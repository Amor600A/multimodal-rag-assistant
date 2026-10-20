# -*- coding: utf-8 -*-
"""pytest 公共 fixture。

Mock 环境零外部依赖：所有后端均为内存实现，测试直接用 TestClient 走真实路由，
种子数据由 lifespan 在启动时注入（部门 / 用户 / 会话 / 知识语料）。

注意：登录与聊天路由存在进程内滑动窗口限流（5 次/分钟），
本文件用 autouse fixture 在每个用例前清空计数桶，避免用例相互干扰。
"""

from __future__ import annotations

from typing import Dict

import pytest
from fastapi.testclient import TestClient

from app.core import ratelimit
from app.main import app

# 种子用户密码（scripts/seed_data.py 常量）
SEED_PASSWORDS: Dict[str, str] = {
    "alice": "alice123",
    "bob": "bob123",
    "admin": "admin123",
}

# 进程级 token 缓存：避免同一用户重复登录触发登录限流
_token_cache: Dict[str, str] = {}


@pytest.fixture(scope="session")
def client():
    """带 lifespan 的 TestClient：进入 with 时触发种子数据注入。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """每个用例前清空限流计数桶（访问内部结构，仅测试使用）。"""
    ratelimit._buckets.clear()
    yield


def login(client: TestClient, username: str, password: str | None = None) -> dict:
    """登录并返回完整响应体（access_token / refresh_token / user）。"""
    r = client.post(
        "/api/auth/login",
        json={"username": username, "password": password or SEED_PASSWORDS[username]},
    )
    assert r.status_code == 200, r.text
    return r.json()


def access_token(client: TestClient, username: str) -> str:
    """登录并缓存 access_token（同一用户全流程只登录一次）。"""
    if username not in _token_cache:
        _token_cache[username] = login(client, username)["access_token"]
    return _token_cache[username]


def auth_headers(client: TestClient, username: str) -> dict:
    """构造 Authorization 头。"""
    return {"Authorization": "Bearer " + access_token(client, username)}


@pytest.fixture
def headers_alice(client) -> dict:
    return auth_headers(client, "alice")


@pytest.fixture
def headers_bob(client) -> dict:
    return auth_headers(client, "bob")


@pytest.fixture
def headers_admin(client) -> dict:
    return auth_headers(client, "admin")
