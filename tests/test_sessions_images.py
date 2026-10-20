# -*- coding: utf-8 -*-
"""集成测试：会话归属隔离 + 图片上传/代理访问 + 图片 bound 置位（方案 12.2 / 7.1 / 5.4）。

覆盖范围：
1. 会话：新建→列表可见、按用户隔离、越权删除 403、删除不存在 404、本人删除 204。
2. 图片：上传 201、本人访问 200 且字节一致、他人访问 404（不暴露存在性）、
   不存在 404、非图片类型 415、query token 访问 200、无凭证 404。
3. 图片引用置位：chat 携带 image_id 后 images.bound 由 False 变 True，未携带的保持 False。

说明：图片仓储为进程级内存单例（app.infra.images），测试中直接用
asyncio.run(repo.get(image_id)) 读取，与 TestClient 内部线程共享同一进程内字典。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.infra.images import get_image_repository
from tests.conftest import access_token

# 1x1 透明 PNG（67 字节，合法 PNG 签名/IHDR/IDAT/IEND）
PNG_BYTES = bytes.fromhex(
    "89504E470D0A1A0A"
    "0000000D49484452000000010000000108060000001F15C489"
    "0000000A49444154789C63000100000500010D0A2DB4"
    "0000000049454E44AE426082"
)


# ---------- 辅助 fixture ----------

@pytest.fixture
def alice_token(client) -> str:
    """alice 的 access_token（供图片 query token 场景）。"""
    return access_token(client, "alice")


def _create_session(client, headers: dict, *, title: str | None = None) -> dict:
    """新建会话并返回响应体（thread_id / title / created_at）。"""
    body: dict = {}
    if title is not None:
        body["title"] = title
    r = client.post("/api/sessions", json=body, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _list_thread_ids(client, headers: dict) -> list[str]:
    """列出某用户全部会话的 thread_id。"""
    r = client.get("/api/sessions", headers=headers)
    assert r.status_code == 200, r.text
    return [s["thread_id"] for s in r.json()]


def _upload_image(client, headers: dict, *, content_type: str = "image/png", content: bytes = PNG_BYTES) -> dict:
    """上传图片，返回响应体（image_id / url）。"""
    r = client.post(
        "/api/knowledge/images",
        files={"file": ("t.png", content, content_type)},
        headers=headers,
    )
    return {"status": r.status_code, "json": (r.json() if r.headers.get("content-type", "").startswith("application/json") else None), "resp": r}


# ========== 1. 会话 ==========

def test_create_session_appears_in_list(client, headers_alice):
    """新建会话 → 列表里出现该会话，title 正确。"""
    title = "测试会话+" + uuid.uuid4().hex[:8]
    created = _create_session(client, headers_alice, title=title)

    r = client.get("/api/sessions", headers=headers_alice)
    assert r.status_code == 200, r.text
    sessions = {s["thread_id"]: s for s in r.json()}

    assert created["thread_id"] in sessions
    assert sessions[created["thread_id"]]["title"] == title
    assert created["title"] == title


def test_session_list_isolated_by_user(client, headers_alice, headers_bob):
    """会话列表按用户隔离：alice 看不到 bob 的会话。"""
    created = _create_session(client, headers_bob, title="bob 私有会话")

    alice_ids = _list_thread_ids(client, headers_alice)
    assert created["thread_id"] not in alice_ids
    # 种子数据里的 bob 会话也不应出现在 alice 列表
    assert "seed-bob-1" not in alice_ids


def test_bob_cannot_delete_alice_session(client, headers_alice, headers_bob):
    """bob 删除 alice 的会话 → 403（越权硬断言）。"""
    alice_session = _create_session(client, headers_alice, title="alice 待保护会话")

    r = client.delete("/api/sessions/" + alice_session["thread_id"], headers=headers_bob)
    assert r.status_code == 403, r.text

    # alice 的会话仍在
    assert alice_session["thread_id"] in _list_thread_ids(client, headers_alice)


def test_delete_nonexistent_session_404(client, headers_alice):
    """删除不存在的会话 → 404。"""
    r = client.delete("/api/sessions/" + uuid.uuid4().hex, headers=headers_alice)
    assert r.status_code == 404, r.text


def test_owner_delete_own_session(client, headers_alice):
    """本人删除自己的会话 → 204，且列表里不再出现。"""
    created = _create_session(client, headers_alice, title="待删除会话")
    assert created["thread_id"] in _list_thread_ids(client, headers_alice)

    r = client.delete("/api/sessions/" + created["thread_id"], headers=headers_alice)
    assert r.status_code == 204, r.text
    assert created["thread_id"] not in _list_thread_ids(client, headers_alice)


# ========== 2. 图片 ==========

def test_upload_image_201(client, headers_alice):
    """alice 上传 1x1 PNG → 201，返回 image_id 与 url。"""
    out = _upload_image(client, headers_alice)
    assert out["status"] == 201, out["resp"].text

    body = out["json"]
    assert body is not None
    image_id = body["image_id"]
    assert image_id
    assert body["url"] == "/api/knowledge/images/" + image_id


def test_owner_get_image_200_bytes_match(client, headers_alice):
    """alice 自己 GET 该图片 → 200，Content-Type=image/png，字节与上传一致。"""
    image_id = _upload_image(client, headers_alice)["json"]["image_id"]

    r = client.get("/api/knowledge/images/" + image_id, headers=headers_alice)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    assert r.content == PNG_BYTES


def test_bob_get_alice_image_404(client, headers_alice, headers_bob):
    """bob GET alice 的图片 → 404（不暴露资源存在性）。"""
    image_id = _upload_image(client, headers_alice)["json"]["image_id"]

    r = client.get("/api/knowledge/images/" + image_id, headers=headers_bob)
    assert r.status_code == 404, r.text


def test_get_nonexistent_image_404(client, headers_alice):
    """不存在的 image_id → 404。"""
    r = client.get("/api/knowledge/images/" + str(uuid.uuid4()), headers=headers_alice)
    assert r.status_code == 404, r.text


def test_upload_non_image_415(client, headers_alice):
    """非图片类型（text/plain）上传 → 415。"""
    out = _upload_image(client, headers_alice, content_type="text/plain", content=b"hello")
    assert out["status"] == 415, out["resp"].text


def test_get_image_with_query_token_200(client, headers_alice, alice_token):
    """GET ...?token=<alice token> → 200（该端点支持 query token，供 img 标签）。"""
    image_id = _upload_image(client, headers_alice)["json"]["image_id"]

    r = client.get("/api/knowledge/images/" + image_id + "?token=" + alice_token)
    assert r.status_code == 200, r.text
    assert r.content == PNG_BYTES


def test_get_image_without_credentials_404(client, headers_alice):
    """未携带任何凭证访问图片 → 404。"""
    image_id = _upload_image(client, headers_alice)["json"]["image_id"]

    r = client.get("/api/knowledge/images/" + image_id)
    assert r.status_code == 404, r.text


# ========== 3. 图片引用置位（bound） ==========

def test_chat_with_image_marks_bound(client, headers_alice):
    """chat 携带 image_id 后，该图片 bound 由 False 变 True；未携带的保持 False。"""
    repo = get_image_repository()

    used_image_id = _upload_image(client, headers_alice)["json"]["image_id"]
    other_image_id = _upload_image(client, headers_alice)["json"]["image_id"]

    # 前置断言：上传时 bound 均为 False
    assert asyncio.run(repo.get(used_image_id)).bound is False
    assert asyncio.run(repo.get(other_image_id)).bound is False

    # 建立 alice 自己的会话后发起 chat（携带 used_image_id）
    session = _create_session(client, headers_alice, title="图片会话")
    r = client.post(
        "/api/chat",
        json={"message": "这张图片里是什么？", "thread_id": session["thread_id"], "image_id": used_image_id},
        headers=headers_alice,
    )
    assert r.status_code == 200, r.text

    # 携带的图片被置位，未携带的保持 False
    assert asyncio.run(repo.get(used_image_id)).bound is True
    assert asyncio.run(repo.get(other_image_id)).bound is False
