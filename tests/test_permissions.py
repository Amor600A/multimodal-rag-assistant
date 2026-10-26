# -*- coding: utf-8 -*-
"""权限回归矩阵（方案 12.2），每条均为「越权必须失败」的硬断言。

背景：EMBEDDING_BACKEND=hash 下检索只按词面重叠召回，无语义泛化，
故所有查询词都包含语料里的关键词；禁止依赖同义词/改写句的语义召回。

防线两条：expr 标量过滤（_build_permission_expr）+ rag_service 返回前二次校验。
本文件直接断言 rag_service.search() 的最终召回集合（两条防线收口后的结果）。
会话删除与图片访问走真实 HTTP 路由断言 403 / 404。
"""

from __future__ import annotations

import base64

import pytest

from app.core.deps import CurrentUser
from app.services.rag_service import RetrievedChunk, get_rag_service

# 1x1 PNG（合法 PNG 头，服务端只校验 MIME 白名单与非空，不解码像素）
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@pytest.fixture
def current_users(client, headers_alice, headers_bob, headers_admin) -> dict[str, CurrentUser]:
    """通过 GET /api/auth/me 构造三类用户的 CurrentUser（实时查库含 dept_id）。"""

    def _fetch(headers: dict) -> CurrentUser:
        r = client.get("/api/auth/me", headers=headers)
        assert r.status_code == 200, r.text
        d = r.json()
        return CurrentUser(
            user_id=d["user_id"],
            username=d["username"],
            role=d["role"],
            dept_id=d["dept_id"],
        )

    return {
        "alice": _fetch(headers_alice),
        "bob": _fetch(headers_bob),
        "admin": _fetch(headers_admin),
    }


def _recall_ids(user: CurrentUser, query: str) -> list[str]:
    """执行一次检索并返回最终召回集合的 doc_id 列表（召回条数由配置控制）。"""
    chunks: list[RetrievedChunk] = get_rag_service().search(user, query)
    return [c.doc_id for c in chunks]


# ---- 1. alice（人事部）检索市场部品牌文档 → 无召回 ----

def test_alice_cannot_recall_marketing_docs(current_users):
    """alice 属人事部，查「品牌视觉手册」不得召回任何 mkt-* 文档。"""
    ids = _recall_ids(current_users["alice"], "品牌视觉手册")
    assert all(not doc_id.startswith("mkt-") for doc_id in ids), ids

    # 正向对照：证明语料存在且被隔离的是权限而非数据缺失
    bob_ids = _recall_ids(current_users["bob"], "品牌视觉手册")
    assert "mkt-brand" in bob_ids


# ---- 2. bob（市场部）检索人事部考勤文档 → 无召回 ----

def test_bob_cannot_recall_hr_docs(current_users):
    """bob 属市场部，查「考勤请假制度」不得召回任何 hr-* 文档。"""
    ids = _recall_ids(current_users["bob"], "考勤请假制度")
    assert all(not doc_id.startswith("hr-") for doc_id in ids), ids

    # 正向对照：alice 可召回人事部考勤文档
    alice_ids = _recall_ids(current_users["alice"], "考勤请假制度")
    assert "hr-attendance" in alice_ids


# ---- 3. alice 检索 public 差旅制度 → 有召回 ----

def test_alice_can_recall_public_docs(current_users):
    ids = _recall_ids(current_users["alice"], "差旅报销制度")
    assert "pub-travel-reimburse" in ids, ids


# ---- 4. alice 检索自己上传的 private 文档 → 有召回；5. bob 检索 → 无召回 ----

def test_alice_private_doc_visible_to_self_only(client, headers_alice, current_users):
    """普通用户经 POST /api/knowledge 上传 private 文档；本人可召回，他人不可。"""
    keyword = "天璇私有笔记"
    text = f"{keyword}：这是 alice 的私有笔记正文，仅本人可见。"
    r = client.post(
        "/api/knowledge",
        headers=headers_alice,
        json={"text": text, "visibility": "private"},
    )
    assert r.status_code == 201, r.text
    doc_id = r.json()["id"]
    assert r.json()["visibility"] == "private"

    # alice 本人：有召回
    alice_ids = _recall_ids(current_users["alice"], keyword)
    assert doc_id in alice_ids, alice_ids

    # bob：无召回（且不应召回任何 personal 域文档）
    bob_chunks = get_rag_service().search(current_users["bob"], keyword)
    bob_ids = [c.doc_id for c in bob_chunks]
    assert doc_id not in bob_ids, bob_ids
    assert all(c.scope != "personal" for c in bob_chunks), bob_ids


# ---- 6. admin 检索部门区文档 → 无召回（仅 public）----

def test_admin_cannot_recall_dept_docs(current_users):
    """admin 无部门，expr 不含 dept 分支：部门文档一律不可召回，public 仍可见。"""
    admin = current_users["admin"]
    for query in ("考勤请假制度", "品牌视觉手册"):
        ids = _recall_ids(admin, query)
        assert all(
            not (doc_id.startswith("hr-") or doc_id.startswith("mkt-"))
            for doc_id in ids
        ), (query, ids)

    # 正向对照：admin 可召回 public 文档
    assert "pub-travel-reimburse" in _recall_ids(admin, "差旅报销制度")


# ---- 7. bob 删除 alice 的会话 → 403 ----

def test_bob_cannot_delete_alice_session(client, headers_alice, headers_bob):
    # 前置：seed-alice-1 确实归属 alice
    alice_sessions = client.get("/api/sessions", headers=headers_alice).json()
    assert any(s["thread_id"] == "seed-alice-1" for s in alice_sessions)

    r = client.delete("/api/sessions/seed-alice-1", headers=headers_bob)
    assert r.status_code == 403, r.text

    # 越权删除后会话仍存在（未被误删）
    still = client.get("/api/sessions", headers=headers_alice).json()
    assert any(s["thread_id"] == "seed-alice-1" for s in still)


# ---- 8. bob 访问 alice 的 image_id → 404（而非 403，防枚举）----

def test_bob_cannot_access_alice_image(client, headers_alice, headers_bob):
    up = client.post(
        "/api/knowledge/images",
        headers=headers_alice,
        files={"file": ("tiny.png", _TINY_PNG, "image/png")},
    )
    assert up.status_code == 201, up.text
    image_id = up.json()["image_id"]

    # alice 本人访问：200
    own = client.get(f"/api/knowledge/images/{image_id}", headers=headers_alice)
    assert own.status_code == 200, own.text

    # bob 越权访问：404（不暴露资源存在性）
    other = client.get(f"/api/knowledge/images/{image_id}", headers=headers_bob)
    assert other.status_code == 404, other.text
