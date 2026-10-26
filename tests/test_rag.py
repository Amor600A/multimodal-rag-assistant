# -*- coding: utf-8 -*-
"""RAG 链路集成测试：入库 -> 检索命中、跨部门权限隔离、HashEmbedder 确定性。

重要声明（方案 8.4 / 12.1）：
本测试仅验证权限过滤与词面匹配链路，不验证语义泛化质量。
EMBEDDING_BACKEND=hash 下检索只按词面重叠召回（中文 2-gram），
因此所有查询词必须包含语料关键词；语义召回质量验证推迟到 api 嵌入阶段。
"""

from __future__ import annotations

import numpy as np

from app.core.deps import CurrentUser
from app.rag.embedder import HashEmbedder
from app.services.rag_service import get_rag_service


def _current_user(client, headers) -> CurrentUser:
    """通过 /api/auth/me 取实时用户身份（含实时 dept_id）。"""
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200, r.text
    b = r.json()
    return CurrentUser(
        user_id=b["user_id"],
        username=b["username"],
        role=b["role"],
        dept_id=b["dept_id"],
    )


def _hit_ids(user: CurrentUser, query: str) -> set[str]:
    """检索并返回命中的 doc_id 集合（召回条数由配置 RECALL_TOP_N 控制）。"""
    return {c.doc_id for c in get_rag_service().search(user, query)}


# ---- 1. 入库 -> 检索命中（词面重叠）----

def test_ingest_then_retrieve_hit():
    """新入库的 public 语料能被检索命中（查询词含语料关键词）。"""
    rag = get_rag_service()
    doc_id = "rag-test-ingest-1"
    rag.ingest([
        {
            "id": doc_id,
            "text": "新员工培训计划：入职首月需完成安全培训与业务培训，培训结束后统一考核。",
            "scope": "domain",
            "visibility": "public",
        }
    ])
    user = CurrentUser(user_id="rag-test-user", username="rag_tester", role="user", dept_id=None)
    assert doc_id in _hit_ids(user, "培训")


# ---- 2. 检索隔离（跨部门权限过滤，轻量断言）----

def test_retrieval_isolation_by_dept(client, headers_alice, headers_bob, headers_admin):
    """不同部门用户召回的集合符合权限矩阵（重活在 test_permissions.py）。"""
    alice = _current_user(client, headers_alice)   # 人事部
    bob = _current_user(client, headers_bob)       # 市场部
    admin = _current_user(client, headers_admin)   # 无部门

    # 市场部 dept 文档：人事部 alice 不可见，市场部 bob 可见
    assert "mkt-brand" not in _hit_ids(alice, "品牌视觉手册")
    assert "mkt-brand" in _hit_ids(bob, "品牌视觉手册")

    # admin 无部门：只能看 public，部门文档不可见（方案 12.2 权限矩阵）
    assert "mkt-brand" not in _hit_ids(admin, "品牌视觉手册")

    # public 文档：全员可见
    assert "pub-travel-reimburse" in _hit_ids(alice, "差旅报销")
    assert "pub-travel-reimburse" in _hit_ids(admin, "差旅报销")


# ---- 3. HashEmbedder 确定性 ----

def test_hash_embedder_determinism_and_shape():
    """同文本两次嵌入完全相等；维度 1536；L2 范数约 1；不同文本不同。"""
    e = HashEmbedder()
    assert e.dim == 1536

    text = "考勤与请假制度：正式员工每月 2 天带薪病假"
    v1 = e.embed(text)
    v2 = e.embed(text)

    assert isinstance(v1, np.ndarray)
    assert v1.shape == (1536,)
    assert v1.dtype == np.float32
    # 确定性：同一实例两次 + 新实例一次，结果完全一致（方案 8.4 红线）
    assert np.array_equal(v1, v2)
    assert np.array_equal(HashEmbedder().embed(text), v1)

    # L2 归一化：范数约等于 1
    norm = float(np.linalg.norm(v1))
    assert abs(norm - 1.0) < 1e-5

    # 不同文本产生不同向量
    v3 = e.embed("完全不同的另一段文本内容，用于验证向量差异")
    assert not np.array_equal(v1, v3)
