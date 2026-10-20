# -*- coding: utf-8 -*-
"""RAG 服务：文档入库、权限 expr 构造、双路召回、二次权限校验。

权限防线（方案 6.3）：
1. _build_permission_expr 生成 Milvus expr（标量过滤）
2. vector_db.search(expr, ...) 执行过滤
3. 返回前逐条二次校验（兜底，即使 expr 出错也能拦截越权）
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.core.deps import CurrentUser
from app.infra.vectordb.factory import get_vector_db

logger = logging.getLogger("app.services.rag")

# expr 字符串字面量安全字符白名单（防 expr 注入）。
# expr 子集语法的字符串字面量不支持转义，故采用白名单校验：
# 仅允许字母/数字/下划线/连字符（uuid hex、平台外部 ID 均满足）。
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_expr_value(value: str, field: str) -> str:
    """校验进入 expr 的值只含安全字符；非法值直接抛错（不静默放行，方案 6.3）。"""
    if not _SAFE_VALUE_RE.match(value):
        raise ValueError(f"非法的 {field} 字符，拒绝构造权限 expr: {value!r}")
    return value


@dataclass
class RetrievedChunk:
    """召回片段（带短编号，供 generate 节点组装 Prompt 和 output_guard 校验）。"""
    index: int            # 短编号 [1] [2] ...
    doc_id: str
    text: str
    source: str
    scope: str
    score: float


class RAGService:
    """RAG 服务单例。"""

    def __init__(self) -> None:
        self._db = get_vector_db()

    # ---- 入库 ----

    def ingest(self, records: list[dict]) -> None:
        """批量入库（幂等：同 id 覆盖）。"""
        self._db.insert(records)
        logger.info("RAG 入库 %d 条", len(records))

    # ---- 权限 expr ----

    @staticmethod
    def _build_permission_expr(user: CurrentUser) -> str:
        """根据当前用户身份构造检索权限 expr。

        三路 or 合并（方案 6.3 可见性矩阵）。仅使用 MockMilvus 支持的语法子集：
        字段名、==、字符串字面量、and、or、括号。
        """
        clauses: list[str] = []
        # public：全员可见
        clauses.append('(scope=="domain" and visibility=="public")')
        # dept：本部门文档（dept_id 非空时才加；白名单校验防注入）
        if user.dept_id:
            dept = _safe_expr_value(user.dept_id, "dept_id")
            clauses.append(
                f'(scope=="domain" and visibility=="dept" and dept_id=="{dept}")'
            )
        # private：仅本人（白名单校验防注入）
        uid = _safe_expr_value(user.user_id, "user_id")
        clauses.append(
            f'((scope=="personal" or visibility=="private") and owner_user_id=="{uid}")'
        )
        return " or ".join(clauses)

    # ---- 二次校验 ----

    @staticmethod
    def _check_permission(rec: dict, user: CurrentUser) -> bool:
        """逐条复核权限字段（兜底防线）。"""
        vis = rec.get("visibility")
        scope = rec.get("scope")
        if vis == "public":
            return scope == "domain"
        if vis == "dept":
            return scope == "domain" and rec.get("dept_id") == user.dept_id
        if vis == "private":
            return rec.get("owner_user_id") == user.user_id
        # personal scope 也是仅本人
        if scope == "personal":
            return rec.get("owner_user_id") == user.user_id
        return False  # 默认拒绝

    # ---- 检索 ----

    def search(self, user: CurrentUser, query: str, limit: int = 5) -> list[RetrievedChunk]:
        """双路召回（domain + personal 在同一个 expr 里合并）+ 二次校验。"""
        expr = self._build_permission_expr(user)
        raw_results = self._db.search(expr, query, limit=limit * 2)  # 多取一些，二次校验可能剔除

        chunks: list[RetrievedChunk] = []
        for rec in raw_results:
            if not self._check_permission(rec, user):
                logger.warning(
                    "权限二次校验剔除: doc=%s user=%s vis=%s scope=%s",
                    rec.get("id"), user.user_id, rec.get("visibility"), rec.get("scope"),
                )
                continue
            chunks.append(RetrievedChunk(
                index=len(chunks) + 1,
                doc_id=rec["id"],
                text=rec["text"],
                source=rec.get("source", "manual"),
                scope=rec.get("scope", "domain"),
                score=rec.get("score", 0.0),
            ))
            if len(chunks) >= limit:
                break
        return chunks


_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service
