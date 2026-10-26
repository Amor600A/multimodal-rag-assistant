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


# ---- 权限 expr ----
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


@dataclass
class RetrievedChunk:
    """召回片段（带短编号，供 generate 节点组装 Prompt 和 output_guard 校验）。"""
    index: int            # 短编号 [1] [2] ...
    doc_id: str
    text: str
    source: str
    scope: str
    score: float
    file_type: str = ""   # 来源文件类型（pdf/docx/txt/png/video）


def _build_chunks(candidates: list[dict]) -> list[RetrievedChunk]:
    """将 dict 候选列表组装成 RetrievedChunk，index 从 1 开始。"""
    chunks: list[RetrievedChunk] = []
    for rec in candidates:
        chunks.append(RetrievedChunk(
            index=len(chunks) + 1,
            doc_id=rec["id"],
            text=rec["text"],
            source=rec.get("source", "manual"),
            scope=rec.get("scope", "domain"),
            score=rec.get("score", 0.0),
            file_type=rec.get("file_type", ""),
        ))
    return chunks


class RAGService:
    """RAG 服务单例。"""

    def __init__(self) -> None:
        self._db = get_vector_db()

    # ---- 入库 ----
    def ingest(self, records: list[dict]) -> None:
        """批量入库（幂等：同 id 覆盖）。"""
        self._db.insert(records)
        logger.info("RAG 入库 %d 条", len(records))

    # ---- 检索 ----
    def search(self, user: CurrentUser, query: str) -> list[RetrievedChunk]:
        """向量初筛 + 可选 Rerank 精排，统一入口。

        取数由配置统一控制：无重排直接召回 RECALL_TOP_N；
        有重排先召回RERANK_INITIAL_TOP_N，精排后取 RECALL_TOP_N。
        """
        from app.core.config import get_settings

        settings = get_settings()
        final_n = settings.recall_top_n
        candidates = self._initial_retrieve(user, query, final_n, settings)

        if settings.rerank_enabled and candidates:
            candidates = self._rerank_candidates(query, candidates, settings)

        return _build_chunks(candidates[:final_n])

    def _initial_retrieve(
        self, user: CurrentUser, query: str, final_n: int, settings
    ) -> list[dict]:
        """向量初筛：权限 expr + 二次校验，返回 dict 候选列表。

        有重排时初检放宽到 RERANK_INITIAL_TOP_N，无重排直接按最终召回数。
        """
        expr = _build_permission_expr(user)
        initial_top_n = settings.rerank_initial_top_n if settings.rerank_enabled else final_n
        raw_results = self._db.search(expr, query, limit=initial_top_n)

        candidates: list[dict] = []
        for rec in raw_results:
            if not _check_permission(rec, user):
                logger.warning(
                    "权限二次校验剔除: doc=%s user=%s vis=%s scope=%s",
                    rec.get("id"), user.user_id, rec.get("visibility"), rec.get("scope"),
                )
                continue
            candidates.append(rec)
        return candidates

    @staticmethod
    def _rerank_candidates(query: str, candidates: list[dict], settings) -> list[dict]:
        """Rerank 精排：DashScope qwen3-rerank 重打分，按 relevance_score 降序返回。

        原地替换 candidates[i]["score"] 为 rerank score，方便下游统一读取。
        """
        from app.rag.reranker import rerank as _rerank

        doc_texts = [c["text"] for c in candidates]
        rank_pairs = _rerank(query, doc_texts)
        logger.info(
            "Rerank 精排: 候选 %d → top %d (model=%s)",
            len(candidates), len(candidates), settings.rerank_model,
        )
        # 按 rerank 分数重排，score 字段替换为 rerank score
        reranked: list[dict] = []
        for original_idx, rerank_score in rank_pairs:
            rec = candidates[original_idx].copy()
            rec["score"] = rerank_score
            reranked.append(rec)
        return reranked


_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service
