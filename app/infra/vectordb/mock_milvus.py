# -*- coding: utf-8 -*-
"""MockMilvus：numpy 余弦检索 + 有限子集 expr 解析器 + 种子语料。

对齐目标：pymilvus MilvusClient 的 search/insert 行为语义（参数形态、返回结构），
不复刻 Milvus 物理实现。切换真实 Milvus 时仅改 factory 分支，业务代码零改动。

expr 子集（方案 6.3）：字段名、==、字符串字面量、and、or、括号。
子集外语法显式抛 ExprParseError，防止权限逻辑静默失效。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from app.rag.embedder import get_embedder

logger = logging.getLogger("app.infra.vectordb.mock_milvus")


class ExprParseError(ValueError):
    """expr 语法错误（子集外）。"""


# ---- 向量记录 ----

@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    text: str
    scope: str           # domain | personal
    visibility: str       # public | dept | private
    owner_user_id: str | None = None
    dept_id: str | None = None
    source: str = "manual"
    file_type: str = ""   # 来源文件类型（pdf/docx/txt/png/video）
    created_at: int = field(default_factory=lambda: int(time.time()))


# ---- expr 解析器（有限子集：== / and / or / 字符串字面量 / 括号）----

_TOKEN_RE = r'(\band\b|\bor\b|==|\(|\)|"[^"]*"|[A-Za-z_][A-Za-z0-9_]*)'


def _tokenize(expr: str) -> list[str]:
    tokens = re.findall(_TOKEN_RE, expr)
    return tokens


def _parse_expr(tokens: list[str], pos: list[int]) -> Callable[[dict], bool]:
    """解析 or_expr。"""
    left = _parse_and(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]] == "or":
        pos[0] += 1
        right = _parse_and(tokens, pos)
        # 延迟求值
        l, r = left, right
        left = lambda rec, _l=l, _r=r: _l(rec) or _r(rec)
    return left


def _parse_and(tokens: list[str], pos: list[int]) -> Callable[[dict], bool]:
    left = _parse_factor(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]] == "and":
        pos[0] += 1
        right = _parse_factor(tokens, pos)
        l, r = left, right
        left = lambda rec, _l=l, _r=r: _l(rec) and _r(rec)
    return left


def _parse_factor(tokens: list[str], pos: list[int]) -> Callable[[dict], bool]:
    if pos[0] >= len(tokens):
        raise ExprParseError("expr 意外结束")
    tok = tokens[pos[0]]
    if tok == "(":
        pos[0] += 1
        inner = _parse_expr(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
            raise ExprParseError("缺少右括号")
        pos[0] += 1
        return inner
    # comparison: IDENT == STRING
    field = tok
    pos[0] += 1
    if pos[0] >= len(tokens) or tokens[pos[0]] != "==":
        raise ExprParseError(f"期望 '=='，得到 {tokens[pos[0]] if pos[0] < len(tokens) else 'EOF'}")
    pos[0] += 1
    if pos[0] >= len(tokens):
        raise ExprParseError("缺少比较值")
    val_tok = tokens[pos[0]]
    pos[0] += 1
    if not (val_tok.startswith('"') and val_tok.endswith('"')):
        raise ExprParseError(f"比较值必须是字符串字面量: {val_tok}")
    value = val_tok[1:-1]
    return lambda rec, f=field, v=value: str(rec.get(f)) == v


def compile_expr(expr: str) -> Callable[[dict], bool]:
    """把 expr 字符串编译为 record 谓词函数。"""
    tokens = _tokenize(expr)
    pos = [0]
    predicate = _parse_expr(tokens, pos)
    if pos[0] != len(tokens):
        raise ExprParseError(f"expr 尾部有多余 token: {tokens[pos[0]:]}")
    return predicate


# ---- MockMilvus ----

class MockMilvus:
    """模拟 MilvusClient 的核心行为。"""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        # 嵌入器由工厂统一提供：Mock 与 Milvus Lite 共用同一实例，向量空间一致
        self._embedder = get_embedder()
        logger.info("MockMilvus 初始化")

    def insert(self, records: list[dict]) -> None:
        """批量插入向量记录。

        record 字段（方案 6.1）：id, text, scope, visibility, owner_user_id?, dept_id?, source?, file_type?
        vector 由 embedder 自动生成（也可显式传入）。
        """
        for r in records:
            vec = r.get("vector")
            if vec is None:
                vec = self._embedder.embed(r["text"])
            elif not isinstance(vec, np.ndarray):
                vec = np.array(vec, dtype=np.float32)
            rec = VectorRecord(
                id=r["id"],
                vector=vec,
                text=r["text"],
                scope=r.get("scope", "domain"),
                visibility=r.get("visibility", "private"),
                owner_user_id=r.get("owner_user_id"),
                dept_id=r.get("dept_id"),
                source=r.get("source", "manual"),
                file_type=r.get("file_type") or "",
            )
            self._records[r["id"]] = rec

    def search(
        self, expr: str, query: str | np.ndarray, limit: int = 10, timeout: float | None = None
    ) -> list[dict]:
        """expr 过滤 + 余弦相似度排序。

        Args:
            expr: 权限过滤表达式（方案 6.3 子集）
            query: 查询文本或向量
            limit: 返回条数
            timeout: 超时秒数（当前 Mock 纯内存计算忽略；Milvus standalone 阶段透传 pymilvus）

        Returns:
            按相似度降序的记录 dict 列表（含 score 字段）
        """
        if isinstance(query, str):
            qvec = self._embedder.embed(query)
        else:
            qvec = query

        predicate = compile_expr(expr)

        # 过滤
        candidates = [r for r in self._records.values() if predicate(_record_to_dict(r))]
        if not candidates:
            return []

        # 余弦相似度
        mat = np.stack([r.vector for r in candidates])
        scores = mat @ qvec  # 已 L2 归一化，点积即余弦
        order = np.argsort(-scores)[:limit]

        results = []
        for idx in order:
            r = candidates[idx]
            d = _record_to_dict(r)
            d["score"] = float(scores[idx])
            results.append(d)
        return results

    def get(self, record_id: str) -> dict | None:
        r = self._records.get(record_id)
        return _record_to_dict(r) if r else None

    def count(self) -> int:
        return len(self._records)


def _record_to_dict(r: VectorRecord) -> dict:
    return {
        "id": r.id,
        "text": r.text,
        "scope": r.scope,
        "visibility": r.visibility,
        "owner_user_id": r.owner_user_id,
        "dept_id": r.dept_id,
        "source": r.source,
        "file_type": r.file_type,
        "created_at": r.created_at,
    }
