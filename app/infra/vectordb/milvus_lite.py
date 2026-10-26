# -*- coding: utf-8 -*-
"""Milvus Lite 向量库实现（VECTOR_BACKEND=milvus_lite）。

用 pymilvus MilvusClient 连接本地 .db 文件，接口与 MockMilvus 保持一致：
insert / search / get / count，rag_service 无需感知底层实现。

Windows 兼容性约束（已实测）：
1. schema 必须固定字段，禁止 enable_dynamic_field（开启动态字段会段错误）
2. 不传 index_params，由 MilvusClient 默认建 COSINE 索引（AUTOINDEX 在 Lite 下段错误）
3. insert 后 flush，保证数据立刻可检索

上线切换：把 VECTOR_BACKEND 改为 milvus、MILVUS_URI 指向 standalone 地址即可，
collection schema 与业务代码保持不变。
"""

from __future__ import annotations

import logging
import os

import numpy as np
from pymilvus import DataType, MilvusClient

from app.rag.embedder import get_embedder

logger = logging.getLogger("app.infra.vectordb.milvus_lite")

# 标量字段定义：(字段名, 最大长度)
_SCALAR_FIELDS = (
    ("text", 65535),
    ("scope", 16),
    ("visibility", 16),
    ("owner_user_id", 64),
    ("dept_id", 64),
    ("source", 512),
    # 来源文件类型（pdf/docx/txt/png/jpg/video），便于按类型区分与过滤
    ("file_type", 16),
)

# 检索时需要回填的输出字段
_OUTPUT_FIELDS = [name for name, _ in _SCALAR_FIELDS]


class MilvusLite:
    """Milvus Lite 向量库（本地单文件数据库）。"""

    def __init__(self, db_path: str, collection_name: str, dim: int) -> None:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._client = MilvusClient(db_path)
        self._collection = collection_name
        self._dim = dim
        # 与 MockMilvus 共用同一嵌入器单例，保证两套后端向量空间一致
        self._embedder = get_embedder()
        self._ensure_collection()
        logger.info(
            "MilvusLite 初始化：db=%s collection=%s dim=%d",
            db_path,
            collection_name,
            dim,
        )

    def _ensure_collection(self) -> None:
        """幂等建 collection：字段固定，不开启动态字段。"""
        if self._client.has_collection(self._collection):
            self._check_schema()
            return

        schema = self._client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        for name, max_length in _SCALAR_FIELDS:
            schema.add_field(name, DataType.VARCHAR, max_length=max_length)

        self._client.create_collection(self._collection, schema=schema)
        logger.info("MilvusLite 已创建 collection：%s", self._collection)

    def _check_schema(self) -> None:
        """校验已存在 collection 的标量字段与当前定义一致。

        旧版本建出的 collection 缺少 file_type 字段，直接读写会报字段不存在，
        这里提前给出明确提示，避免用户面对晦涩的 Milvus 报错。
        """
        info = self._client.describe_collection(self._collection)
        existing = {field["name"] for field in info.get("fields", [])}
        missing = [name for name, _ in _SCALAR_FIELDS if name not in existing]
        if missing:
            raise RuntimeError(
                f"collection {self._collection} 缺少字段 {missing}："
                "当前 schema 已升级，请删除数据文件后重新建库"
            )

    def insert(self, records: list[dict]) -> None:
        """批量写入向量记录（同 id 覆盖，幂等）。

        Milvus 的 insert 对相同主键是追加而非覆盖，会产生重复记录，
        故使用 upsert 保持与 MockMilvus 的"同 id 覆盖"语义一致
        （种子数据与记忆写入会重复执行同一批 id）。
        """
        rows: list[dict] = []
        for r in records:
            vec = r.get("vector")
            if vec is None:
                vec = self._embedder.embed(r["text"])
            elif not isinstance(vec, np.ndarray):
                vec = np.array(vec, dtype=np.float32)

            rows.append({
                "id": r["id"],
                "vector": vec.tolist(),
                "text": r["text"],
                "scope": r.get("scope", "domain"),
                "visibility": r.get("visibility", "private"),
                # Milvus VARCHAR 不接受 null，统一用空串表示"无归属"
                "owner_user_id": r.get("owner_user_id") or "",
                "dept_id": r.get("dept_id") or "",
                "source": r.get("source", "manual"),
                "file_type": r.get("file_type") or "",
            })

        if not rows:
            return

        self._client.upsert(self._collection, rows)
        self._client.flush(self._collection)

    def search(
        self, expr: str, query: str | np.ndarray, limit: int = 10, timeout: float | None = None
    ) -> list[dict]:
        """expr 过滤 + COSINE 相似度排序。

        expr 直接交给 Milvus 标量过滤，语法与 MockMilvus 解析器支持的子集一致。
        timeout 预留接口：当前 Milvus Lite 本地单文件忽略；Milvus standalone 阶段透传 pymilvus。
        """
        if isinstance(query, str):
            qvec = self._embedder.embed(query)
        else:
            qvec = np.asarray(query, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(qvec)
            if norm > 0:
                qvec = qvec / norm

        search_kwargs = {
            "data": [qvec.tolist()],
            "limit": limit,
            "output_fields": _OUTPUT_FIELDS,
        }
        # expr 为空表示不做权限过滤（仅调试脚本使用；线上检索始终传 expr）
        if expr:
            search_kwargs["filter"] = expr

        hits = self._client.search(self._collection, **search_kwargs)[0]

        results: list[dict] = []
        for hit in hits:
            entity = hit["entity"]
            item = {"id": hit["id"], "score": float(hit["distance"])}
            for name in _OUTPUT_FIELDS:
                item[name] = entity.get(name, "")
            results.append(item)
        return results

    def get(self, record_id: str) -> dict | None:
        """按主键取单条记录。"""
        rows = self._client.get(
            self._collection,
            ids=[record_id],
            output_fields=_OUTPUT_FIELDS,
        )
        if not rows:
            return None
        row = rows[0]
        item = {"id": row["id"]}
        for name in _OUTPUT_FIELDS:
            item[name] = row.get(name, "")
        return item

    def count(self) -> int:
        """集合内记录总数。"""
        stats = self._client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))
