# -*- coding: utf-8 -*-
"""向量库工厂：按 VECTOR_BACKEND 返回对应实现。

mock：MockMilvus（numpy 余弦检索，进程内存储）
milvus_lite：MilvusLite（本地 .db 文件，pymilvus MilvusClient）
milvus：pymilvus MilvusClient 连接 standalone（待实现）
"""

from __future__ import annotations

import warnings

from app.core.config import get_settings

warnings.filterwarnings("ignore", message="LangChainPendingDeprecationWarning")

_cached = None


def get_vector_db():
    """进程级向量库单例。"""
    global _cached
    if _cached is not None:
        return _cached

    settings = get_settings()
    backend = settings.vector_backend
    if backend == "mock":
        from app.infra.vectordb.mock_milvus import MockMilvus

        _cached = MockMilvus()
    elif backend == "milvus_lite":
        from app.infra.vectordb.milvus_lite import MilvusLite

        _cached = MilvusLite(
            db_path=settings.milvus_db_path,
            collection_name=settings.milvus_collection,
            dim=settings.embedding_dim,
        )
    else:
        # milvus 阶段 8 实现；当前 fallback 到 mock
        import logging

        logging.getLogger("app.infra.vectordb").warning(
            "vector_backend=%s 暂未实现，fallback 到 mock", backend
        )
        from app.infra.vectordb.mock_milvus import MockMilvus

        _cached = MockMilvus()
    return _cached
