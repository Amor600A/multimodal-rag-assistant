# -*- coding: utf-8 -*-
"""MockPostgresStore：模拟 Postgres Store 行为。

对齐目标：BaseStore API 语义（put/get/search/list_namespaces），namespace 元组 (user_id, "memories")。
不复刻官方物理 DDL，切换真实 Postgres 时由 AsyncPostgresStore.setup() 自动建表。

语义检索：继承 InMemoryStore 的 search 实现（mock 模式返回 namespace 下全部 items，
按 put 顺序；阶段 6 接 MockMilvus 后再做真正的向量排序）。
"""

from __future__ import annotations

import logging

from langgraph.store.memory import InMemoryStore

logger = logging.getLogger("app.infra.store.mock_postgres")


class MockPostgresStore(InMemoryStore):
    """模拟 Postgres Store。

    预置画像由 seed_data.py 在启动时写入（需要 user_id，故不在构造函数内置）。
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("MockPostgresStore 初始化（InMemoryStore + 预置画像待 seed 注入）")
