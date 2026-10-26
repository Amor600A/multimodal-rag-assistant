# -*- coding: utf-8 -*-
"""Rerank 精排：DashScope TextReRank（qwen3-rerank）。

初检向量召回 top-50 → Rerank 重打分 → 取 top-5，兼顾召回率（初检放宽）
和精确率（精排筛选）。

统一接口：rerank(query, documents) -> list[tuple[原索引, rerank_score]]。
documents 是字符串列表，返回结果按 rerank_score 降序排列。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def rerank(query: str, documents: List[str]) -> List[Tuple[int, float]]:
    """调用 DashScope TextReRank 对候选文档重排序。

    参数:
        query: 查询文本
        documents: 候选文档文本列表
    返回:
        [(原始索引, rerank_score), ...] 按 score 降序
    """
    import dashscope

    settings = get_settings()
    dashscope.api_key = settings.embedding_api_key

    if len(documents) <= 1:
        return [(0, 1.0)] if documents else []

    resp = dashscope.TextReRank.call(
        model=settings.rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),  # 返回全部候选，由调用方按需截取
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Rerank API 失败: status={resp.status_code}, message={resp.message}"
        )

    results = []
    for item in resp.output["results"]:
        results.append((int(item["index"]), float(item["relevance_score"])))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
