# -*- coding: utf-8 -*-
"""查询 Milvus Lite 知识库脚本。

流程：连接本地知识库 -> 生成查询向量 -> 按权限 expr 过滤 + 相似度检索 -> 打印结果。

改造说明（相对最初参考脚本）：
1. 检索改用项目统一嵌入器（与建库脚本共用同一向量空间）
2. 连接本地 Milvus Lite 数据文件，不再依赖 langchain_milvus
3. 可选传入 --user-id / --dept-id 走项目权限 expr，与线上检索口径一致；
   不传则不做过滤，用于纯调试查看全库内容

用法：
    python -m app.infra.vectordb.query_knowledge_base "请假流程是什么"
    python -m app.infra.vectordb.query_knowledge_base "竞品分析怎么做" --user-id alice --dept-id <部门ID>
    python -m app.infra.vectordb.query_knowledge_base "考勤" --limit 5 --db-path data/milvus/rag_knowledge.db

注意：Milvus Lite 为单文件独占，运行本脚本前请先停止后端服务，避免文件被占用。
"""

from __future__ import annotations

import argparse


def load_knowledge_base(load_path: str):
    """连接已持久化的 Milvus Lite 知识库（保留原函数名）。

    参数:
        load_path: Milvus Lite 数据文件路径
    返回:
        MilvusLite 实例
    """
    from app.core.config import get_settings
    from app.infra.vectordb.milvus_lite import MilvusLite

    settings = get_settings()
    db = MilvusLite(
        db_path=load_path,
        collection_name=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    print(f"向量数据库已从 {load_path} 加载，当前记录数: {db.count()}")
    print(f"嵌入后端: {settings.embedding_backend}（model={settings.embedding_model}）")
    return db


def build_expr(user_id: str | None, dept_id: str | None) -> str:
    """按项目权限规则构造 expr；未提供用户身份时返回空串（不过滤）。"""
    if not user_id:
        return ""

    from app.core.deps import CurrentUser
    from app.services.rag_service import RAGService

    user = CurrentUser(
        user_id=user_id,
        username=user_id,
        role="user",
        dept_id=dept_id,
    )
    return RAGService._build_permission_expr(user)


def main() -> None:
    parser = argparse.ArgumentParser(description="查询 Milvus Lite 知识库")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--user-id", default=None, help="按该用户身份做权限过滤")
    parser.add_argument("--dept-id", default=None, help="该用户所属部门")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="置信度过滤阈值（0~1 的 COSINE 相似度，如 0.6），低于该值的召回不展示",
    )
    parser.add_argument("--db-path", default=None, help="覆盖 MILVUS_DB_PATH")
    args = parser.parse_args()

    if args.min_score is not None and not 0 <= args.min_score <= 1:
        raise SystemExit("--min-score 取值范围为 0~1")

    from app.core.config import get_settings

    settings = get_settings()
    db_path = args.db_path or settings.milvus_db_path

    kb = load_knowledge_base(db_path)
    expr = build_expr(args.user_id, args.dept_id)
    print(f"\n问题: {args.query}")
    print(f"权限 expr: {expr or '（无过滤，查看全库）'}\n")

    hits = kb.search(expr, args.query, limit=args.limit)

    if args.min_score is not None:
        before = len(hits)
        hits = [h for h in hits if h["score"] >= args.min_score]
        print(f"置信度过滤: score >= {args.min_score:.2f}，{before} 条召回保留 {len(hits)} 条\n")

    print("========== 检索到的文本块 ==========")
    for i, hit in enumerate(hits, start=1):
        print(
            f"\n--- 第 {i} 个文本块 (score: {hit['score']:.4f}, 来源: {hit['source']}, "
            f"类型: {hit['file_type'] or '-'}) ---"
        )
        print(f"可见性: {hit['visibility']}  归属部门: {hit['dept_id'] or '-'}")
        # score 为 COSINE 相似度（-1~1，越大越相似），按百分比展示置信度
        print(f"置信度: {hit['score'] * 100:.2f}%")
        print(f"内容:\n{hit['text']}")
    print("\n========== 检索到的文本块打印结束 ==========")

    if not hits:
        print("未命中任何记录，请检查权限过滤条件或先运行建库脚本。")


if __name__ == "__main__":
    main()
