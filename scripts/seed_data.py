# -*- coding: utf-8 -*-
"""seed_data.py：预置部门、用户、会话种子数据。

执行时机（方案 8.0）：
  - Mock 后端：main.py lifespan 启动时自动调用 seed_all()（见 app/main.py）
  - 真实后端：不自动执行；`python -m scripts.seed_data` 手动运行

幂等性：
  - 部门 / 用户：存在即跳过，bcrypt 只在首次生成
  - 会话：存在即跳过（固定 thread_id，见下方常量）
  - 调用方统一走仓储抽象接口（与业务代码同路径），保证 Mock/Postgres 代码不变
"""

from __future__ import annotations

import asyncio
import logging

from app.core.security import hash_password
from app.infra.org.mock_org import get_department_repository
from app.infra.session.mock_session import get_session_repository
from app.infra.user.mock_user import get_user_repository

logger = logging.getLogger("app.seed")

# ---- 种子数据常量（密码统一走 hash_password，禁止明文）----

SEED_PASSWORDS = {
    "alice": "alice123",
    "bob": "bob123",
    "admin": "admin123",
}

SEED_DEPARTMENTS = [
    {"code": "hr", "name": "人事部"},
    {"code": "marketing", "name": "市场部"},
]

SEED_USERS = [
    {"username": "alice", "role": "user", "dept_code": "hr", "email": "alice@example.com"},
    {"username": "bob", "role": "user", "dept_code": "marketing", "email": "bob@example.com"},
    {"username": "admin", "role": "admin", "dept_code": None, "email": "admin@example.com"},
]

# 预置 thread_id（固定 UUID，方便对话快照回放，见方案阶段 4 对话快照）
SEED_SESSIONS = [
    {
        "thread_id": "seed-alice-1",
        "username": "alice",
        "title": "考勤请假咨询",
    },
    {
        "thread_id": "seed-bob-1",
        "username": "bob",
        "title": "竞品分析模板使用",
    },
]


async def _seed_checkpoint_snapshot(user_ids: dict[str, str]) -> None:
    """为 alice 的 seed-alice-1 会话生成两轮对话 checkpoint 快照。

    MockRedisSaver 会自动把 checkpoint 持久化到 data/checkpoint_snapshot.json，
    下次应用启动时直接加载，实现"重启后 mock 数据可回放"。
    """
    import os

    from langchain_core.messages import HumanMessage

    from app.agent.graph import get_compiled_graph
    from app.infra.checkpointer.mock_redis import MockRedisSaver
    from app.infra.checkpointer.memory_checkpointer import get_checkpointer

    snapshot_file = "data/checkpoint_snapshot.json"
    if os.path.exists(snapshot_file):
        logger.info("checkpoint 快照已存在，跳过生成: %s", snapshot_file)
        return

    alice_id = user_ids.get("alice")
    if not alice_id:
        logger.warning("跳过 checkpoint 快照：alice 未创建")
        return

    checkpointer = get_checkpointer()
    if not isinstance(checkpointer, MockRedisSaver):
        return  # 仅 mock_redis 需要生成快照

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": "seed-alice-1", "user_id": alice_id}}

    # 第一轮：考勤请假
    await graph.ainvoke(
        {"messages": [HumanMessage(content="考勤请假流程是怎样的？")], "user_id": alice_id},
        config,
    )
    # 第二轮：追问病假证明（验证多轮记忆）
    await graph.ainvoke(
        {"messages": [HumanMessage(content="那病假需要什么证明？")], "user_id": alice_id},
        config,
    )
    logger.info("checkpoint 快照已生成: %s（alice 两轮对话）", snapshot_file)


def _seed_user_profiles(user_ids: dict[str, str]) -> None:
    """向 Store 写入预置用户画像/偏好。

    与提取写入同一 namespace (user_id, "memories")，检索路径一致。
    覆盖写（put 语义天然幂等），重复执行安全。
    """
    from app.infra.store.factory import get_store

    store = get_store()
    profiles = {
        "alice": {"reply_preference": "中文简洁回复"},
        "bob": {"reply_preference": "要点列表式回复"},
    }
    for username, prefs in profiles.items():
        uid = user_ids.get(username)
        if not uid:
            continue
        for key, value in prefs.items():
            store.put(
                (uid, "memories"),
                key,
                {"value": value, "type": "preference"},
            )
        logger.info("seed 画像: %s → %s", username, prefs)


def _seed_knowledge(dept_ids: dict[str, str]) -> None:
    """注入知识库种子语料（方案 8.2：public + 部门文档）。

    查询词需包含语料关键词（hash 嵌入仅词面重叠，无语义泛化）。
    """
    from app.services.rag_service import get_rag_service

    records = [
        # public 域
        {"id": "pub-company-intro", "text": "公司简介：本公司成立于 2015 年，专注于企业级 SaaS 服务，总部位于北京。", "scope": "domain", "visibility": "public"},
        {"id": "pub-travel-reimburse", "text": "差旅报销制度：员工出差需提前在 OA 提交申请，住宿标准一线城市 500 元/晚，机票经济舱。", "scope": "domain", "visibility": "public"},
        {"id": "pub-info-security", "text": "信息安全规范：禁止将客户数据导出至个人设备，密码每 90 天更换一次，启用双因素认证。", "scope": "domain", "visibility": "public"},
        # 人事部
        {"id": "hr-attendance", "text": "考勤与请假制度：正式员工每月 2 天带薪病假，事假需提前 3 个工作日申请，迟到 30 分钟扣半日薪。", "scope": "domain", "visibility": "dept", "dept_id": dept_ids.get("hr")},
        {"id": "hr-recruit", "text": "招聘流程规范：简历筛选 → 初试 → 复试 → HR 面试 → 背调 → offer，全流程不超过 15 个工作日。", "scope": "domain", "visibility": "dept", "dept_id": dept_ids.get("hr")},
        # 市场部
        {"id": "mkt-brand", "text": "品牌视觉手册：Logo 使用规范见附件，主色调为品牌蓝 #1677FF，禁止擅自修改配色方案。", "scope": "domain", "visibility": "dept", "dept_id": dept_ids.get("marketing")},
        {"id": "mkt-approval", "text": "市场活动审批流程：活动预算 5 万以下部门经理审批，5-20 万需 VP 审批，20 万以上需 CEO 审批。", "scope": "domain", "visibility": "dept", "dept_id": dept_ids.get("marketing")},
    ]
    get_rag_service().ingest(records)
    logger.info("seed 知识语料: %d 条", len(records))


async def seed_all() -> None:
    """向全部 Mock 仓储注入种子数据（幂等，重复执行安全）。"""
    dept_repo = get_department_repository()
    user_repo = get_user_repository()
    session_repo = get_session_repository()

    # 1. 部门
    dept_ids: dict[str, str] = {}
    for d in SEED_DEPARTMENTS:
        existing = await dept_repo.get_by_code(d["code"])
        if existing is None:
            created = await dept_repo.create(code=d["code"], name=d["name"])
            logger.info("seed 部门: %s (id=%s)", d["name"], created.id)
            dept_ids[d["code"]] = created.id
        else:
            logger.debug("seed 部门已存在: %s", d["name"])
            dept_ids[d["code"]] = existing.id

    # 2. 用户
    user_ids: dict[str, str] = {}
    for u in SEED_USERS:
        existing = await user_repo.get_by_username(u["username"])
        if existing is None:
            dept_id = dept_ids.get(u["dept_code"]) if u["dept_code"] else None
            hashed = hash_password(SEED_PASSWORDS[u["username"]])
            created = await user_repo.create(
                username=u["username"],
                password_hash=hashed,
                role=u["role"],
                dept_id=dept_id,
                email=u["email"],
            )
            logger.info("seed 用户: %s (role=%s, dept=%s, id=%s)", u["username"], u["role"], u["dept_code"], created.id)
            user_ids[u["username"]] = created.id
        else:
            logger.debug("seed 用户已存在: %s", u["username"])
            user_ids[u["username"]] = existing.id

    # 3. 会话元数据
    for s in SEED_SESSIONS:
        user_id = user_ids.get(s["username"])
        if user_id is None:
            logger.warning("跳过 seed 会话 %s：用户 %s 未创建", s["thread_id"], s["username"])
            continue
        await session_repo.create(
            thread_id=s["thread_id"],
            user_id=user_id,
            title=s["title"],
        )
        logger.debug("seed 会话: %s", s["thread_id"])

    # 4. Checkpointer 预置快照（仅 mock_redis 模式：跑两轮 alice 对话生成 checkpoint 落盘）
    from app.core.config import get_settings

    if get_settings().checkpointer_backend == "mock_redis":
        await _seed_checkpoint_snapshot(user_ids)

    # 5. Store 预置画像（alice/bob 的回复偏好）
    _seed_user_profiles(user_ids)

    # 6. 知识库种子语料（public + dept）
    _seed_knowledge(dept_ids)

    logger.info("种子数据注入完成（%d 部门, %d 用户, %d 会话）", len(dept_ids), len(user_ids), len(SEED_SESSIONS))


# 允许 `python -m scripts.seed_data` 手动运行（真实后端场景）
if __name__ == "__main__":
    asyncio.run(seed_all())
