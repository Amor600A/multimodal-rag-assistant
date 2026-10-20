# -*- coding: utf-8 -*-
"""memory_service：长期记忆读写 + 事实提取 + 会话摘要。

两条异步链路（方案 6.6）：
1. 结构化事实提取 → Store（rule 提取器默认，llm 预留）
2. 对话摘要 → 向量库 personal 区（mock 模式固定模板，阶段 6 接向量库）

触发时机：SSE 流 done 后由 chat 路由 asyncio.create_task 异步触发，不阻塞主链路。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import get_settings
from app.infra.store.factory import get_store

logger = logging.getLogger("app.services.memory")


# ---- 事实模型 ----

@dataclass
class Fact:
    """从用户消息中提取的结构化事实。"""
    key: str       # Store key，如 reply_preference / name / dept / note
    value: str     # 事实值
    type: str = "preference"  # preference | identity | note


# ---- 事实提取器 ----

class FactExtractor(ABC):
    """事实提取器抽象接口。"""

    @abstractmethod
    def extract(self, user_message: str) -> list[Fact]:
        raise NotImplementedError


class RuleBasedExtractor(FactExtractor):
    """规则提取器：正则/关键词模式库匹配。

    模式集（阶段 5 实施时扩充，覆盖常见表述）：
    - `我叫X` / `我的名字是X` → name
    - `我是X部门的` / `我在X部门` / `X部门的` → dept
    - `请用X回复` / `我喜欢X风格` / `用X的方式回复` / `回复请X` → reply_preference
    - `记住：X` / `记一下X` → note
    """

    # (regex, key, type) 列表
    _PATTERNS: list[tuple[re.Pattern, str, str]] = [
        # name
        (re.compile(r"我叫([^，。,.\s]{1,20})"), "name", "identity"),
        (re.compile(r"我的名字是([^，。,.\s]{1,20})"), "name", "identity"),
        # dept
        (re.compile(r"我是([^，。,.\s]{1,10})部门的"), "dept", "identity"),
        (re.compile(r"我在([^，。,.\s]{1,10})部门"), "dept", "identity"),
        # reply_preference
        (re.compile(r"请用([^，。,.\s]{1,20})(回复|的方式)"), "reply_preference", "preference"),
        (re.compile(r"我喜欢([^，。,.\s]{1,20})风格"), "reply_preference", "preference"),
        (re.compile(r"用([^，。,.\s]{1,20})的方式回复"), "reply_preference", "preference"),
        (re.compile(r"回复请([^，。,.\s]{1,20})"), "reply_preference", "preference"),
        (re.compile(r"我喜欢([^，。,.\s]{1,20})回复"), "reply_preference", "preference"),
        # note
        (re.compile(r"记住[：:](.+?)(?:[。,.]|$)"), "note", "note"),
        (re.compile(r"记一下([^，。,.\s]{1,50})"), "note", "note"),
    ]

    def extract(self, user_message: str) -> list[Fact]:
        facts: list[Fact] = []
        seen_keys: set[str] = set()
        for pattern, key, ftype in self._PATTERNS:
            match = pattern.search(user_message)
            if match and key not in seen_keys:
                value = match.group(1).strip()
                if value:
                    facts.append(Fact(key=key, value=value, type=ftype))
                    seen_keys.add(key)
        return facts


class LLMExtractor(FactExtractor):
    """LLM 提取器（预留，阶段 5 不实现）。

    用当前 LLM 以固定 Prompt 输出 JSON 事实数组，解析失败静默跳过。
    """

    def extract(self, user_message: str) -> list[Fact]:
        logger.warning("LLMExtractor 未实现，返回空事实列表")
        return []


# ---- 记忆服务 ----

class MemoryService:
    """长期记忆服务：事实提取入库 + 偏好检索 + 会话摘要。"""

    def __init__(self) -> None:
        self._store = get_store()
        self._extractor = self._build_extractor()

    @staticmethod
    def _build_extractor() -> FactExtractor:
        backend = get_settings().memory_extractor
        if backend == "llm":
            return LLMExtractor()
        return RuleBasedExtractor()

    @staticmethod
    def _namespace(user_id: str) -> tuple[str, ...]:
        return (user_id, "memories")

    async def extract_and_store(self, user_id: str, user_message: str) -> list[Fact]:
        """从用户消息提取事实并写入 Store（同 key 覆盖）。

        提取失败只记日志，绝不抛异常影响主链路。
        """
        try:
            facts = self._extractor.extract(user_message)
        except Exception as e:
            logger.warning("事实提取异常（已跳过）: %s", e)
            return []

        for fact in facts:
            try:
                self._store.put(
                    self._namespace(user_id),
                    fact.key,
                    {"value": fact.value, "type": fact.type},
                )
                logger.info("记忆入库: user=%s key=%s value=%s", user_id, fact.key, fact.value)
            except Exception as e:
                logger.warning("Store.put 失败（已跳过）key=%s: %s", fact.key, e)
        return facts

    def get_user_facts(self, user_id: str) -> list[dict]:
        """检索用户全部结构化事实。"""
        try:
            items = self._store.search(self._namespace(user_id), limit=100)
            return [{"key": it.key, "value": it.value.get("value"), "type": it.value.get("type")} for it in items]
        except Exception as e:
            logger.warning("Store.search 失败: %s", e)
            return []

    def get_reply_preference(self, user_id: str) -> str | None:
        """获取用户回复偏好（reply_preference）。"""
        try:
            item = self._store.get(self._namespace(user_id), "reply_preference")
            if item is not None:
                return item.value.get("value")
        except Exception as e:
            logger.warning("Store.get reply_preference 失败: %s", e)
        return None

    def summarize_session(self, thread_id: str, messages: list) -> str:
        """生成会话摘要（mock 模式：固定模板拼接前 N 条消息）。

        阶段 6 接向量库后，此摘要写入 personal 区。
        """
        # mock 模式：取前 5 条消息拼接
        preview = []
        for m in messages[:5]:
            role = getattr(m, "type", "unknown")
            content = str(getattr(m, "content", ""))[:50]
            preview.append(f"{role}: {content}")
        summary = f"[会话摘要 thread={thread_id}] " + " | ".join(preview)
        logger.info("会话摘要生成: thread=%s len=%d", thread_id, len(summary))
        return summary


# 进程级单例
_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _service
    if _service is None:
        _service = MemoryService()
    return _service
