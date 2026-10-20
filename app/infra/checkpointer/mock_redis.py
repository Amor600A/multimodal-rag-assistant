# -*- coding: utf-8 -*-
"""MockRedisSaver：模拟 Redis Checkpointer，带磁盘持久化 + 预置快照。

设计要点（对应方案 6.4 / 8.3）：
1. 继承 InMemorySaver，复用 get/list/put_writes 等全部官方实现，零行为偏差
2. 覆盖 put/aput/put_writes/aput_writes，每次写入后把 storage/writes/blobs
   序列化（base64）落盘到 snapshot_path，重启进程后自动加载 → 验收点"重启回放"
3. 预置快照：snapshot_path 不存在时从内置 ALICE_SNAPSHOT 加载，演示"有历史会话"
4. user_id 不进 checkpointer 索引（仅 thread_id 维度），会话归属由路由层校验
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections import defaultdict
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger("app.infra.checkpointer.mock_redis")


def _bytes_to_jsonable(obj: Any) -> Any:
    """递归把含 bytes 的结构转成 JSON 可序列化结构。"""
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): _bytes_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_bytes_to_jsonable(v) for v in obj]
    return obj


def _jsonable_to_bytes(obj: Any) -> Any:
    """反向还原。"""
    if isinstance(obj, dict):
        if set(obj.keys()) == {"__bytes__"}:
            return base64.b64decode(obj["__bytes__"])
        return {k: _jsonable_to_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable_to_bytes(v) for v in obj]
    return obj


class MockRedisSaver(InMemorySaver):
    """模拟 Redis 的 checkpointer：内存字典 + 磁盘持久化 + 预置快照。"""

    def __init__(self, snapshot_path: str = "data/checkpoint_snapshot.json") -> None:
        super().__init__()
        self._snapshot_path = snapshot_path
        self._load_or_seed()

    # ---------- 持久化 ----------

    def _load_or_seed(self) -> None:
        """启动时从磁盘加载；磁盘无快照则从内置 seed 初始化。"""
        if os.path.exists(self._snapshot_path):
            try:
                with open(self._snapshot_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._restore(raw)
                logger.info("MockRedisSaver 从 %s 加载 checkpoint 快照", self._snapshot_path)
                return
            except Exception as e:
                logger.warning("MockRedisSaver 快照加载失败，使用 seed 初始化: %s", e)

        # 无快照文件：从内置 seed 初始化
        seed = _build_alice_seed_snapshot()
        self._restore(seed)
        self._persist()
        logger.info("MockRedisSaver 使用内置 alice 快照初始化")

    def _restore(self, raw: dict) -> None:
        """把 JSON 结构还原为 InMemorySaver 的 storage/writes/blobs。"""
        storage_raw = _jsonable_to_bytes(raw.get("storage", {}))
        writes_raw = _jsonable_to_bytes(raw.get("writes", {}))
        blobs_raw = _jsonable_to_bytes(raw.get("blobs", {}))

        # 还原为 defaultdict 结构
        restored_storage: defaultdict = defaultdict(lambda: defaultdict(dict))
        for tid, ns_dict in storage_raw.items():
            for ns, cp_dict in ns_dict.items():
                for cid, val in cp_dict.items():
                    # val = [checkpoint_tuple, metadata_tuple, parent_id]
                    cp_tup = tuple(val[0]) if isinstance(val[0], list) else val[0]
                    meta_tup = tuple(val[1]) if isinstance(val[1], list) else val[1]
                    parent = val[2]
                    restored_storage[tid][ns][cid] = (cp_tup, meta_tup, parent)

        restored_writes: defaultdict = defaultdict(dict)
        for key_str, val in writes_raw.items():
            # key 是 "tid|ns|cid" 字符串
            tid, ns, cid = key_str.split("|", 2)
            inner: dict = {}
            for tk, wv in val.items():
                task_id, idx = tk.rsplit("|", 1)
                inner[(task_id, int(idx))] = tuple(wv)
            restored_writes[(tid, ns, cid)] = inner

        restored_blobs: dict = {}
        for key_str, val in blobs_raw.items():
            parts = key_str.split("|", 3)
            if len(parts) == 4:
                tid, ns, ch, ver = parts
                restored_blobs[(tid, ns, ch, ver)] = tuple(val)

        self.storage = restored_storage
        self.writes = restored_writes
        self.blobs = restored_blobs

    def _persist(self) -> None:
        """把 storage/writes/blobs 序列化落盘。"""
        os.makedirs(os.path.dirname(self._snapshot_path) or ".", exist_ok=True)
        # 把 (thread_id, ns, cid) 等 tuple key 转成字符串
        storage_serializable = {}
        for tid, ns_dict in self.storage.items():
            storage_serializable[tid] = {}
            for ns, cp_dict in ns_dict.items():
                storage_serializable[tid][ns] = {}
                for cid, val in cp_dict.items():
                    storage_serializable[tid][ns][cid] = _bytes_to_jsonable(list(val))

        writes_serializable = {}
        for (tid, ns, cid), inner in self.writes.items():
            key = f"{tid}|{ns}|{cid}"
            writes_serializable[key] = {}
            for (task_id, idx), wv in inner.items():
                writes_serializable[key][f"{task_id}|{idx}"] = _bytes_to_jsonable(list(wv))

        blobs_serializable = {}
        for (tid, ns, ch, ver), val in self.blobs.items():
            key = f"{tid}|{ns}|{ch}|{ver}"
            blobs_serializable[key] = _bytes_to_jsonable(list(val))

        raw = {
            "storage": storage_serializable,
            "writes": writes_serializable,
            "blobs": blobs_serializable,
        }
        with open(self._snapshot_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)

    # ---------- 覆盖写入方法，每次写入后落盘 ----------

    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        self._persist()
        return result

    async def aput(self, config, checkpoint, metadata, new_versions):
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self._persist()
        return result

    def put_writes(self, config, writes, task_id, task_path=""):
        super().put_writes(config, writes, task_id, task_path)
        self._persist()

    async def aput_writes(self, config, writes, task_id, task_path=""):
        await super().aput_writes(config, writes, task_id, task_path)
        self._persist()


def _build_alice_seed_snapshot() -> dict:
    """内置 alice 历史会话快照（空 storage，首次对话后自动填充）。

    预置多轮对话快照通过 seed_data.py 在应用启动时主动调用 graph 生成；
    此处返回空结构作为 fallback（避免首次启动崩溃）。
    """
    return {"storage": {}, "writes": {}, "blobs": {}}
