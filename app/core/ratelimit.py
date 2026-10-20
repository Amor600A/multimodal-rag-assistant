# -*- coding: utf-8 -*-
"""简易内存限流（per key，滑动窗口）。

阶段 2 auth 路由已在用，阶段 3 chat 路由复用。
多实例部署时切 Redis / slowapi 集中式限流即可，调用方代码零改动。
"""

from __future__ import annotations

import logging
import time

from fastapi import HTTPException

logger = logging.getLogger("app.core.ratelimit")

# 计数桶：_buckets[key] = [timestamp1, timestamp2, ...]
# key 形如 "chat:ip=1.2.3.4"
_buckets: dict[str, list[float]] = {}


def _audit_rate_limited(bucket_key: str, remaining: int) -> None:
    from app.core.logger import audit

    audit("rate_limited", detail=f"bucket={bucket_key} remaining={remaining}")


def check_rate_limit(bucket_key: str, limit: int = 5, window: float = 60.0) -> None:
    """滑动窗口限流。

    Args:
        bucket_key: 唯一键，如 "login:ip=1.2.3.4"、"chat:user=alice"
        limit: 时间窗口内允许的最大次数
        window: 窗口大小（秒）

    Raises:
        HTTPException(429): 超过限流阈值
    """
    now = time.time()
    timestamps = _buckets.get(bucket_key, [])
    # 丢弃过期时间戳
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= limit:
        _buckets[bucket_key] = timestamps
        _audit_rate_limited(bucket_key, 0)
        raise HTTPException(status_code=429, detail="请求过于频繁")
    timestamps.append(now)
    _buckets[bucket_key] = timestamps


# 兼容 auth.py 的旧调用方式（按 IP 固定 5/min）
def _check_rate_limit(ip: str) -> None:
    """auth 路由用的简易版本：固定 5/min，按 IP 桶。"""
    check_rate_limit(f"auth:ip={ip}", limit=5, window=60.0)
