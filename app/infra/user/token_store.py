# -*- coding: utf-8 -*-
"""TokenStore：refresh token jti 吊销（登出时写入）。

当前阶段：进程内字典模拟（key=jti, value=exp_ts）。过期自动清理在下次
调用时惰性执行。切真实后端时改为 Redis SET + EXPIRE，factory 分支即可。
"""

from __future__ import annotations

import time


class MockTokenStore:
    def __init__(self) -> None:
        # jti -> exp_ts（refresh token 过期时间戳）
        self._revoked: dict[str, int] = {}

    def _gc(self) -> None:
        """惰性清理过期吊销记录。"""
        now = int(time.time())
        expired = [k for k, v in self._revoked.items() if v < now]
        for k in expired:
            del self._revoked[k]

    def revoke(self, jti: str, exp_ts: int) -> None:
        self._gc()
        self._revoked[jti] = exp_ts

    def is_revoked(self, jti: str) -> bool:
        self._gc()
        return jti in self._revoked

    def clear(self) -> None:
        self._revoked.clear()


_token_store = MockTokenStore()


def get_token_store() -> MockTokenStore:
    return _token_store
