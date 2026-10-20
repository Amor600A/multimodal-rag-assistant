# -*- coding: utf-8 -*-
"""哈希嵌入器（EMBEDDING_BACKEND=hash）。

确定性伪向量：同文本恒得同向量，跨进程结果一致，无嵌入 API 时检索/隔离逻辑可测。

算法（方案 8.4）：
1. 分词：中文按 2-gram、英文/数字按空格，全部小写
2. 每个 token 取 sha256 前 8 字节转无符号整数
3. 整数 % 1536 映射到维度桶，对应桶累加 1
4. L2 归一化输出 float32 向量

红线：禁止使用 Python 内置 hash()（受 PYTHONHASHSEED 影响）。
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_DIM = 1536
# 中文 2-gram：连续两个汉字；英文/数字按空格切
_CN_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """确定性哈希嵌入器。"""

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens: list[str] = []
        # 中文 2-gram
        cn_chars = _CN_PATTERN.findall(text)
        for i in range(len(cn_chars) - 1):
            tokens.append(cn_chars[i] + cn_chars[i + 1])
        # 英文/数字单词
        tokens.extend(_WORD_PATTERN.findall(text))
        return tokens

    def embed(self, text: str) -> np.ndarray:
        """生成 1536 维 L2 归一化 float32 向量。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big", signed=False) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
