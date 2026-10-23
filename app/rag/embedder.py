# -*- coding: utf-8 -*-
"""嵌入器：哈希嵌入器（离线）、OpenAI 兼容 API 嵌入器、DashScope 多模态嵌入器。

EMBEDDING_BACKEND=hash        ：HashEmbedder，确定性伪向量，离线可测
EMBEDDING_BACKEND=api         ：ApiEmbedder，走 OpenAI 兼容 /embeddings 接口
                                当前对接 DashScope text-embedding-v1（1536 维）
EMBEDDING_BACKEND=dashscope_mm：MultimodalEmbedder，走 DashScope 原生 SDK
                                multimodal-embedding-v1（固定 1024 维），
                                文本/图片/视频处于同一语义空间，支持跨模态检索

统一接口约定：embed(text) -> np.ndarray[float32]，L2 归一化，长度 = embedding_dim。
MultimodalEmbedder 额外提供 embed_image / embed_video，产出同空间向量。
建库与检索必须共用同一个嵌入器实例，否则向量空间不一致，检索结果无意义。
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("app.rag.embedder")

# HashEmbedder 无参构造时的默认维度，仅供单元测试等独立使用场景；
# 生产链路中 get_embedder() 始终显式传入 settings.embedding_dim，该默认值不生效
_DIM = 1536
# 中文 2-gram：连续两个汉字；英文/数字按空格切
_CN_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """确定性哈希嵌入器。

    维度由构造参数 dim 决定：生产链路中由 get_embedder() 传入
    settings.embedding_dim（配置项 EMBEDDING_DIM，默认 1024），并非固定 1536；
    无参构造时的默认值 _DIM=1536 仅供单元测试等独立使用场景。
    """

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
        """生成 self.dim 维 L2 归一化 float32 向量。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big", signed=False) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)


class ApiEmbedder:
    """OpenAI 兼容嵌入接口嵌入器。

    对接 DashScope 兼容模式时：
      EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
      EMBEDDING_MODEL=text-embedding-v1
    维度由模型决定（v1 为 1536），必须与 EMBEDDING_DIM 及向量库 collection 一致。
    """

    def __init__(self, base_url: str, api_key: str, model: str, dim: int) -> None:
        from openai import OpenAI

        self.dim = dim
        self._model = model
        # 同步客户端：与 rag_service 的同步接口保持一致
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        """调用嵌入接口，返回 L2 归一化 float32 向量。"""
        resp = self._client.embeddings.create(model=self._model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        # 维度不一致直接报错，不做降级（否则写入的向量无法与 collection 匹配）
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"嵌入维度不匹配：模型 {self._model} 返回 {vec.shape[0]} 维，"
                f"配置 EMBEDDING_DIM={self.dim}"
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)


class MultimodalEmbedder:
    """DashScope 多模态嵌入器（multimodal-embedding-v1）。

    文本、图片、视频统一映射到同一个 1024 维语义空间，支持以文搜图、以图搜图。
    兼容模式 /embeddings 不支持该模型，必须走 dashscope 原生 SDK 的
    MultiModalEmbedding 接口（实测文本/图片均返回 1024 维）。

    注意：multimodal-embedding-v1 不支持 dimension 参数，维度固定 1024，
    配置 EMBEDDING_DIM 必须为 1024，否则与向量库 collection 不匹配。
    """

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        from dashscope import MultiModalEmbedding

        self.dim = dim
        self._model = model
        self._api = MultiModalEmbedding
        # 显式传 key，不依赖进程环境变量 DASHSCOPE_API_KEY
        import dashscope

        dashscope.api_key = api_key

    def _call(self, item: dict) -> np.ndarray:
        """调用多模态嵌入接口，返回 L2 归一化 float32 向量。"""
        resp = self._api.call(model=self._model, input=[item])
        if resp.status_code != 200:
            raise RuntimeError(
                f"多模态嵌入失败：status={resp.status_code} code={resp.code} msg={resp.message}"
            )
        vec = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"嵌入维度不匹配：模型 {self._model} 返回 {vec.shape[0]} 维，"
                f"配置 EMBEDDING_DIM={self.dim}"
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        """文本向量（与图片/视频同空间，用于检索文本、图片、视频）。"""
        return self._call({"text": text})

    def embed_image(self, image_path: str) -> np.ndarray:
        """图片向量：读本地文件转 Base64 data URI（支持 jpg/jpeg/png/bmp）。"""
        import base64
        import os

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        return self._call({"image": f"data:image/{ext};base64,{image_b64}"})

    def embed_video(self, video_url: str) -> np.ndarray:
        """视频向量：仅接受公网可访问的 http(s) URL（接口不支持本地视频路径）。"""
        return self._call({"video": video_url})


_embedder = None


def get_embedder():
    """进程级嵌入器单例：入库与检索共用同一实例，保证向量空间一致。"""
    global _embedder
    if _embedder is not None:
        return _embedder

    settings = get_settings()
    if settings.embedding_backend == "api":
        _embedder = ApiEmbedder(
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
        logger.info(
            "嵌入器已初始化：api（model=%s dim=%d）",
            settings.embedding_model,
            settings.embedding_dim,
        )
    elif settings.embedding_backend == "dashscope_mm":
        _embedder = MultimodalEmbedder(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
        logger.info(
            "嵌入器已初始化：dashscope_mm（model=%s dim=%d）",
            settings.embedding_model,
            settings.embedding_dim,
        )
    else:
        _embedder = HashEmbedder(dim=settings.embedding_dim)
        logger.info("嵌入器已初始化：hash（dim=%d）", settings.embedding_dim)
    return _embedder
