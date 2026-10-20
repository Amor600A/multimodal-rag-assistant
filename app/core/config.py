# -*- coding: utf-8 -*-
"""全局配置（Pydantic Settings）。

所有后端专属配置声明为可空字符串默认值；由 model_validator 按 BACKEND 开关
条件校验，保证 Mock 模式零配置可启动，切真实后端时缺失必填项会明确报错。
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 应用 ----
    app_name: str = "Multimodal RAG Assistant"
    api_prefix: str = "/api"
    log_level: str = "INFO"

    # ---- JWT ----
    jwt_secret: str = "dev-only-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120        # access token 有效期（分钟）
    jwt_refresh_expire_days: int = 7     # refresh token 有效期（天）

    # ---- 六大后端开关（mock 优先，真实后端预留）----
    user_db_backend: str = "mock_postgres"      # mock_postgres | postgres
    checkpointer_backend: str = "mock_redis"    # memory | mock_redis | redis | postgres
    store_backend: str = "mock_postgres"            # in_memory | mock_postgres | postgres
    vector_backend: str = "mock"                # mock | milvus
    llm_backend: str = "mock"                   # mock | openai_compatible
    embedding_backend: str = "hash"             # hash | api
    token_revocation_backend: str = "memory"    # memory | redis

    # ---- LLM / Embedding（mock 模式下全部可留空，不校验）----
    llm_base_url: str = ""               # LLM_BACKEND=openai_compatible 时必填
    llm_api_key: str = ""                # 同上
    llm_model: str = "qwen-plus"         # 同上
    llm_vision_model: str = "qwen-vl-plus"  # 同上（vision 节点用）
    embedding_api_base: str = ""         # EMBEDDING_BACKEND=api 时必填
    embedding_api_key: str = ""          # 同上
    embedding_model: str = "text-embedding-v3"  # 同上
    embedding_dim: int = 1536            # 嵌入维度，须与向量库 collection 一致

    # ---- Agent ----
    agent_enabled_tools: str = ""        # 留空 = 纯 RAG 链路；启用：已注册工具名逗号分隔，如 retrieval,memory
    memory_to_vector_enabled: bool = True    # 非结构化记忆入向量库 personal 区
    output_guard_enabled: bool = True        # 生成侧输出护栏
    memory_extractor: str = "rule"           # 事实提取器：rule | llm
    memory_summary_every_turns: int = 10     # 每累计 N 轮生成/更新一次会话摘要

    # ---- Milvus（VECTOR_BACKEND=milvus 时必填）----
    milvus_uri: str = ""
    milvus_collection: str = "rag_knowledge"

    # ---- 真实后端连接（对应 *_BACKEND 切换时启用，mock 可留空）----
    postgres_dsn: str = ""               # postgresql+asyncpg://...（业务表 + store 共用）
    redis_uri: str = ""                  # redis://localhost:6379/0（checkpointer + token 吊销）

    # ---- 图片上传 ----
    uploads_dir: str = "uploads"         # 图片落盘根目录（按 user_id 分子目录）
    image_max_mb: int = 10               # 单图大小上限（MB）

    # 各后端开关的合法取值（防拼写错误静默失效）
    _allowed = {
        "user_db_backend": {"mock_postgres", "postgres"},
        "checkpointer_backend": {"memory", "mock_redis", "redis", "postgres"},
        "store_backend": {"in_memory", "mock_postgres", "postgres"},
        "vector_backend": {"mock", "milvus"},
        "llm_backend": {"mock", "openai_compatible"},
        "embedding_backend": {"hash", "api"},
        "token_revocation_backend": {"memory", "redis"},
        "memory_extractor": {"rule", "llm"},
    }

    @model_validator(mode="after")
    def _validate_backends(self) -> "Settings":
        """按 BACKEND 开关条件校验：mock 模式不校验连接配置，真实后端缺必填项时明确报错。"""
        errors: list[str] = []

        # 合法取值校验
        for field, allowed in self._allowed.items():
            value = getattr(self, field)
            if value not in allowed:
                errors.append(f"{field}={value!r} 非法，允许值：{sorted(allowed)}")

        # 条件必填校验（mock 模式可留空）
        if self.llm_backend == "openai_compatible":
            for f in ("llm_base_url", "llm_api_key", "llm_model"):
                if not getattr(self, f):
                    errors.append(f"LLM_BACKEND=openai_compatible 需要设置 {f.upper()}")
        if self.embedding_backend == "api":
            for f in ("embedding_api_base", "embedding_api_key", "embedding_model"):
                if not getattr(self, f):
                    errors.append(f"EMBEDDING_BACKEND=api 需要设置 {f.upper()}")
        if self.vector_backend == "milvus" and not self.milvus_uri:
            errors.append("VECTOR_BACKEND=milvus 需要设置 MILVUS_URI")
        if self.user_db_backend == "postgres" and not self.postgres_dsn:
            errors.append("USER_DB_BACKEND=postgres 需要设置 POSTGRES_DSN")
        if self.store_backend == "postgres" and not self.postgres_dsn:
            errors.append("STORE_BACKEND=postgres 需要设置 POSTGRES_DSN")
        if self.checkpointer_backend == "postgres" and not self.postgres_dsn:
            errors.append("CHECKPOINTER_BACKEND=postgres 需要设置 POSTGRES_DSN")
        if self.checkpointer_backend == "redis" and not self.redis_uri:
            errors.append("CHECKPOINTER_BACKEND=redis 需要设置 REDIS_URI")
        if self.token_revocation_backend == "redis" and not self.redis_uri:
            errors.append("TOKEN_REVOCATION_BACKEND=redis 需要设置 REDIS_URI")

        if errors:
            raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errors))
        return self

    @property
    def enabled_tools_list(self) -> list[str]:
        """解析 AGENT_ENABLED_TOOLS 逗号分隔列表；留空返回空列表（纯 RAG 链路）。

        工具名是否已在 TOOL_REGISTRY 注册的校验在 graph 构建时执行（见方案 4.3）。
        """
        return [t.strip() for t in self.agent_enabled_tools.split(",") if t.strip()]

    def backend_summary(self) -> dict[str, str]:
        """启动时打印的组件清单（见方案阶段 1 验收点）。"""
        return {
            "用户仓储": self.user_db_backend,
            "Checkpointer(短期记忆)": self.checkpointer_backend,
            "Store(长期记忆)": self.store_backend,
            "向量库": self.vector_backend,
            "LLM": self.llm_backend,
            "Embedding": self.embedding_backend,
            "Token 吊销": self.token_revocation_backend,
        }


@lru_cache
def get_settings() -> Settings:
    """全局 Settings 单例（进程内缓存）。"""
    return Settings()
