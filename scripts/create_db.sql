-- ============================================================
-- 多模态垂直领域 RAG 助手 — 数据库初始化脚本（PostgreSQL 13+）
--
-- 说明：
--   1. 本脚本仅创建业务侧数据表：departments / users / sessions
--   2. checkpoints（短期记忆）与 store（长期记忆）由 langgraph 的
--      AsyncPostgresSaver / AsyncPostgresStore 在首次启动时自动建表，
--      无需在此手动创建（逻辑结构参考 PROJECT_PLAN.md 第 5 节）
--   3. 向量数据存于 Milvus，不占用 PostgreSQL 表
--
-- 执行方式：
--   1) 先创建数据库：CREATE DATABASE rag_assistant ENCODING 'UTF8';
--   2) 再执行本脚本：psql -U postgres -d rag_assistant -f scripts/create_db.sql
-- ============================================================

-- ------------------------------------------------------------
-- 0.（可选）长期记忆表概念参考 — 禁止手动创建
-- 生产环境由 AsyncPostgresStore.setup() 自动建表并管理：
--   官方物理结构与下述参考不同（namespace 为数组类型、含迁移版本
--   记录表），且随 langgraph 版本演进，手动预建可能与之冲突。
-- 下述 DDL 仅用于理解概念或校验 MockPostgresStore 的 API 语义。
-- ------------------------------------------------------------
-- CREATE EXTENSION IF NOT EXISTS vector;
-- CREATE TABLE store (
--     prefix     TEXT NOT NULL,              -- 命名空间路径，如 '{user_id}/memories'，按用户隔离
--     key        TEXT NOT NULL,              -- 记忆条目键
--     value      JSONB NOT NULL,             -- 结构化记忆内容（事实/偏好/画像）
--     embedding  vector(1536),               -- 语义检索向量（供 semantic_search）
--     created_at TIMESTAMPTZ DEFAULT now(),
--     updated_at TIMESTAMPTZ DEFAULT now(),
--     PRIMARY KEY (prefix, key)
-- );
-- CREATE INDEX idx_store_prefix    ON store (prefix);
-- CREATE INDEX idx_store_embedding ON store USING hnsw (embedding vector_cosine_ops);

-- ------------------------------------------------------------
-- 1. departments（部门表）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS departments (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(64)  UNIQUE NOT NULL,   -- 部门名称，如 人事部/市场部
    code       VARCHAR(32)  UNIQUE NOT NULL,   -- 部门编码，如 hr/marketing
    parent_id  UUID REFERENCES departments(id), -- 上级部门（预留层级）
    created_at TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_departments_parent ON departments (parent_id);

-- ------------------------------------------------------------
-- 2. users（用户表）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(64)  UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,        -- bcrypt 哈希，不存明文
    email         VARCHAR(128) UNIQUE,
    role          VARCHAR(16)  DEFAULT 'user',  -- user / admin
    dept_id       UUID REFERENCES departments(id),
    is_active     BOOLEAN      DEFAULT TRUE,
    last_login_at TIMESTAMPTZ,                  -- 最近登录时间（update_last_login 使用）
    created_at    TIMESTAMPTZ  DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_dept ON users (dept_id);

-- ------------------------------------------------------------
-- 3. sessions（会话元数据表）
-- Checkpointer 只存对话状态，无法按用户枚举会话列表，故需本表。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    thread_id  UUID PRIMARY KEY,               -- 与 langgraph checkpointer 的 thread_id 一致
    user_id    UUID REFERENCES users(id),
    title      VARCHAR(256),                   -- 会话标题（首句截断 20 字）
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);

-- ------------------------------------------------------------
-- 4. images（图片元数据表）
-- 记录上传图片元数据，配合代理端点 /api/knowledge/images/{image_id}
-- 做权限校验；图片文件本体存本地 uploads/ 目录，不入库。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS images (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- image_id，UUIDv4 不可猜测
    user_id    UUID REFERENCES users(id),
    mime       VARCHAR(32) NOT NULL,           -- image/png 等（白名单校验后入库）
    file_path  VARCHAR(256) NOT NULL,          -- 相对路径 {user_id}/{image_id}.{ext}
    bound      BOOLEAN DEFAULT FALSE,          -- 是否已被会话消息实际引用（孤儿清理判定依据，见方案 7.1 待办）
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_images_user ON images (user_id);
