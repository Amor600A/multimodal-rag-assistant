# 切换真实后端操作文档

对应 `PROJECT_PLAN.md` 第 14.2 / 14.3 / 15 节。用于把当前全 Mock 环境切换到
PostgreSQL + Redis + Milvus + 真实 LLM/Embedding，并完成全链路联调。

操作环境：Windows + PowerShell。命令中的 `.venv\Scripts\python` 为项目自带虚拟环境。

---

## 0. 前置条件

- Docker Desktop 已安装且 `docker version` 可用（本文件所有容器操作依赖它）。
- 可用内存 >= 4GB（Milvus 三件套占用较高，8GB 更稳妥，见方案 14.4）。
- 已在项目根目录准备好 `.env`（从 `.env.example` 复制，见第 1 步）。

---

## 1. 一键起依赖服务

```powershell
docker compose -f scripts/docker-compose.yml up -d
docker compose -f scripts/docker-compose.yml ps
```

等待 `postgres`、`redis` 健康检查变为 healthy，`milvus` 状态为 running：

```powershell
docker compose -f scripts/docker-compose.yml ps --format "table {{.Service}}`t{{.Status}}"
```

说明：

- 编排含 5 个服务：`postgres`、`redis`、`etcd`、`minio`、`milvus`（Milvus standalone 必须带 etcd + minio）。
- 全部数据落在 named volumes，`down` 不丢数据，`down -v` 才清空。
- 纯 Mock 模式开发无需启动任何容器。

---

## 2. 安装真实后端依赖

```powershell
.venv\Scripts\python -m pip install asyncpg sqlalchemy alembic langgraph-checkpoint-postgres langgraph-checkpoint-redis pymilvus
```

版本策略遵循方案第 11 节（`>=x.y,<下一 major`）。安装完成后建议追加到
`requirements.txt` 的「切真实后端时安装」注释块下方，避免后续环境重建时遗漏。

---

## 3. PostgreSQL 建表（业务表 + 迁移基线）

### 3.1 建库

```powershell
docker exec -it (docker compose -f scripts/docker-compose.yml ps -q postgres) psql -U rag -d postgres -c "CREATE DATABASE rag_assistant ENCODING 'UTF8';"
```

### 3.2 业务表

用项目已有脚本建业务表（`departments` / `users` / `sessions` / `images`）：

```powershell
docker exec -i (docker compose -f scripts/docker-compose.yml ps -q postgres) psql -U rag -d rag_assistant -f - < scripts/create_db.sql
```

**不要手动创建 `checkpoints` 与 `store` 表**：前者由 langgraph 的 Checkpointer、
后者由 `AsyncPostgresStore.setup()` 在首次启动时自动建表，手动预建会与之冲突
（见 `scripts/create_db.sql` 顶部说明与方案 4.5 节）。

### 3.3 引入 Alembic（迁移基线）

1. 在项目根 `alembic init`（或手写 `alembic.ini` + `migrations/`），把 `sqlalchemy.url` 指向
   `postgresql+asyncpg://rag:rag_dev_password@localhost:5432/rag_assistant`。
2. 编写 SQLAlchemy 2.0 Declarative 模型（users / departments / sessions / images，字段与
   `scripts/create_db.sql` 一致），放在基础设施层对应仓储模块旁。
3. 生成初始 migration，并与 DDL 比对校验：

```powershell
.venv\Scripts\python -m alembic revision --autogenerate -m "init business tables"
.venv\Scripts\python -m alembic upgrade head
```

4. 后续业务表字段变更（users / sessions 等）一律走
   `.venv\Scripts\python -m alembic revision` 生成迁移脚本，**禁止手工 ALTER 线上表**。

---

## 4. Milvus

Milvus 已随 compose 启动，无需额外安装（Milvus Lite 不支持 Windows，本机必须用容器版）：

- 服务地址 `http://localhost:19530`；`9091` 为健康检查端口。
- collection 与索引由 `rag_service` 启动时 ensure（幂等），首次启动自动创建。

---

## 5. 修改 .env 开关（配置联动）

复制 `.env.example` 为 `.env` 后，按下表逐项修改：

| compose 服务 | .env 配置项 |
|---|---|
| postgres（业务表 + store） | `USER_DB_BACKEND=postgres`、`STORE_BACKEND=postgres`、`POSTGRES_DSN=postgresql+asyncpg://rag:rag_dev_password@localhost:5432/rag_assistant` |
| redis（checkpointer + token 吊销） | `CHECKPOINTER_BACKEND=redis`、`TOKEN_REVOCATION_BACKEND=redis`、`REDIS_URI=redis://localhost:6379/0` |
| milvus | `VECTOR_BACKEND=milvus`、`MILVUS_URI=http://localhost:19530` |
| 真实 LLM | `LLM_BACKEND=openai_compatible`、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`LLM_VISION_MODEL` |
| 真实 Embedding | `EMBEDDING_BACKEND=api`、`EMBEDDING_API_BASE`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_DIM`（须与 Milvus collection 维度一致） |

配置缺失时启动即报错（`app/core/config.py` 的 `_validate_backends`），不会静默降级。

### 5.1 当前实现状态（切换前必读）

配置校验与工厂分发已就绪，但**真实后端分支代码尚未实现**，当前会打印 warning 并
回退到 Mock 实现：

| 开关 | 真实取值 | 当前行为 |
|---|---|---|
| `USER_DB_BACKEND=postgres` | 需实现 PostgresUserRepository 等 | 仍走 Mock 仓储 |
| `CHECKPOINTER_BACKEND=redis/postgres` | 需实现 AsyncRedisSaver / AsyncPostgresSaver 分支 | 回退 `MockRedisSaver` |
| `STORE_BACKEND=postgres` | 需实现 AsyncPostgresStore 分支 | 回退 `MockPostgresStore` |
| `VECTOR_BACKEND=milvus` | 需实现 pymilvus 分支 | 回退 `MockMilvus` |

因此真正切换前，需先补齐上表 4 个分支的实现（各工厂 else 分支内），再执行第 5 步重启验证。

---

## 6. 重启应用并验证

```powershell
# 先停掉可能在跑的旧进程，再启动
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

启动日志应打印组件清单（`app/core/config.py::backend_summary`）为真实后端，且**不再出现
"暂未实现，fallback 到 ..." 的 warning**。

---

## 7. 全链路联调验收清单

按方案第 15 节各阶段验收点，逐项在真实后端下复验：

| # | 验收项 | 期望结果 |
|---|---|---|
| 1 | 组件清单 | 启动日志显示 postgres / redis / milvus / openai_compatible，无 fallback warning |
| 2 | 登录三账号 | 种子账号登录成功，`/api/auth/me` 返回实时部门 |
| 3 | 会话列表 | 新建会话落库 sessions 表，刷新后列表仍在 |
| 4 | SSE 流式 | 提问后逐 token 出流，最终收到 done 帧（不与 error 帧同时出现） |
| 5 | 多轮记忆 | 同一 thread 连续提问，第二问能带上文；重启服务后历史仍可回放（checkpoint 落 Redis/Postgres） |
| 6 | 长期记忆 | 对话中说"我喜欢X"，跨会话生效（偏好注入 Prompt） |
| 7 | 权限过滤 | alice（人事部）检索不到市场部文档；`perm` 表达式由服务端构造，前端不可传 |
| 8 | 引用校验 | 伪造引用编号被 output_guard 净化，前端收到 warning 帧 |
| 9 | 图片链路 | 上传返回 image_id+代理 URL；本人可访问，他人访问 404；发送携带 image_id 后 images.bound 置位 |
| 10 | 前端全流程 | 浏览器完成 登录 → 提问 → 流式输出；同一会话连点发送不产生并发写 |
| 11 | 回归 | `.venv\Scripts\python -m pytest tests -q` 全量通过（59 passed） |
| 12 | 幂等种子 | 重复执行 `scripts/seed_data.py` 不产生重复数据 |

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `docker compose up` 后 milvus 反复重启 | 内存不足（见方案 14.4）；调大 Docker 内存上限或先用 Mock 模式开发 |
| 启动报 `CHECKPOINTER_BACKEND=redis 需要设置 REDIS_URI` | `.env` 缺连接配置，按第 5 步补齐 |
| langgraph 报表不存在 | `store` / `checkpoints` 表由 langgraph 自建，检查是否误用手工建表脚本覆盖 |
| 切换后行为仍是 Mock | 真实后端分支未实现（见 5.1），检查启动日志是否出现 fallback warning |
| 中文日志/输出乱码 | 所有文件须 UTF-8；PowerShell 解码 UTF-8 JSON 会乱码，验收断言请用 `.venv\Scripts\python` 执行 |

---

## 9. 回滚到 Mock

把 `.env` 中 6 个开关改回 `.env.example` 默认值即可（`mock_postgres` / `mock_redis` /
`mock_postgres` / `mock` / `mock` / `hash`），无需改动代码；依赖容器可保留不关。
