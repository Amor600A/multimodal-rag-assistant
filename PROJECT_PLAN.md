# 多模态垂直领域 RAG 助手 Agent — 项目方案

> 版本：v1.0
> 日期：2026-10-19
> 状态：方案评审稿（未开始编码）

---

## 1. 项目概述

### 1.1 目标

构建一个多用户、多模态、垂直领域的 RAG 智能助手 Agent，以企业内部知识助手为初始场景（人事/市场两个部门模拟数据），具备：

- 异步流式对话（LangGraph + SSE）
- 短期记忆（会话内多轮）与长期记忆（跨会话用户画像/偏好）
- RAG 知识检索（领域知识 + 个人知识双域召回）
- 知识权限隔离（公开/部门/个人三级可见性）
- 图片输入的多模态理解
- 工具调用（Tool）接口预留
- 全部底层组件可按配置切换（Mock 优先，真实后端预留）

### 1.2 核心需求映射

| 需求 | 方案落点 |
|---|---|
| 多文件项目结构（router/service/agent 分层） | `app/api` / `app/services` / `app/agent` 三层 |
| 异步版 | `graph.astream` + FastAPI 异步路由 |
| JWT 认证中间件 | `app/core/security.py` + `core/deps.py` 依赖注入 |
| 前端页面（SSE 流式） | `web/` 目录，fetch 流式解析 |
| 多用户 HTTP 服务 | FastAPI + 内存用户仓储（Mock） |
| Checkpointer：MemorySaver 起步，预留 Redis | `infra/checkpointer` 工厂 + MockRedisSaver |
| Store：InMemoryStore 起步，预留 Postgres | `infra/store` 工厂 + MockPostgresStore |
| 向量库 Milvus（暂不装服务端） | `infra/vectordb` 工厂 + MockMilvus（numpy） |
| Tool 接口预留 | `agent/tools` 注册表 + ReAct 条件边预留 |
| 用户信息存储预留 | `infra/user` / `infra/org` 工厂，生产用 PostgreSQL |

---

## 2. 整体架构

### 2.1 分层结构

![整体架构图](docs/architecture.svg)

### 2.2 核心请求链路（SSE 对话）

```
1. 前端 fetch POST /api/chat/stream（带 Authorization: Bearer <JWT>）
2. JWT 中间件校验 → CurrentUser(user_id, role) → 实时查库补全 dept_id
3. chat_service 组装 graph 输入与 config：
   config = {"thread_id": 会话ID, "user_id": 用户ID}
4. graph.astream(input, config, stream_mode="messages") 异步迭代
5. retrieve 节点双路召回：
   - domain 区：expr = 部门/公开可见性过滤
   - personal 区：expr = owner_user_id 过滤
   - 合并去重排序，来源标注进 Prompt
6. generate 节点流式生成 → output_guard 校验引用/合规
7. 逐 token 组装 SSE 帧yield：data: {"type":"token","content":"..."}
8. 流结束发送 data: {"type":"done"}，检查点由 Checkpointer 自动落库
```

---

## 3. 项目目录结构

```
multimodal-rag-assistant/
├── PROJECT_PLAN.md                  # 本方案文档
├── docs/
│   └── architecture.svg             # 分层架构图（2.1 节引用）
├── requirements.txt                 # 依赖清单
├── .env.example                     # 配置模板（真实 .env 不入库）
├── .gitignore                       # 排除 .env、uploads/、__pycache__ 等
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 入口：挂路由/静态资源/启动初始化（lifespan 调 seed_all 注入种子，见 8.0）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic Settings：全部后端开关与模型配置
│   │   ├── logger.py                # logging 统一配置（分层）+ 审计 logger（app.audit，JSON 行）
│   │   ├── security.py              # JWT 签发/校验、bcrypt 密码哈希
│   │   ├── ratelimit.py             # slowapi Limiter：login/register/refresh 按 IP 限流（见 4.1）
│   │   └── deps.py                  # 依赖注入：get_current_user、各 service 单例
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py                # 总路由聚合（api_prefix=/api）
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── auth.py              # 注册 / 登录 / 当前用户信息
│   │       ├── chat.py              # SSE 流式对话（核心路由）
│   │       ├── session.py           # 会话列表 / 新建 / 删除
│   │       ├── knowledge.py         # 文档、图片上传入库（带权限打标）
│   │       ├── dept.py              # 部门列表（admin CRUD 预留）
│   │       └── health.py            # 健康检查
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py          # 注册/登录/签发 token（走用户仓储）
│   │   ├── chat_service.py          # 组装 graph → astream → 产出 SSE 帧
│   │   ├── session_service.py       # 会话元数据管理（走 sessions 仓储）
│   │   ├── memory_service.py        # 短期/长期记忆读写、事实提取器（rule/llm，见 6.6）、摘要生成
│   │   ├── rag_service.py           # 入库 pipeline、权限 expr 构造、双路召回
│   │   └── dept_service.py          # 部门查询
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py                 # 异步 StateGraph 编排 + ReAct 条件边预留
│   │   ├── state.py                 # TypedDict 状态定义
│   │   ├── prompts.py               # 垂直领域 Prompt 模板（含防注入声明）
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── retrieve.py          # 双域召回节点
│   │   │   ├── vision.py            # 图片理解节点（多模态）
│   │   │   ├── generate.py          # 流式生成节点
│   │   │   └── output_guard.py      # 输出护栏：引用校验 + 合规规则
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── base.py              # TOOL_REGISTRY + register_tool 装饰器
│   │       ├── registry.py          # get_tools()：按配置发现并实例化
│   │       ├── builtin/
│   │       │   ├── __init__.py
│   │       │   ├── retrieval.py     # 知识库检索工具（包装 rag_service）
│   │       │   └── memory_tool.py   # 长期记忆读写工具（包装 memory_service）
│   │       └── custom/              # 预留目录：后续新增业务工具
│   │           └── __init__.py
│   ├── infra/
│   │   ├── __init__.py
│   │   ├── user/                    # 用户仓储
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # UserRepository 抽象接口
│   │   │   ├── factory.py           # get_user_repository()
│   │   │   ├── mock_user.py         # Mock 实现（模拟 Postgres 表语义）+ 预置用户
│   │   │   └── postgres_user.py     # 预留：SQLAlchemy async + asyncpg
│   │   ├── org/                     # 部门仓储
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # DepartmentRepository 抽象接口
│   │   │   ├── factory.py           # get_department_repository()
│   │   │   ├── mock_org.py          # Mock 实现 + 预置人事部/市场部
│   │   │   └── postgres_org.py      # 预留
│   │   ├── session/                 # 会话元数据仓储
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   ├── mock_session.py      # Mock 实现 + 预置历史会话
│   │   │   └── postgres_session.py  # 预留
│   │   ├── checkpointer/
│   │   │   ├── __init__.py
│   │   │   ├── factory.py           # get_checkpointer()
│   │   │   └── mock_redis.py        # MockRedisSaver：字典 + TTL 语义 + 预置对话快照
│   │   ├── store/
│   │   │   ├── __init__.py
│   │   │   ├── factory.py           # get_store()
│   │   │   └── mock_postgres.py     # MockPostgresStore：模拟表语义 + 预置用户画像
│   │   └── vectordb/
│   │       ├── __init__.py
│   │       ├── base.py              # 抽象接口（对齐 pymilvus MilvusClient 签名）
│   │       ├── factory.py           # get_vector_client()
│   │       └── mock_milvus.py       # 有限子集 expr 解析器 + numpy 余弦检索 + 种子语料
│   └── rag/
│       ├── __init__.py
│       ├── chunker.py               # 文档切分（按段落/长度，重叠窗口）
│       ├── embedder.py              # 嵌入封装：hash（默认，确定性）/ api（预留）
│       └── ingest.py                # 加载 → 切分 → 嵌入 → 打权限标 → 入库
├── web/                             # 前端（FastAPI StaticFiles 同域托管，唯一静态目录）
│   ├── index.html                   # 登录页 + 主界面（会话列表 + 聊天窗）
│   ├── app.js                       # JWT 存储、401 自动调 refresh 重试、fetch 流式解析 SSE、消息渲染、图片上传
│   └── style.css
├── tests/
│   ├── conftest.py                  # 公共 fixture：种子数据构建、登录辅助（见 12 节）
│   ├── test_permissions.py          # 权限回归矩阵（核心）
│   └── test_security.py 等          # 按分层补齐（expr/rag/chat_sse/sessions_images）
├── scripts/
│   ├── create_db.sql                # PostgreSQL 建表脚本（departments/users/sessions/images）
│   └── seed_data.py                 # 预置全部模拟数据（用户/部门/知识/记忆）
└── uploads/                         # 图片目录（运行时创建，git 忽略；不挂静态，仅经 /api/knowledge/images/{id} 代理访问）
```

---

## 4. 模块设计

### 4.1 HTTP 层（FastAPI + JWT）

#### API 一览

| 方法 | 路径 | 说明 | 权限 |
|---|---|---|---|
| POST | /api/auth/register | 注册 | 公开 |
| POST | /api/auth/login | 登录，签发 access + refresh 双 token | 公开 |
| POST | /api/auth/refresh | 用 refresh token 换新 access token | 持有效 refresh token |
| POST | /api/auth/logout | 登出，服务端吊销 refresh token | 登录 |
| GET | /api/auth/me | 当前用户信息 | 登录 |
| POST | /api/chat/stream | SSE 流式对话 | 登录 |
| GET | /api/sessions | 会话列表 | 登录 |
| POST | /api/sessions | 新建会话 | 登录 |
| DELETE | /api/sessions/{thread_id} | 删除会话 | 登录（限本人） |
| POST | /api/knowledge/documents | 文档上传入库 | 登录（admin 可选 dept/public） |
| DELETE | /api/knowledge/documents/{doc_id} | 删除文档（向量 + 元数据，限本人或 admin） | 登录（限本人/admin） |
| POST | /api/knowledge/images | 图片上传（MIME 白名单 + 10MB 上限），返回代理 URL | 登录 |
| GET | /api/knowledge/images/{image_id} | 图片访问代理（权限校验 + 支持 ?token= 供 img 标签） | 登录（仅本人/admin） |
| GET | /api/depts | 部门列表 | 登录 |
| GET | /api/health | 健康检查 | 公开 |

#### SSE 帧协议（chat 流，前后端唯一约定）

| 帧类型 | 语义 | 终态 |
|---|---|---|
| token | 增量内容 `{"type":"token","content":"..."}` | 否 |
| warning | 非致命提示（如引用净化，见 6.4），流继续 | 否 |
| done | 正常结束 | **是** |
| error | 异常结束（附 message） | **是** |

- **终态帧互斥**：一条流只发一个终态帧，`error` 之后**不再发 `done`**，前端收到任一终态帧即认为流结束。
- **服务端实现保证**：chat_service 用状态标志 + `finally` 兜底——异常路径发 error 帧后关闭；若异常发生在终态帧发出之前且未被捕获，`finally` 检测"未发终态"则补发 error 再关闭，杜绝"无终态的悬挂流"。
- **断流语义**：连接断开但未收到终态帧 → 前端走"连接中断"处理（见 9.2），不当作正常结束。
- **心跳（可配置）**：服务端周期发送 SSE 注释行 `: ping`，防代理/网关空闲超时断连；uvicorn 直连默认关闭。注意：部分代理服务器仍可能缓冲注释行，生产环境部署在 NGINX / CloudFlare 之后时，需配置 `proxy_buffering off`（NGINX）或关闭 CloudFlare 缓冲/使用其 SSE 支持，确保注释行与 token 帧即时透传。

#### JWT 设计

- 签发：pyjwt，HS256。
- **双 token 机制**（缩短泄露风险窗口）：
  - `access_token`：有效期 120 分钟，payload 含 `sub(user_id)`、`role`、`exp`；无状态校验，所有业务接口使用。
  - `refresh_token`：有效期 7 天，payload 额外含 `type=refresh`、`jti`（唯一标识）；签发时 `auth_service` 将 `jti → exp` 记入 TokenStore（当前 Mock 内存字典，生产 Redis）。
  - `/auth/refresh`：校验 refresh 签名 + `jti` 未吊销 → 换发新 access；refresh 用后轮换（旧 jti 作废、签发新 refresh），防止 refresh token 长期重放。
  - `/auth/logout`：吊销 refresh 的 `jti` + 前端清 localStorage。已签发的 access 剩余有效期（≤2h）内仍有效，此窗口风险企业内部应用可接受，不做 access 黑名单。
- 校验：`core/deps.py` 的 `get_current_user` 依赖注入，解析 token 后**实时查库**组装 `CurrentUser(user_id, username, role, dept_id)`。
- 关键安全决策：**JWT 不存 dept_id**。用户调岗后旧 token 不会按旧部门越权检索（部门以库内实时数据为准）。
- **SSE 流中途过期的处理**（明确决策）：`get_current_user` 只在请求进入时（HTTP 层依赖注入）校验一次，`astream` 过程中**不重复校验、不主动断流**——连接建立在鉴权之后，视为已授权会话；下一次新请求/新连接时自然 401。前端统一处理：收到 401 → 调 `/auth/refresh` 重试一次 → 仍失败跳登录页。流中途失效风险窗口 = access 剩余有效期，与上述登出窗口同等对待。
- 密码：passlib[bcrypt] 哈希存储，不存明文；**`seed_data.py` 预置用户密码必须经 bcrypt 哈希后写入**（调用与 auth_service 相同的哈希函数），禁止明文入库。
- **XSS 与 localStorage 的权衡（显式声明）**：token 存 localStorage 意味着一旦前端出现 XSS，脚本可窃取 token 冒充用户。当前防线：前端渲染一律 `textContent`、禁 `innerHTML`（硬性约束，见 9.2）。已知风险窗口企业内部应用可接受。**未来增强方向（标注即可，暂不实现）**：refresh token 改存 `httpOnly + Secure + SameSite=Strict` Cookie（JS 不可读，彻底杜绝窃取），access token 仍走内存/Authorization 头；服务端需相应增加 CSRF 防护（Cookie 自动携带带来的新面），故作为独立迭代而非当前范围。
- **限流（slowapi，防暴力破解）**：
  - `/api/auth/login`、`/api/auth/register`、`/api/auth/refresh`：按 IP 限 5 次/分钟，超限返回 429
  - 实现放 `core/ratelimit.py`（Limiter 实例 + 规则），路由声明式接入
  - 触发限流记审计 `rate_limited`（见 13.2），作为爆破检测素材
  - 当前单进程内存计数即可；多实例部署时计数后端随 TokenStore 一起切 Redis

### 4.2 服务层（services）

服务层是路由与 Agent 之间的隔离带，路由不直接 import Agent 与底层仓储。

| 服务 | 职责 |
|---|---|
| auth_service | 注册校验、密码哈希、登录验证、双 token 签发与刷新轮换、登出吊销（TokenStore） |
| chat_service | 组装 graph config、调用 astream、SSE 帧协议实现（终态互斥 + finally 兜底，见 4.1）、thread 级单飞锁（同会话并发请求 409，见 4.5）、图片引用置位（消息实际携带 image_id 时置 images.bound=TRUE，见 7.1 孤儿清理待办） |
| session_service | 会话 CRUD（走 session 仓储）、标题生成（文本首句截断 20 字；首条消息含图片：有文字取文字并加 `[图片]` 前缀，纯图片为"图片对话"） |
| memory_service | Store 读写（namespace 按 user 隔离）、事实提取（rule/llm 提取器，见 6.6）、会话摘要入向量库（personal 区） |
| rag_service | 文档/图片入库 pipeline、`_build_permission_expr()`、双路召回合并、召回后二次权限校验 |
| dept_service | 部门列表查询 |

### 4.3 Agent 层（LangGraph）

#### 状态定义（state.py）

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 对话消息（含多模态内容）
    user_id: str
    role: str
    dept_id: str | None
    retrieved_docs: list[dict]                # 本次召回片段（含权限元数据）
    recalled_from_store: list[dict]           # Store 语义检索结果
```

#### 图结构

```
START → retrieve ──→ generate ──→ output_guard ──→ END
              │           │
              │（预留 ReAct 条件边）
              │           └─ should_continue()
              │                ├─ "tools" → tool_node → 回 generate
              │                └─ END
              └─（输入含图片时先经 vision 节点）
```

- 编排：`StateGraph(AgentState)`，全异步节点，`graph.astream(stream_mode="messages")` 流式输出。
- 条件边 `should_continue` 先预留不启用（`AGENT_ENABLED_TOOLS` 为空时直达 output_guard）。**未来启用方式**：`.env` 设 `AGENT_ENABLED_TOOLS=retrieval,memory`（已注册工具名的逗号分隔列表）→ graph 构建时读取该列表，对 LLM `bind_tools` 并挂载条件边到 ToolNode——不改图代码；启动时校验工具名均已注册，未注册报错（配置细节见第 10 节校验规则）。
- Checkpointer 通过工厂注入，`compile(checkpointer=..., store=...)`。

#### Tool 接口预留（tools/）

```python
# base.py —— 注册机制
TOOL_REGISTRY: dict[str, StructuredTool] = {}

def register_tool(name: str, description: str):
    """注册装饰器：装饰 async 函数即可被 Agent 发现和绑定"""
    def decorator(fn):
        TOOL_REGISTRY[name] = StructuredTool.from_function(
            coroutine=fn, name=name, description=description
        )
        return fn
    return decorator

def get_tools(names: list[str]) -> list[StructuredTool]:
    """按配置名列表返回工具实例，供 graph bind_tools"""
    return [TOOL_REGISTRY[n] for n in names if n in TOOL_REGISTRY]
```

- 工具通过参数注入 service 实例，不直接 import 底层仓储。
- 内置：`search_knowledge`（检索）、`save_memory` / `recall_memory`（长期记忆读写）。
- 新增工具三步：`custom/` 写 async 函数 + `@register_tool` → `.env` 加名字 → 自动绑定，图代码零改动。

### 4.4 可切换基础设施层（infra）— 四大工厂

统一模式：**抽象接口 → 当前实现 → Mock 实现（带模拟数据）→ 真实后端预留**。
工厂返回类型对齐 langgraph / pymilvus 官方接口签名，**切换后端业务代码零改动，只改 .env**。

| 工厂 | 当前 | Mock（预置模拟数据） | 预留真实后端 |
|---|---|---|---|
| Checkpointer（短期记忆） | `MemorySaver` | `MockRedisSaver`：字典 + TTL 语义 + 预置多轮对话快照 | `AsyncRedisSaver` / `AsyncPostgresSaver` |
| Store（长期记忆） | `InMemoryStore` | `MockPostgresStore`：模拟表语义 + 预置用户画像/偏好 | `AsyncPostgresStore` |
| 用户/部门/会话仓储 | — | 内存字典模拟 Postgres 表语义 + 全量种子 | PostgreSQL（SQLAlchemy 2.0 async + asyncpg） |
| VectorDB | — | `MockMilvus`：numpy 余弦检索 + 有限子集 expr 解析 + 种子语料 | `pymilvus.MilvusClient` + Milvus 服务端 |

关键约定：

- 短期记忆隔离：checkpointer config = `{"thread_id": 会话ID, "user_id": 用户ID}`（`user_id` 仅用于 config 传递与审计，**checkpointer 自身只按 `thread_id` 索引**）。
- 长期记忆隔离：Store namespace = `(user_id, "memories")`，支持 `semantic_search`。
- Mock 语义对齐：Mock 组件的方法签名、过滤语义与真实后端一致，本地即可验证切换正确性。

### 4.5 Mock 与真实后端切换的边界约定（三条隐性坑点的对策）

总原则：**Mock 对齐 langgraph 抽象接口（`BaseCheckpointSaver` / `BaseStore`）与仓储抽象接口的行为语义，不对齐官方实现的物理表结构。** 物理结构是官方库的内部实现细节，跨版本会变化，也不需要业务代码感知。

**1）Checkpointer（MemorySaver / MockRedisSaver → AsyncRedisSaver / AsyncPostgresSaver）**

- 序列化：`MockRedisSaver` 存取 checkpoint 用 langgraph 的 `JsonPlusSerializer`（与官方 serde 一致），**不裸存 dict**，避免与官方序列化格式形成隐式耦合。
- 索引方式：checkpointer 只有 `thread_id` 维度的键空间；**`user_id` 不进 checkpointer 索引**——会话归属校验由 sessions 元数据表 + 路由层完成（thread 不属于当前用户即 403），Mock 与真实后端一致。
- 预置快照：仅用于本地演示"有历史会话"，**不承诺跨后端迁移**；切换真实后端 = 空库起步（短期记忆本有 TTL 语义，历史对话丢失可接受）。
- **同一会话并发写入（明确边界与对策）**：`MemorySaver` 是进程内字典**无锁**；`AsyncPostgresSaver` 按 `(thread_id, checkpoint_ns, checkpoint_id)` 追加式存储，OSS 版**不做 thread 级运行互斥**（per-thread 锁是 LangGraph 平台版能力）——同一 thread_id 同时跑两个 astream 会交错写入，后完成者覆盖"最新 checkpoint"指针，导致历史错乱。**三层防护**：
  1. 前端发送期间禁用发送按钮与回车（体验层，见 9.2）
  2. 服务端 `chat_service` 持 thread 级 `asyncio.Lock` 字典，同 thread 的第二个请求在进入 astream 前直接返回 **409**（正确性层，当前实现，约 10 行；锁在 finally 中释放，防异常悬挂）
  3. 多实例部署时锁后端切 **Redis SETNX**（随 TokenStore 一起切，预留）

**2）Store（InMemoryStore / MockPostgresStore → AsyncPostgresStore）**

- 官方 `AsyncPostgresStore`（`langgraph.store.postgres`）**自建自管物理表**（namespace 为数组类型、value JSONB、语义检索用 pgvector 向量列，另有迁移版本记录表），且 DDL 随版本演进。
- `MockPostgresStore` 的对齐目标是 **BaseStore API 行为**（`put` / `get` / `search` / `list_namespaces` 的参数形态与返回结构，namespace 元组 `(user_id, "memories")`），**不复刻官方物理 DDL**。
- 切换时由 `AsyncPostgresStore.setup()` 自动建表迁移，业务代码（memory_service）零改动；第 5.5 节的 store 表 DDL 仅为概念参考，**禁止手动创建**（详见 create_db.sql 注释）。

**3）用户/部门/会话仓储（内存字典 → PostgreSQL）**

- 切换实现为 SQLAlchemy 2.0 Declarative 模型 + `asyncpg`，模型字段与第 5 节 DDL 一一对应。
- 会话管理：请求级 `AsyncSession`（FastAPI 依赖注入签发，`async_sessionmaker(expire_on_commit=False)`），repository 方法接收 session 参数，不自行开连接。
- 连接池：`pool_size=10, max_overflow=20, pool_pre_ping=True`（防云环境断连），事务提交边界在 service 层。
- Mock 字典实现的接口签名与仓储抽象完全一致，切换只替换工厂返回的实现类，service/路由层零改动。

---

## 5. 数据库设计（生产端 PostgreSQL）

> 当前阶段全部由 Mock 组件按以下表结构模拟；切换生产时建表即可。

### 5.1 users（用户表）

```sql
CREATE TABLE users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      VARCHAR(64)  UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,        -- bcrypt
    email         VARCHAR(128) UNIQUE,
    role          VARCHAR(16)  DEFAULT 'user',  -- user / admin
    dept_id       UUID REFERENCES departments(id),
    is_active     BOOLEAN      DEFAULT TRUE,
    last_login_at TIMESTAMPTZ,                 -- 最近登录时间（update_last_login 使用）
    created_at    TIMESTAMPTZ  DEFAULT now()
);
```

> 建表脚本见 [scripts/create_db.sql](scripts/create_db.sql)。

### 5.2 departments（部门表）

```sql
CREATE TABLE departments (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(64)  UNIQUE NOT NULL,
    code       VARCHAR(32)  UNIQUE NOT NULL,
    parent_id  UUID REFERENCES departments(id),   -- 预留层级
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 5.3 sessions（会话元数据表）

> Checkpointer 只存对话状态，无法按用户枚举会话列表，故需本表。

```sql
CREATE TABLE sessions (
    thread_id  UUID PRIMARY KEY,
    user_id    UUID REFERENCES users(id),
    title      VARCHAR(256),
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 5.4 images（图片元数据表）

记录上传图片元数据，配合代理端点做权限校验（见 7.1）；图片文件本体存本地 uploads/ 目录，不入库。

```sql
CREATE TABLE IF NOT EXISTS images (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- image_id，UUIDv4 不可猜测
    user_id    UUID REFERENCES users(id),
    mime       VARCHAR(32) NOT NULL,           -- image/png 等（白名单校验后入库）
    file_path  VARCHAR(256) NOT NULL,          -- 相对路径 {user_id}/{image_id}.{ext}
    bound      BOOLEAN DEFAULT FALSE,          -- 是否已被会话消息实际引用（孤儿清理判定依据，见 7.1 待办）
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_images_user ON images (user_id);
```

### 5.5 store（长期记忆表 — langgraph Store 底层存储）

> 生产由 `AsyncPostgresStore` 自动建表管理（需 pgvector 扩展），业务代码只经 Store API（put/get/search）读写，不直接写 SQL。以下仅为**概念级逻辑结构**——官方物理表结构与此不同（namespace 为数组类型、含迁移版本表）且随版本演进，`MockPostgresStore` 对齐的是 BaseStore API 语义而非此 DDL，切换时由库的 `setup()` 自动建表（边界约定见 4.5）。

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE store (
    prefix     TEXT NOT NULL,              -- 命名空间路径，如 '{user_id}/memories'，按用户隔离
    key        TEXT NOT NULL,              -- 记忆条目键
    value      JSONB NOT NULL,             -- 结构化记忆内容（事实/偏好/画像）
    embedding  vector(1536),               -- 语义检索向量（供 semantic_search）
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (prefix, key)
);

CREATE INDEX idx_store_prefix   ON store (prefix);
CREATE INDEX idx_store_embedding ON store USING hnsw (embedding vector_cosine_ops);
```

说明：

- 与 6.2 的记忆分流对应：**只存结构化事实**；非结构化长文本不入此表，走向量库 personal 区（见第 6 节），检索入口统一由 Agent 的 retrieve/记忆节点封装。
- 写入路径：memory_service 提取事实 → Store.put(namespace, key, value)；检索路径：Store.search(namespace, query=...) 语义召回。
- 语义检索开启受 Store 索引配置控制；Mock 环境用哈希向量模拟同样语义。

### 5.6 checkpoints（langgraph 自管）

由 `AsyncPostgresSaver` 自动建表管理，业务代码不直接读写。

---

## 6. RAG 知识库设计

### 6.1 向量记录 schema（Mock 与 pymilvus 保持一致）

```python
{
    "id": str,
    "vector": [...],                       # 维度 1536（对齐主流嵌入模型）
    "text": str,
    "scope": "domain" | "personal",        # 检索域
    "visibility": "public" | "dept" | "private",
    "owner_user_id": str | None,           # private/personal 必填
    "dept_id": str | None,                 # dept 时必填
    "source": str,                         # manual/summary/upload
    "created_at": int
}
```

### 6.2 长时记忆分流原则

| 记忆类型 | 例子 | 归宿 |
|---|---|---|
| 结构化事实 | 用户偏好、画像、简短结论 | Store（namespace 按 user 隔离） |
| 非结构化长文本 | 对话摘要、用户上传文档、笔记 | 向量库 personal 区（需 chunking + 语义召回） |

判断标准：需要"切片 + 相关性召回"的进向量库，短事实进 Store。
受 `MEMORY_TO_VECTOR_ENABLED` 开关控制，关闭则退回纯 Store 模式。

### 6.3 知识权限隔离方案

**方案组合：标量过滤（主）+ Partition 加速（辅）+ 检索后校验（兜底）**

#### 可见性矩阵 → 服务端统一拼 expr

| 知识类型 | visibility | 谁能检索 | 过滤表达式 |
|---|---|---|---|
| 领域公开文档 | public | 所有用户 | `scope=="domain" and visibility=="public"` |
| 部门文档 | dept | 本部门用户 | `... and visibility=="dept" and dept_id=="{当前用户dept}"` |
| 个人记忆/上传 | private | 仅本人 | `(scope=="personal" or visibility=="private") and owner_user_id=="{当前用户id}"` |

三路以 `or` 合并为一条 expr 传入检索。

**Mock 的 expr 一致性策略（明确决策）：有限子集解析器**

- `MockMilvus` 内置一个极简表达式求值器（约 60 行），仅支持本系统实际生成的语法子集：字段名、`==`、字符串字面量、`and`、`or`、括号；不支持范围、模糊匹配、`in`、JSON 字段等高级语法。
- 约束对等：`rag_service._build_permission_expr()` 只生成该子集内的表达式（代码注释显式声明此约束），**Mock 与真实 Milvus 接收并解析同一个 expr 字符串**——Mock 解析求值后对通过记录做 numpy 余弦排序。
- 一致性收益：expr 语法路径在本地开发期即被真实测试覆盖，切换真实 Milvus 不会出现"语法从未跑过"的兼容性问题；硬编码过滤则会让 expr 沦为 Mock 模式下的死代码，切换时才暴露语法错误，故不采用。
- 失败显式：求值器遇到子集外的 expr 直接抛出解析错误（而非静默漏过滤），防止权限逻辑悄悄失效。
- 第二道防线兜底：召回后的二次权限校验独立于 expr 执行，即使 expr 生成或解析有偏差也能拦截越权结果。

#### 加速

Milvus 侧将 `owner_user_id` 设为 partition key（自动按用户分区）；部门文档量大时再按 `dept_id` 分区。

#### 安全红线（纵深防御）

1. **过滤条件只能由服务端构造**：`rag_service._build_permission_expr(CurrentUser)` 依据 JWT 解出的身份生成；请求体中任何前端传来的过滤参数一律忽略。
2. **默认拒绝**：没打权限标的知识默认不可检索；写入时由 knowledge 路由按上传者身份自动打标（admin 可选 public/dept，普通用户强制 private）。
3. **二次校验（位置明确：rag_service 双路召回返回前，服务层收口）**：逐条复核召回结果的 `scope / visibility / owner_user_id / dept_id` 是否与当前用户匹配，不合规条目剔除并记审计 `permission_denied`。放在服务层而非 retrieve 节点内的理由：未来 retrieval 工具（AGENT_ENABLED_TOOLS 启用）复用同一 `rag_service.search()` 入口，防线自动覆盖两条调用路径。
4. **审计预留**：记录每轮召回的 doc id，防"多轮套取聚合泄露"（Mosaic 效应）后续做异常检测。

#### 权限过滤完整时序（一次对话请求）

```
① 路由依赖 get_current_user     JWT 校验 → 实时查库 → CurrentUser(user_id, role, dept_id)
② chat_service                  组装 graph config（user_id 注入 config）
③ graph.astream                 启动异步流
④ retrieve 节点                 调 rag_service.search(user, query)
⑤ rag_service                   _build_permission_expr(CurrentUser) 生成权限 expr
⑥ vector_client.search          expr 过滤 + 向量召回（Mock：掩码过滤 → 余弦排序）
⑦ rag_service（二次校验）        逐条复核权限字段 → 剔除不合规 → 记审计 permission_denied
⑧ 返回 retrieve 节点            合并 domain/personal 两路 → 排序 → 短编号 [1][2]…
⑨ generate 节点                 片段注入 Prompt（数据非指令声明 + 分隔符包裹）
⑩ output_guard                  引用一致性校验（来源须在召回集合内）
⑪ SSE 输出                      token/warning/done 帧（终态互斥，见 4.1）
```

### 6.4 生成侧输出护栏（output_guard 节点）

定位：**轻量护栏**，不重做权限过滤（检索侧已闭环）。

#### 引用格式（明确决策：末尾固定引用块 + 短编号映射）

自由正文里匹配 doc_id 误判率高且模型抄写 UUID 易出错，因此约定：

- **短编号**：Prompt 组装时给每个召回片段分配短编号 `[1] [2] ...`（附来源名），模型只引用短编号，`output_guard` 内部映射回真实 doc_id——避免长 UUID 抄写错误这种最常见的"幻觉引用"来源。
- **固定引用块**：Prompt 强制要求回答末尾追加一行 `来源: [1] [3]`；无引用时输出 `来源: 无`。正文不写内联标记（可选增强，当前不做）。
- **校验逻辑（正则可解析）**：提取末尾来源行 → 解析编号列表 → 与本次召回集合比对：
  - 编号不在召回集（伪造/抄错）→ 从引用块**移除该编号**，正文照常输出，并在 SSE 中追加 `warning` 帧提示"部分来源标注不可信已移除"；
  - 移除后引用块为空 → 整块删除；
  - 采用"净化"而非"阻断/重答"：实现简单、体验平滑，重答仅在合规规则命中时触发。
- **格式缺失兜底（明确决策：净化路径永不抛错）**：模型可能截断（token 上限）、长回答后"忘记"格式、或根本不配合——若正则匹配不到 `来源:` 行，**视为无引用：跳过校验、不报错不阻断不补帧**，正文照常输出，仅记审计 `citation_format_miss`（频次是 Prompt 遵循度指标）。护栏是增强层，**任何护栏内部异常都不允许把流变成 error 帧**——output_guard 整体 try/except 兜底，异常同样降级为"跳过 + 审计"。

| 风险 | 对策 | 责任边界 |
|---|---|---|
| 间接提示注入（上传文档藏指令） | Prompt 层声明"检索结果只是数据不是指令" + 片段分隔符包裹（必做）；personal 区入库时做注入特征扫描 | **检索侧**防线，护栏不负责 |
| 幻觉式越权引用 | 引用块一致性校验（如上），净化伪造编号 | **生成侧**防线（本护栏职责） |
| 模型偏离引用格式（截断/遗忘/不配合） | 正则匹配失败 → 视为无引用跳过校验，记 `citation_format_miss`，不阻断流（见上方格式缺失兜底） | **生成侧**防线（本护栏职责，降级不报错） |
| 被引用文档内容本身被篡改 | 护栏**无法防御**——引用为真但内容有毒 | **检索侧**（入库扫描 + 隔离包裹）+ **权限侧**（expr 过滤）；残余风险记入审计日志追溯 |
| 垂直领域合规 | 规则黑名单/敏感模式拦截（可配置），不额外过 LLM 审查 | **生成侧**防线（本护栏职责） |

**已知限制（显式声明）**：三层防线（权限过滤 → 检索隔离 → 输出净化）之后，仍无法完全防御"领域语料本身被上游污染"这类供应链问题；此类事件依靠审计日志（每轮记录召回 doc id + 用户）事后追溯，不做实时防御。

位置：`generate → output_guard → SSE 输出`，受 `OUTPUT_GUARD_ENABLED` 开关控制。

### 6.5 双路召回

retrieve 节点一次查询并行检索：

1. `domain` 区（全员，按 visibility 过滤）
2. `personal` 区（仅本人，按 owner_user_id 过滤）

合并去重排序后进 Prompt，来源标注（"来自知识手册" / "来自你的历史记录"）。

### 6.6 长期记忆提取与摘要（memory_service 的两条异步链路）

**触发时机**：每轮 SSE 流 done 之后由 `asyncio.create_task` 异步触发，**不阻塞响应主链路**（提取失败只记日志/审计，绝不影响对话）。

**链路一：结构化事实提取 → Store**（明确决策：初期规则提取器，预留 LLM 提取器）

- 抽象接口 `FactExtractor.extract(user_message) -> list[Fact(key, value, type)]`，两个实现按配置切换：
  - `RuleBasedExtractor`（默认，`MEMORY_EXTRACTOR=rule`）：正则/关键词模式库匹配，仅处理 user 角色消息。初期模式集：`我叫X`→name、`我是X部门的`→dept、`请用X回复 / 我喜欢X风格`→reply_preference、`记住：X`→note；模式库为常量表。**注意：以上仅为示意，真实用户表述多样（如"我在X部门"就不匹配"我是X部门的"），模式集需在阶段 5 实施时根据实际语料扩充与验证，并在 12 节测试中覆盖各模式的命中用例。**
  - `LLMExtractor`（预留，`MEMORY_EXTRACTOR=llm`）：用当前 LLM 以固定 Prompt 输出 JSON 事实数组，解析失败静默跳过该轮（提取是增强能力，不允许拖垮主链路）。
- 冲突策略：同 key 新值覆盖旧值（Store.put 语义），不做复杂合并。
- 预置画像（8.1）与提取写入同一 namespace，检索路径一致。

**链路二：对话摘要 → 向量库 personal 区**

- 触发条件：同一 thread 累计轮次达到 `MEMORY_SUMMARY_EVERY_TURNS`（默认 10）的倍数时，把该会话历史压缩为摘要文本，替换更新旧摘要（同 doc_key 覆盖），按 private 入 personal 区。
- mock 模式下摘要 = 固定模板拼接（前 N 条消息截断），LLM 模式下用模型生成。

**配置**：`MEMORY_EXTRACTOR=rule`、`MEMORY_SUMMARY_EVERY_TURNS=10`（第 10 节）。

---

## 7. 多模态设计

- **输入**：聊天请求支持图文混合；图片先上传 `POST /api/knowledge/images`，消息中携带**代理 URL** `/api/knowledge/images/{image_id}`（相对路径，前端拼接）。
- **理解**：vision 节点调用多模态 LLM（qwen-vl / GPT-4o 等兼容接口）生成图片描述，与文本一起进入检索/生成。
- **入库**：用户上传的图片经视觉模型生成文字描述后，按 private 权限入 personal 区。
- **Mock 模式**：`LLM_BACKEND=mock` 时图片理解返回固定描述文案，链路可测。

### 7.1 图片存储与访问（明确决策：API 代理，不走裸静态目录）

**存储**：

- 文件落盘 `uploads/{user_id}/{image_id}.{ext}`，按用户分目录物理隔离；文件名由服务端生成（UUID），不信任原始文件名。
- 上传校验：MIME 白名单（png/jpg/jpeg/webp）+ 大小上限 10MB + 扩展名由 MIME 映射，防止伪装文件。
- 元数据入 `images` 表（见 5.4）：`image_id / user_id / mime / file_path / created_at`。

**访问**：

- `uploads/` **不做 StaticFiles 挂载**（对比：仅 `web/` 由 FastAPI 同域托管），文件系统不直接暴露，唯一出口是代理端点 `GET /api/knowledge/images/{image_id}`（登录鉴权）。
- 权限校验：查 images 元数据，`user_id == 当前用户` 或 `role == admin` 才返回文件，否则返回 **404**（用 404 而非 403，不暴露资源存在性，防枚举探测）。
- `<img>` 标签无法携带 Authorization 头：该端点额外接受 query 参数 `?token=`（复用 access JWT，**仅此端点**支持）；已知风险是 URL 可能进入访问日志，企业内部应用可接受，后续增强可换 httpOnly Cookie（预留）。
- 双保险：image_id 为 UUIDv4 不可猜测 + 访问仍需校验。

**生产切换**：本地目录 → 对象存储（OSS/S3）时仅替换代理端点的读文件实现为 SDK 调用，权限校验与 URL 结构不变。

**孤儿图片清理（已知待办，后续迭代实现）**：用户上传后放弃发送或直接关页，会产生无引用的图片文件与 images 记录。Mock 内存字典阶段无影响；切 PostgreSQL 后会累积垃圾。预留设计：

- images 表已带 `bound` 标记（见 5.4）：上传时 `FALSE`，chat 请求实际携带该 image_id 成功入会话后由 chat_service 置 `TRUE`——**判定依据是标记位而非扫描消息体**（checkpoints 为 langgraph 自管表，扫消息文本既脆弱又费）
- 清理规则：`bound == FALSE` 且 `created_at` 超过 7 天 → 删文件 + 删记录（先文件后记录，失败重试），限流为每日一次定时任务
- 当前阶段不实现，仅保证：`bound` 字段与置位逻辑（chat_service 引用时置 TRUE）随阶段 3/6 一起落地，后续补定时任务即可无缝启用

---

## 8. 模拟数据设计（scripts/seed_data.py）

### 8.0 执行时机与幂等性

**执行时机（按后端区分，明确决策）**：

- **Mock 后端（当前）**：**服务启动时自动执行**——Mock 仓储是内存字典，进程重启数据即失，种子必须随启动注入（main.py 的 lifespan 挂钩调用 `seed_all()`），否则重启后连 alice 都无法登录。
- **真实后端（切换后）**：**不自动执行**（防止污染生产库），生产数据由建库脚本/DBA 管理；`seed_data.py` 保留 `python -m scripts.seed_data` 手动入口，仅供本地/开发环境验证（输出各仓储插入统计）。

**幂等性（存在即跳过，check-then-insert）**：

| 种子内容 | 幂等键 | 重复执行行为 |
|---|---|---|
| 部门 / 用户 | code / username | 存在即跳过（bcrypt 只在首次生成，复用 `core/security.py` 同一哈希函数，见 8.1 安全要求） |
| 知识语料向量 | doc_key（来源标识，如文件名+标题） | 存在即跳过，不产生重复向量 |
| Store 结构化画像 | namespace+key | 覆盖写（put 语义天然幂等） |
| 历史会话/摘要快照 | 固定 thread_id / 记忆键 | 存在即跳过 |

- 内存 Mock 场景每次启动都是全新字典，天然幂等；幂等设计主要保障**手动重复执行**与未来切换真实 Postgres 后的安全性。
- seed 模块只调用各仓储抽象接口（与业务代码同一路径），不直接操作字典内部结构。

### 8.1 用户与部门

| 用户 | 密码 | 角色 | 部门 | 预置数据 |
|---|---|---|---|---|
| alice | alice123 | user | 人事部(hr) | 3 条历史会话、1 段咨询摘要、1 篇上传文档、偏好：中文简洁回复 |
| bob | bob123 | user | 市场部(marketing) | 1 条历史会话、1 段咨询摘要、1 篇上传文档、偏好：要点列表式回复 |
| admin | admin123 | admin | 无部门 | 仅可检索 public 区 |

> 安全要求：表中密码仅为登录口令示意，`seed_data.py` 写入前必须经 bcrypt 哈希（复用 `core/security.py` 的哈希函数），库中只存 `password_hash`，禁止明文。

### 8.2 知识库种子（企业内部知识助手场景）

| 区域 | visibility | 语料 |
|---|---|---|
| public | public | 公司简介、差旅报销制度、信息安全规范 |
| 人事部 | dept(hr) | 考勤与请假制度、招聘流程规范、薪酬保密制度 |
| 市场部 | dept(marketing) | 品牌视觉手册、市场活动审批流程、竞品分析模板 |
| personal | private | alice/bob 的历史摘要与上传文档（见 8.1） |

### 8.3 Mock 组件行为

| 组件 | 模拟语义 |
|---|---|
| MockRedisSaver | 字典存储 checkpoint，TTL 过期语义，预置 alice 历史会话的多轮消息快照 |
| MockLLM | 流式输出固定模板文案，**末尾含 `来源: [1]` 引用块样例**（保证 output_guard 引用校验链路在 mock 模式下可测） |
| MockPostgresStore | 模拟表语义，预置用户画像/偏好的 namespace 数据，支持语义检索（hash 向量） |
| MockMilvus | numpy 余弦检索 + 有限子集 expr 解析器（仅 == / and / or / 字面量，子集外显式报错）+ 种子语料向量 |
| 哈希嵌入 | 确定性伪向量（同文本结果稳定、维度 1536），无嵌入 API 时检索/隔离逻辑可测（算法见 8.4） |

### 8.4 哈希嵌入算法（EMBEDDING_BACKEND=hash，rag/embedder.py 的 HashEmbedder）

**步骤**（确定性，跨进程结果一致）：

1. 分词：中文按 2-gram（相邻汉字滑窗）、英文/数字按空格分词，全部小写化
2. 对每个 token 计算 `sha256(token)`，取前 8 字节转无符号整数
3. `整数 % 1536` 映射到维度桶，对应桶累加 1
4. 结果向量做 L2 归一化，输出 1536 维 float32

**关键性质与红线**：

- **禁止使用 Python 内置 `hash()`**——其受 `PYTHONHASHSEED` 影响每次进程结果不同，会导致重启后向量对不上；必须用 hashlib 系列稳定哈希
- 同文本恒得同向量（幂等）；不同文本映射不同桶序列，碰撞概率极低
- **局限与用例编写约束（显式声明）**：哈希嵌入只编码**词面重叠**、无语义泛化——同义词/改写句永远召不回（如查"请假" vs 语料写"考勤"）。因此：① **Mock 阶段检索测试用例的查询词必须包含语料关键词，禁止编写依赖"语义召回"的用例**，语义召回质量验证推迟到 `EMBEDDING_BACKEND=api` 阶段（用例结构不变，仅检索行为从词面变为语义，避免切后端时测试行为突变）；② `test_rag.py` docstring 显式标注测试边界（见 12.1 用例编写规范）；③ 切 api 后检索代码零改动（工厂层已隔离）

---

## 9. 前端设计（web/）

### 9.1 页面结构

```
┌─────────────────────────────────────────────┐
│  顶栏：系统名 · 当前用户 · 退出登录             │
├────────────┬────────────────────────────────┤
│ 会话列表     │  聊天区                        │
│ (GET        │  ┌──────────────────────────┐ │
│  /sessions) │  │ 消息气泡（支持引用来源标注）│ │
│ + 新建会话   │  │ 图片消息渲染              │ │
│ + 删除会话   │  └──────────────────────────┘ │
│             │  [输入框] [图片上传] [发送]      │
└────────────┴────────────────────────────────┘
```

### 9.2 关键实现

- **SSE 解析**：用 `fetch` + `ReadableStream` 手动解析 `data:` 帧（EventSource 不支持 POST）；按帧协议状态机处理（见 4.1 SSE 帧协议）：token 增量追加、warning 展示提示条、done/error 为终态（error 展示错误，之后不再期待 done），任一终态帧即收尾并解锁输入框。**发送请求期间禁用发送按钮与回车**（同会话并发第一层防护，服务端 409 兜底，见 4.5）。
- **连接中断处理**（明确决策：提示重发，不自动重连生成流）：
  - 连接建立阶段失败（未收到任何 token）→ 自动重试 1 次（含 401 刷新后重试），仍失败提示；
  - 已开始接收 token 后断流且无终态帧 → 保留已生成文本并标注"回答可能不完整"，提示用户重新发送，**不自动重试**——LLM 生成不幂等，重试会产生重复回答；会话历史由 checkpointer 在节点边界落库保证完整，重新发送无损。
- **JWT 存储**：localStorage，请求统一带 Authorization 头；401 时自动调 `/auth/refresh` 重试一次，仍失败跳登录页（见 4.1 JWT 设计）。
- **XSS 防护（硬性约束）**：用户输入与消息内容（含模型输出、检索引用）渲染时**必须使用 `textContent` / `innerText` 创建文本节点，禁止 `innerHTML` 直接插入未过滤内容**——本项目为原生 JS 无框架自动转义，转义责任在渲染函数；图片 `<img src>` 仅允许指向本站代理端点 `/api/knowledge/images/{id}`（URL 由代码拼接 image_id，不拼接用户输入字符串）；`dangerouslySetInnerHTML` 类用法一律不允许出现。localStorage 存 token 的 XSS 权衡与未来增强方向见 4.1。
- **流式渲染**：token 增量追加到当前气泡；异常帧提示后恢复输入。
- **图片预览与消息缩略图**：选择文件后用 `URL.createObjectURL` 在输入区上方显示本地预览（可移除，发送完成后 `revokeObjectURL` 释放）；发送流程 = 先 `POST /api/knowledge/images` 拿 image_id → 再发起 chat 请求携带 → 气泡内渲染缩略图（`GET /api/knowledge/images/{id}?token=`，见 7.1），点击查看原图。
- **响应式布局（基础移动适配）**：`viewport` meta + CSS 媒体查询断点 768px——窄屏时会话列表折叠为抽屉（顶栏按钮切换）、聊天区全宽、触控目标不小于 40px；不做 PWA / 复杂手势（深度移动办公需求出现时再升级）。
- **引用展示**：回答末尾渲染来源列表（文档名/来源类型）。
- **编码**：全站 UTF-8，`<meta charset="utf-8">`。
- **托管**：FastAPI StaticFiles 同域托管（`/` 指向 web/），无需 CORS 配置。

---

## 10. 配置项清单（.env.example）

```ini
# ---- 应用 ----
APP_HOST=0.0.0.0
APP_PORT=8000
JWT_SECRET_KEY=change-me-in-production
JWT_EXPIRE_MINUTES=120               # access token 有效期
JWT_REFRESH_EXPIRE_DAYS=7            # refresh token 有效期
LOG_LEVEL=INFO                       # 日志级别（DEBUG/INFO/WARNING）

# ---- 后端切换开关（当前 Mock，生产切换值见注释）----
USER_DB_BACKEND=mock_postgres        # postgres
CHECKPOINTER_BACKEND=memory          # mock_redis | redis | postgres
STORE_BACKEND=in_memory              # mock_postgres | postgres
VECTOR_BACKEND=mock                  # milvus
LLM_BACKEND=mock                     # openai_compatible
EMBEDDING_BACKEND=hash               # api

# ---- LLM / Embedding（mock 模式下全部可留空，不校验）----
LLM_BASE_URL=                        # LLM_BACKEND=openai_compatible 时必填
LLM_API_KEY=                         # 同上
LLM_MODEL=qwen-plus                  # 同上
LLM_VISION_MODEL=qwen-vl-plus        # 同上（vision 节点用）
EMBEDDING_API_BASE=                  # EMBEDDING_BACKEND=api 时必填
EMBEDDING_API_KEY=                   # 同上
EMBEDDING_MODEL=text-embedding-v3    # 同上
EMBEDDING_DIM=1536                   # 嵌入维度，须与向量库 collection 一致

# ---- Agent ----
AGENT_ENABLED_TOOLS=                 # 留空 = 纯 RAG 链路（ReAct 条件边不启用）
                                     # 启用工具：设为已注册工具名的逗号分隔列表，如 retrieval,memory
                                     # 生效方式：graph 构建时按此列表 bind_tools 并启用条件边，无需改图代码
MEMORY_TO_VECTOR_ENABLED=true
OUTPUT_GUARD_ENABLED=true
MEMORY_EXTRACTOR=rule                # 事实提取器：rule（默认）| llm（预留，见 6.6）
MEMORY_SUMMARY_EVERY_TURNS=10        # 每累计 N 轮生成/更新一次会话摘要

# ---- Milvus（VECTOR_BACKEND=milvus 时必填，mock 模式下可留空）----
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=rag_knowledge

# ---- 真实后端连接（对应 *_BACKEND 切换时启用，mock 模式可留空，见 14.3）----
POSTGRES_DSN=                        # postgresql+asyncpg://rag:***@localhost:5432/rag_assistant（业务表 + store 共用）
REDIS_URI=                           # redis://localhost:6379/0（checkpointer + token 吊销）
TOKEN_REVOCATION_BACKEND=memory      # memory（默认，重启即失）| redis（阶段 8 切真实后端时改为 redis，见 14.3/15 节步骤 3）
```

**配置校验规则（config.py Settings，避免 mock 模式启动报错）**：

- 所有后端专属配置（`LLM_*` / `EMBEDDING_*` / `MILVUS_*`）声明为 `Optional`，默认空串。
- `config.py` 使用 model_validator **按 BACKEND 开关条件校验**：仅当对应 `*_BACKEND` 切到真实实现时，才检查必填项是否非空；缺失时报错信息明确列出字段名（如 `LLM_BACKEND=openai_compatible 需要设置 LLM_API_KEY`）。
- `AGENT_ENABLED_TOOLS` 启动时校验：列表中的工具名必须已在 TOOL_REGISTRY 注册，未注册的直接报错（防拼写错误静默失效）。

---

## 11. 依赖清单（requirements.txt）

```
# ---- 运行依赖（上界锁 major/minor，升级视为独立任务，见下方版本策略）----
fastapi>=0.115,<1.0
uvicorn[standard]>=0.30,<1.0
pydantic>=2.7,<3.0
pydantic-settings>=2.2,<3.0
langgraph>=0.2,<0.3
langchain-core>=0.3,<0.4
langchain-openai>=0.2,<0.3
pyjwt>=2.8,<3.0
passlib[bcrypt]>=1.7,<2.0
bcrypt==4.0.1                        # 已知坑：passlib 1.7.4 与 bcrypt>=4.1 不兼容（读取 __about__ 报错），必须钉住
python-multipart>=0.0.9,<1.0
slowapi>=0.1.9,<0.2                  # 登录等接口限流（见 4.1 限流）
pymilvus>=2.4,<3.0                   # MockMilvus 对齐其 MilvusClient API
numpy>=1.26,<3.0
# ---- 测试依赖（阶段 2 起）----
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
```

**版本策略**：

- **Python 3.10+**（建议 3.11；实施前先验证本机版本，阶段 1 验收项）。
- 运行依赖一律 `>=x.y,<下一 major` 上界约束；langgraph / langchain 生态演进快，**升级 minor 视为独立任务**：改约束 → pytest 全量通过（12 节）→ 才允许合入。
- 阶段 1 完成时锁定 `pip freeze > requirements.lock`，后续安装以 lock 为准。
- Redis / asyncpg / sqlalchemy 客户端暂不安装，切换阶段再加：`asyncpg`、`sqlalchemy>=2.0,<2.1`、`langgraph-checkpoint-postgres`、`langgraph-checkpoint-redis`（版本与 langgraph 配套，见 15 节步骤 2）。

---

## 12. 测试策略（pytest，随阶段编写）

Mock 环境零外部依赖，测试直接使用 Mock 组件 + `seed_data.py` 的种子构建函数（conftest fixture 复用同一套数据）。

### 12.1 目录与分层

```
tests/
├── conftest.py               # 公共 fixture：app 实例、种子数据构建、登录辅助（取 token）
├── test_security.py          # 单元：JWT 签发/过期/refresh 轮换/吊销、bcrypt 哈希
├── test_expr.py              # 单元：expr 解析器（==/and/or/字面量/括号、子集外报错）、_build_permission_expr 三类用户快照
├── test_permissions.py       # 回归（核心）：权限矩阵全量硬断言，防 expr 被后续修改破坏
├── test_rag.py               # 集成：入库 → 检索 → 命中/隔离；hash 嵌入确定性（docstring 声明：仅验证权限过滤与词面匹配链路，不验证语义泛化质量，见 8.4）
├── test_chat_sse.py          # 集成：SSE 帧序列、终态互斥（error 后无 done）、warning 帧
└── test_sessions_images.py   # 集成：会话归属 403、图片代理 404/200
```

**用例编写规范（Mock 嵌入边界，显式约束）**：

- `EMBEDDING_BACKEND=hash` 下检索只按**词面重叠**召回，无语义泛化——同义词/改写句（如查"请假" vs 语料写"考勤"）永远召不回。
- **Mock 阶段检索测试用例的查询词必须包含语料关键词；禁止编写依赖"语义召回"的用例**，语义召回质量验证推迟到 `EMBEDDING_BACKEND=api` 阶段（切 api 后用例结构不变，仅检索行为从词面变为语义）。
- `test_rag.py` 文件 docstring 显式标注：**本测试仅验证权限过滤与词面匹配链路，不验证语义泛化质量**（见 8.4）。

### 12.2 权限回归矩阵（test_permissions.py）

以种子数据为基准，每条都是"越权必须失败"的硬断言：

| 操作 | 期望 |
|---|---|
| alice 检索"市场部品牌手册" | 无召回 |
| bob 检索"人事部考勤制度" | 无召回 |
| alice 检索 public 差旅制度 | 有召回 |
| alice 检索自己上传的 private 文档 | 有召回 |
| bob 检索 alice 的 private 文档 | 无召回 |
| admin 检索任意 dept 区文档 | 无召回（仅 public） |
| bob 删除 alice 的会话 | 403 |
| bob 请求 alice 的 image_id | 404（而非 403：不暴露资源存在性，防枚举，设计意图见 7.1） |

防退化机制：expr 逻辑集中在 `_build_permission_expr` 一处，任何后续修改只要破坏矩阵任一条，`pytest tests/test_permissions.py` 即红；该文件同时是阶段 6 验收点的自动化形式。

### 12.3 编写时机（随阶段，不攒到最后）

| 阶段 | 同步编写 |
|---|---|
| 2 | test_security.py（login → me → refresh 轮换 → logout 吊销全链路） |
| 3 | test_chat_sse.py（帧序列与终态互斥）+ 同 thread 并发第二请求返回 409 的单飞锁用例 |
| 6 | test_expr.py + test_permissions.py + test_rag.py |
| 7 / 8 | test_sessions_images.py；阶段 8 验收 = pytest 全量通过 |

---

## 13. 监控与日志

### 13.1 日志规范（标准 logging，按层级输出）

- **实现**：`core/logger.py` 统一配置，`logging.getLogger(__name__)` 按模块路径天然分层；不自研封装、不引入三方日志库。
- **格式**：`时间 | 级别 | logger名 | user_id | 消息`；控制台输出为主（uvicorn 场景不落文件，生产由 Docker / 日志收集器接管），级别由 `.env` 的 `LOG_LEVEL` 控制（默认 INFO）。

| 层级（logger 命名空间） | 记录内容 | 级别约定 |
|---|---|---|
| `app.api` | 请求出入口：method、path、status、耗时 | 4xx=WARNING，5xx=ERROR，其余 INFO |
| `app.services` | 业务事件：检索耗时（duration_ms）、LLM 首 token/总耗时、SSE 异常 | 关键事件 INFO，失败 ERROR |
| `app.agent` | 节点执行：retrieve/generate 耗时 | INFO |
| `app.infra` | 后端组件事件：工厂实例化用了哪个后端、连接建立 | INFO（启动时打印组件清单） |

- **性能日志**：检索耗时、LLM 调用耗时用固定字段 `duration_ms` 输出，便于后续采集聚合。

### 13.2 审计日志（关键操作，独立分流）

- **实现**：独立 logger `app.audit`，输出**结构化 JSON 行**，与普通日志分流；当前阶段只落日志**不建审计表**（生产可加 audit 表或接 ELK，logger 接口不变，与 6.4 检索审计为同一通道）。
- **记录字段**：`ts / user_id / username / action / target / detail / ip`。
- **审计范围（写操作 + 安全事件）**：

| action | 触发点 |
|---|---|
| login_success / login_failed | auth 路由（failed 含用户名，供爆破检测） |
| logout | auth 路由 |
| rate_limited | slowapi 限流触发（4.1，记被限接口与 IP） |
| knowledge_upload / knowledge_delete | knowledge 路由（记文档标题与 scope，不记正文） |
| image_upload | 图片上传（记 image_id） |
| session_delete | 会话删除（记 thread_id） |
| permission_denied | expr 拦截、二次校验拦截、会话归属 403、图片代理 404（记被拒目标） |
| citation_sanitized / citation_format_miss | output_guard：净化伪造引用编号 / 模型未按格式输出引用块（记 thread_id，不记正文） |

- **敏感信息红线**：日志不记密码与 token 全文（token 只记 `jti` 前 8 位），不记文档正文（记 id + 标题）。

---

## 14. 部署与运维（Docker Compose）

### 14.1 三种部署形态

| 形态 | 构成 | 适用场景 |
|---|---|---|
| 本地纯 Mock（当前阶段） | 仅本机 `uvicorn app.main:app`，零外部依赖 | 功能开发、pytest 回归 |
| **依赖编排（开发联调，最常用）** | compose 起依赖服务，FastAPI 本机运行 | 切换真实后端联调（15 节阶段 8） |
| 完整容器化（生产演示） | compose 同时起 app + 全部依赖 | 部署验证、演示环境 |

### 14.2 docker-compose.yml 示例（依赖编排版，scripts/docker-compose.yml）

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag_dev_password
      POSTGRES_DB: rag_assistant
    ports: ["5432:5432"]
    volumes: [pg_data:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag_assistant"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7
    ports: ["6379:6379"]
    volumes: [redis_data:/data]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

  # ---- Milvus standalone 三件套：etcd（元数据）+ minio（对象存储）+ milvus ----
  etcd:
    image: quay.io/coreos/etcd:v3.5.14
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes: [etcd_data:/etcd]

  minio:
    image: minio/minio:RELEASE.2024-05-28T17-19-04Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
    volumes: [minio_data:/minio_data]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

  milvus:
    image: milvusdb/milvus:v2.4.15
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports: ["19530:19530", "9091:9091"]
    volumes: [milvus_data:/var/lib/milvus]
    depends_on:
      minio:
        condition: service_healthy

  # ---- 可选：应用容器化（完整容器化形态时取消注释）----
  # app:
  #   build: .
  #   ports: ["8000:8000"]
  #   env_file: .env
  #   depends_on:
  #     postgres: { condition: service_healthy }
  #     redis: { condition: service_healthy }
  #     milvus: { condition: service_started }

volumes:
  pg_data:
  redis_data:
  etcd_data:
  minio_data:
  milvus_data:
```

**要点**：
- 一键启动依赖：`docker compose -f scripts/docker-compose.yml up -d`，之后本机改 `.env` 开关重启 uvicorn 即切换后端
- Milvus standalone 必须带 etcd + minio，不能单容器起
- healthcheck + `depends_on.condition` 保证依赖就绪顺序；全部数据走 named volumes，`down` 不丢数据（`down -v` 才清空）
- 完整容器化时补 `Dockerfile`（`python:3.11-slim` + `pip install -r requirements.txt` + `uvicorn`），app 服务取消注释即可

### 14.3 切真实后端的 .env 联动（配置项见第 10 节）

| compose 服务 | 对应 .env 配置 |
|---|---|
| postgres（业务表 + store） | `USER_DB_BACKEND=postgres`、`STORE_BACKEND=postgres`、`POSTGRES_DSN=postgresql+asyncpg://rag:rag_dev_password@localhost:5432/rag_assistant` |
| redis（checkpointer + token 吊销） | `CHECKPOINTER_BACKEND=redis`、`TOKEN_REVOCATION_BACKEND=redis`、`REDIS_URI=redis://localhost:6379/0` |
| milvus | `VECTOR_BACKEND=milvus`、`MILVUS_URI=http://localhost:19530`（端口 9091 为健康检查） |

### 14.4 运维注意

- compose 为**单机演示/联调级**，生产高可用需 Kubernetes / 云托管服务，不在本项目范围
- **Milvus standalone 内存要求**：etcd + minio + milvus 三件套内存占用较高，官方建议运行环境可用内存 ≥ 4GB（8GB 更稳妥），资源受限机器上 `docker compose up` 可能 OOM。注意：**纯 Mock 模式开发无需启动任何容器**；仅当切 `VECTOR_BACKEND=milvus` 联调时才需要全套依赖，若机器内存不足，可评估轻量替代（如 Qdrant 单机版）作为 VECTOR_BACKEND 备选实现（工厂模式已预留扩展位）
- 应用日志按 13 节规范输出控制台，容器化后由 Docker/收集器接管，无需文件落盘
- **表结构迁移**：切真实 PostgreSQL 后由 Alembic 统一管理（初始 migration 基于第 5 节 DDL，见 15 节步骤 3）；users / departments / sessions / images 等业务表的后续字段变更一律通过 `alembic revision` 生成迁移脚本并在测试库验证后再上线；Mock 阶段为内存仓储无表结构，无迁移需求
- Milvus 集合创建由 `rag_service` 启动时 ensure（幂等），首次启动自动建 collection 与索引

---

## 15. 实施阶段计划

| 阶段 | 内容 | 验收点 |
|---|---|---|
| 1 | 项目骨架 + config 全量开关 + `core/logger.py` 日志/审计配置 + 依赖 + .env.example | 目录与方案一致，settings 可加载，启动打印组件清单 |
| 2 | JWT 认证 + 用户/部门/会话仓储（Mock + 种子）+ auth/dept/session 路由 | 登录三账号成功，/me 返回实时部门 |
| 3 | Agent 图（astream）+ Tool 预留接口 + SSE chat 路由 | mock LLM 下 SSE 逐 token 出流 |
| 4 | Checkpointer 工厂（MemorySaver + MockRedisSaver + 对话快照） | 多轮对话记住上文，重启后 mock 数据可回放 |
| 5 | Store 工厂（InMemoryStore + MockPostgresStore + 画像）+ memory_service（提取器 + 摘要，见 6.6） | 偏好注入 Prompt；对话中说"我喜欢X"后跨会话生效 |
| 6 | RAG：MockMilvus + 种子语料 + ingest + 权限 expr + 双路召回 + output_guard + knowledge 路由 + vision 节点 | alice 检索不到市场部文档；引用校验生效；`test_permissions.py` 全绿 |
| 7 | 前端页面（登录/会话/流式渲染/图片上传） | 浏览器完成登录→提问→流式输出全流程 |
| 8 | 全链路联调 + `scripts/docker-compose.yml`（见 14.2）+ 切换真实后端操作文档 | 对照阶段验收点逐项通过；pytest 全量通过 |

### 切换真实后端步骤（阶段 8 后）

1. 一键起依赖：`docker compose -f scripts/docker-compose.yml up -d`（postgres / redis / milvus 三件套，见 14.2）
2. 安装 `asyncpg sqlalchemy langgraph-checkpoint-postgres langgraph-checkpoint-redis` + `alembic`（迁移管理）
3. PostgreSQL 建表：**引入 Alembic，初始 migration 基于第 5 节 DDL 生成**（`alembic revision --autogenerate` 与 DDL 比对校验；**不含 store/checkpoints**，由 langgraph 库自建）→ 写 SQLAlchemy 2.0 async 模型（请求级 AsyncSession、连接池见 4.5 约定）→ `.env` 改 BACKEND 开关与连接配置（14.3 联动表）。**后续表结构变更（users/sessions 等加字段）一律走 `alembic revision` 生成迁移脚本，禁止手工 ALTER 线上表**
4. Milvus 已随 compose 启动（也可单独部署于 Linux，注意 Milvus Lite 不支持 Windows）→ `VECTOR_BACKEND=milvus`
5. 配置真实 LLM/Embedding API → 重启，业务代码零改动

---

## 16. 风险与注意事项

| 项 | 说明 | 对策 |
|---|---|---|
| Milvus Lite 不支持 Windows | 无法本地起 Lite 实例 | MockMilvus 模拟接口签名，服务端部署后仅改配置 |
| 中文乱码 | Windows 控制台/文件编码 | 全部文件 UTF-8，控制台输出按 Windows 适配，生成后检查乱码 |
| 敏感信息入库 | JWT_SECRET / API_KEY 泄露 | 走 .env，.gitignore 排除，仓库只留 .env.example |
| XSS 窃取 token | 前端出现 XSS 时脚本可读 localStorage 中的 token | 渲染硬性约束：一律 textContent、禁 innerHTML（9.2）；未来 refresh 改 httpOnly Cookie（4.1 预留，需配套 CSRF 防护） |
| SSE 挂死/悬挂流 | 上游异常导致前端收不到终态帧 | 终态帧互斥（done/error 二选一）+ chat_service `finally` 兜底补发 error 后关闭流（见 4.1 SSE 帧协议） |
| 同一会话并发写入 | 同 thread_id 两个 astream 并发（连点发送/双开标签页/重试），checkpoint 交错或覆盖，历史错乱；官方 OSS checkpointer 无 thread 级互斥 | 三层防护：前端发送中禁用按钮（9.2）+ 服务端 thread 级 asyncio.Lock 409 拒绝（4.5）+ 多实例时 Redis SETNX（预留） |
| 日志泄露敏感信息 | 密码/token/正文进入日志 | 13.2 敏感信息红线：token 只记 jti 前 8 位，正文不落日志，code review 纳入检查 |
| 权限 expr 回归 | 过滤条件拼错导致越权 | expr 集中在 `_build_permission_expr` 一处 + 召回后二次校验 + 隔离用例纳入阶段 6 验收 |
| 功能边界 | 已完成功能不做无关改动 | 每阶段只动本阶段涉及文件 |

---

## 17. 默认假设（如有偏差，实施前可调整）

1. 垂直领域 = 企业内部知识助手（人事部 + 市场部模拟）；换领域只换种子语料。
2. LLM = OpenAI 兼容接口（可配 base_url），默认附 mock 模式便于无 key 联调。
3. 嵌入 = 默认确定性哈希伪向量，预留 API 嵌入。
4. 会话标题 = 文本首句截断 20 字；首条消息含图片按 4.2 session_service 规则（后续可换模型生成）。
5. 图片存储 = 本地 uploads/ 目录，经代理端点访问（7.1），不挂静态目录。
6. 移动端 = 仅基础响应式布局（768px 断点 + 抽屉式会话列表），无 PWA/专用移动端。
