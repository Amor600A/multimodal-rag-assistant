# 多模态垂直领域 RAG 助手

面向企业内部知识助手场景的多用户、多模态 RAG 智能助手。支持部门级权限过滤的知识检索、图文混合输入、SSE 流式对话、短期/长期记忆。

当前为**全 Mock 模式**：零外部依赖即可启动，所有可切换组件（用户仓储 / Checkpointer / Store / 向量库 / LLM / Embedding）通过工厂模式预留了真实后端接口，切换只需改 `.env` 开关（真实后端分支实现进度见 `TODO.md`）。

---

## 1. 启动准备

### 环境要求

- Windows + Python 3.10+（项目开发使用 3.11.9）
- 无需安装 Docker / PostgreSQL / Redis / Milvus（Mock 模式零依赖）

### 首次准备

```powershell
# 1. 创建虚拟环境（若仓库已带 .venv 可跳过）
python -m venv .venv

# 2. 安装依赖
.venv\Scripts\python -m pip install -r requirements.txt

# 3.（可选）复制配置示例；不复制也能启动，全部走默认值
copy .env.example .env
```

### 启动

```powershell
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

启动日志会打印组件清单（各后端开关当前取值），随后访问：

| 入口 | 地址 |
|---|---|
| 前端页面 | http://127.0.0.1:8001/ |
| API 文档（Swagger） | http://127.0.0.1:8001/docs |
| 健康检查 | http://127.0.0.1:8001/api/health |

### 种子账号（启动时自动注入，幂等）

| 账号 | 密码 | 角色 | 部门 |
|---|---|---|---|
| alice | alice123 | user | 人事部 |
| bob | bob123 | user | 市场部 |
| admin | admin123 | admin | - |

> 权限体验：用 alice 登录问市场部问题（如"竞品分析"），检索不到市场部文档；bob 则可以。

---

## 2. 一次提问的完整流程

```
浏览器(web/)                    FastAPI                          LangGraph Agent
─────────────  POST /api/chat (JWT, message, thread_id, image_id?)
                                │
                                ├─ 鉴权 get_current_user（解析 JWT + 实时查库取 dept_id）
                                ├─ 限流（5 次/分钟/IP）
                                ├─ 会话归属校验（thread_id 必须属于当前用户）
                                ├─ 单飞锁（同用户同会话并发请求排队，防 checkpoint 写冲突）
                                │
                                ▼  astream(stream_mode=["messages","updates"])
                    ┌──────────────────────────────────────────┐
                    │ START                                    │
                    │   │                                      │
                    │   ├─ 携带 image_id ─→ vision 节点         │  图片理解，产出 vision_desc
                    │   │                      │               │
                    │   └─ 无图片 ────────────→│               │
                    │                          ▼               │
                    │                    retrieve 节点          │  权限 expr 过滤 + 双路召回
                    │                          │               │  （vision_desc 并入查询词）
                    │                          ▼               │
                    │                    generate 节点          │  偏好注入 + 图片描述注入 + 回复生成
                    │                          │               │
                    │                          ▼               │
                    │                    output_guard 节点      │  引用编号校验，伪造编号净化
                    │                          │               │
                    │                          END             │
                    └──────────────────────────────────────────┘
                                │
                                ▼  SSE 逐帧推送
   token 帧（逐字）→ warning 帧（如有净化）→ done 帧（终态，与 error 互斥）
                                │
                                ├─ image_id 引用置位 images.bound=TRUE
                                └─ 异步触发长期记忆提取（不阻塞响应）
```

### SSE 帧协议

| event | data.type | 说明 |
|---|---|---|
| message | token | 回复正文逐字片段 |
| node | node | 节点切换通知（vision/retrieve/generate/output_guard） |
| warning | warning | output_guard 净化了伪造引用编号 |
| done | done | 正常终态帧 |
| error | error | 异常终态帧（与 done 互斥，只出现其一） |

---

## 3. 目录结构速览

```
app/
├── main.py                  # 应用入口：lifespan 种子注入 + 路由挂载 + web/ 静态托管
├── core/                    # config(全量开关+条件校验) / security(JWT双token) / deps(鉴权) / logger(日志+审计) / ratelimit
├── api/routes/              # auth(登录/刷新/吊销) chat(SSE) session dept knowledge images(上传+代理) health
├── services/                # auth_service / rag_service(权限expr+双路召回) / memory_service(事实提取+摘要)
├── agent/
│   ├── graph.py             # 图编排：START →(vision)→ retrieve → generate → output_guard → END
│   ├── state.py             # AgentState（messages 累加，user/dept/role，vision_desc，retrieved_chunks）
│   ├── nodes/               # vision(图片理解) retrieve(权限召回) generate(生成) output_guard(引用护栏)
│   ├── tools/               # TOOL_REGISTRY 预留（AGENT_ENABLED_TOOLS 启用，当前为空=纯RAG链路）
│   └── mock_llm.py          # Mock LLM：关键词匹配预设回复
├── infra/                   # 四大可切换组件工厂（真实后端分支见 TODO.md）
│   ├── user/                # 用户仓储（含 token_store 吊销）
│   ├── org/  session/  images/   # 部门 / 会话元数据 / 图片元数据仓储
│   ├── checkpointer/        # memory | mock_redis（磁盘快照，重启可回放）
│   ├── store/               # 长期记忆（mock_postgres，namespace=(user_id,"memories")）
│   └── vectordb/            # mock_milvus（numpy 余弦 + 有限 expr 子集解析器）
├── rag/embedder.py          # hash 嵌入（sha256，重启不漂移）
web/                         # 前端：index.html / app.js / style.css（textContent 渲染，禁 innerHTML）
scripts/                     # create_db.sql（业务表DDL） seed_data.py（幂等种子） docker-compose.yml（依赖编排）
tests/                       # pytest 59 用例：安全/expr/权限/RAG/SSE/会话图片
docs/                        # architecture.svg / switch-to-real-backend.md（切真实后端操作手册）
PROJECT_PLAN.md              # 完整设计方案  TODO.md 真实后端切换任务清单
```

---

## 4. 六大后端开关（.env）

| 开关 | 当前默认 | 可切换值 |
|---|---|---|
| USER_DB_BACKEND | mock_postgres | postgres（未实现，见 TODO.md） |
| CHECKPOINTER_BACKEND | mock_redis | memory / redis / postgres（后两者未实现） |
| STORE_BACKEND | mock_postgres | in_memory / postgres（未实现） |
| VECTOR_BACKEND | mock | milvus_lite（已实现）/ milvus（未实现，见 TODO.md） |
| LLM_BACKEND | mock | openai_compatible |
| EMBEDDING_BACKEND | hash | api（DashScope text-embedding-v1） |

规则：

- Mock 模式零配置可启动；切真实后端时缺失必填连接配置会**启动即报错**，不会静默降级
- `TOKEN_REVOCATION_BACKEND=memory|redis`：token 吊销存储，多实例部署必须 redis
- `AGENT_ENABLED_TOOLS` 留空 = 纯 RAG 链路；填已注册工具名（逗号分隔）启用 ReAct 工具调用

---

## 5. 常用命令

```powershell
# 跑测试（59 用例）
.venv\Scripts\python -m pytest tests -q

# 重复注入种子数据（幂等，不会产生重复数据）
.venv\Scripts\python -m scripts.seed_data

# 起依赖容器（切真实后端联调时才需要，需先装 Docker）
docker compose -f scripts/docker-compose.yml up -d
```

## 6. 开发注意

- 前端渲染一律 `textContent`/`innerText`，禁止 `innerHTML` 插入未过滤内容（防 XSS 窃取 token）
- 图片访问唯一出口是 `GET /api/knowledge/images/{id}` 代理端点（鉴权），`uploads/` 不挂静态目录
- 权限过滤表达式由服务端统一构造，前端不可传入；子集外语法直接抛错，不静默漏过滤
- 表结构变更走 Alembic，禁止手工 ALTER；store/checkpoints 表由 langgraph 自建，禁止手工预建
- 切换真实后端：先读 `docs/switch-to-real-backend.md`，任务清单见 `TODO.md`
