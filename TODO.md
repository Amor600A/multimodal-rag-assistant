# 真实后端切换 TODO（4 大工厂分支 + 隐藏分支 + 验收）

> 前置说明：当前 4 个工厂的真实后端分支均未实现，切开关只会打 warning 并回退 Mock。
> 详见 `docs/switch-to-real-backend.md` 5.1 节。本清单按依赖顺序排列，逐项执行。

## 0. 公共前置

- [ ] **b0-1** 装依赖：`asyncpg sqlalchemy alembic langgraph-checkpoint-postgres langgraph-checkpoint-redis`
  - 解锁 `requirements.txt` 末尾注释块，写入版本约束（>=x.y,<下一 major）
  - Alembic init：初始 migration 只含 4 张业务表（departments/users/sessions/images），DDL 对齐 `scripts/create_db.sql`
  - 禁止包含 store/checkpoints 表（由 langgraph 组件 setup() 自建，手工建会冲突）

## 1. 分支 1：用户仓储 `USER_DB_BACKEND=postgres`

- [ ] **b1-1** `app/infra/user/` 补真工厂分发层（当前 `__init__.py` 只有一行注释，`get_user_repository()` 直接定义在 mock_user.py）
  - 新建 `postgres_user.py`：SQLAlchemy 2.0 Declarative 模型 + 请求级 AsyncSession + 连接池（方案 4.5）
  - 对齐 5 个接口：`create / update_last_login / set_dept / get_by_username / get_by_id`
  - 返回同一 `User` dataclass，事务提交在服务层完成
- [ ] **b1-2** 连带实现同批 SQLAlchemy 模型的另外 3 个仓储（否则切换后会话列表/部门/图片元数据仍落内存）
  - `app/infra/session/postgres_session.py`：create / get / list_by_user / delete / update_title
  - `app/infra/org/postgres_org.py`：部门查询接口对齐 mock_org
  - `app/infra/images/postgres_images.py`：create / get / mark_bound（对齐 mock_images）

## 2. 分支 2：Checkpointer `CHECKPOINTER_BACKEND=redis | postgres`

- [ ] **b2-1** redis 分支：`langgraph-checkpoint-redis` 的 AsyncRedisSaver
  - 关键改造：官方组件是 async context manager 用法，现工厂是同步单例函数
    → 在 lifespan 用 `AsyncExitStack` 管理生命周期，`graph.compile` 前完成初始化
  - 序列化格式与 MockRedisSaver JSON 快照不兼容，切换后历史对话不延续（方案 4.5 已接受）
  - 无 thread 级锁，继续依赖 chat 路由 asyncio.Lock 单飞
- [ ] **b2-2** postgres 分支：AsyncPostgresSaver
  - 首次启动 `await setup()` 自动建 checkpoints 表（禁止手工建）
  - 与 b2-1 共用同一套 AsyncExitStack 生命周期改造

## 3. 分支 3：Store `STORE_BACKEND=postgres`

- [ ] **b3-1** 官方 AsyncPostgresStore 分支（langgraph-checkpoint-postgres 包内）
  - 对齐接口：put / get / search / list_namespaces，namespace 元组 `(user_id, "memories")`
  - 唯一调用方 `app/services/memory_service.py`（偏好注入 + 事实提取 + 摘要）
  - 生命周期随 graph.compile 一起改造（get_store() 目前在 service `__init__` 同步调用，初始化时序要改）
  - 表由 `setup()` 自建（含迁移版本表），禁止手工建 store 表
  - 验收注意：Mock 语义检索返回 namespace 全量，官方 semantic search 需 embedding 配置，行为差异需确认

## 4. 分支 4：向量库 `VECTOR_BACKEND=milvus`

- [ ] **b4-1** `app/infra/vectordb/milvus.py`：pymilvus MilvusClient
  - 对齐 4 个方法：`insert(records) / search(expr, query, limit) / get(record_id) / count()`
  - 调用方 `app/services/rag_service.py`（insert 在入库 pipeline、search 在双路召回）
  - collection schema 新建：向量字段维度 = `EMBEDDING_DIM`，标量字段 `scope / visibility / owner_user_id / dept_id / source / text`
  - rag_service 启动时幂等 ensure collection 与索引（方案 14.4）
  - expr 双向兼容是验收核心：`_build_permission_expr()` 只生成有限子集（字段名、==、字符串字面量、and/or、括号），Mock 解析器与真实 Milvus 对同一字符串都必须可解析
  - Milvus Lite 不支持 Windows，必须连 compose 起的 standalone（内存 >= 4GB）
  - 注意：VECTOR_BACKEND 与 EMBEDDING_BACKEND 是两个独立开关，语义召回质量需同时切 `EMBEDDING_BACKEND=api`

## 5. 隐藏分支：Token 吊销 `TOKEN_REVOCATION_BACKEND=redis`

- [ ] **b5-1** `app/infra/user/token_store.py` 改 Redis 实现
  - 不在四大工厂内，但切真实后端时必须一并改，否则吊销状态仍落进程内存、多实例失效
  - jti 吊销标记写 Redis（带 TTL = token 剩余有效期），与 checkpointer 共用 `REDIS_URI`

## 6. 总验收

- [ ] **b9-1** 全链路联调
  - `.env` 按方案 14.3 联动表改开关
  - 启动日志组件清单为真实后端，无 "fallback 到 ..." warning
  - 按 `docs/switch-to-real-backend.md` 第 7 节 12 项清单逐项复验
  - `pytest tests -q` 全量绿（重点：test_expr.py / test_permissions.py 不回归）

---

**执行顺序**：b0-1 → b1-1+b1-2 → b2-1+b2-2+b3-1（共用生命周期改造）→ b4-1（独立）→ b5-1 → b9-1

**阻断提醒**：b9-1 真连服务验证需本机 Docker 可用（当前未安装）；此前的代码实现可靠 pytest + 静态审查保证质量。

---

# 召回率优化 TODO（独立线，与真实后端切换不冲突）

> 背景：当前库 67 条记录、dashscope_mm 嵌入、COSINE 相似度。跨模态（文→图/视频）分数天然 0.2~0.4，被文-文切片（0.5+）挤出 top-N；`--min-score` 0.5 会把所有媒体召回全部过滤掉。
> 按落地优先级排序，R0 是所有调参的基础。

## R0. 评测集（前置基础）

- [ ] **R0-1** 整理 20~50 条「问题 → 应命中文档」对，字段：问题文本、预期命中的 source（文件名#p页码 / #c块序号）、模态（text/image/video）
  - 覆盖场景：纯文本查询文本、纯文本查询图片、纯文本查询视频、部门隔离查询、public 查询
  - 保存为 `data/eval_set.json`，后续所有调参以 recall@k 量化对比

## R1. 查询脚本分模态过滤（立即见效）

- [ ] **R1-1** `query_knowledge_base.py` 新增 `--file-type` 参数，取值 `text | image | video` 或组合（如 `text,video`）
  - 按 `file_type` 字段预过滤后再查，媒体不再被文本切片挤掉
  - 与 `--min-score` 配合时，可对不同模态设不同阈值（媒体放宽到 0.2）

## R2. Rerank 精排（业界标准）

- [ ] **R2-1** 初检放宽到 top-50 → DashScope gte-rerank 重打分 → 取 top-5
  - 召回率由初检保证，精确率由精排保证
  - `rag_service.py` 双路召回返回后、写入 state 前接入
  - 调试脚本 `query_knowledge_base.py` 同步支持 `--rerank` 开关

## R3. 混合检索（向量 + 关键词）

- [ ] **R3-1** 向量召回 top-25 + BM25 关键词召回 top-25 → RRF 融合取 top-10
  - BM25 对人名、编号、专有名词召回更准
  - 可先用 `rank-bm25` 轻量库验证，生产环境再考虑 es/vector 插件
  - `milvus_lite.py` 新增 `search_hybrid(expr, query, limit)` 接口

## R4. 查询改写

- [ ] **R4-1** LLM 将用户问题扩成 2~3 个不同表述（口语化、书面、关键词式）→ 并行检索 → 结果去重合并
  - 对简短/口语化查询提升明显
  - 独立节点 `query_rewrite_node`，可通过配置开关启用

## R5. 媒体描述入库（根治跨模态分数低）

- [ ] **R5-1** 图片/视频入库前，用多模态 LLM 生成详细 caption（而不是默认文件名）
  - 例：`car.mp4` → 「一辆白色轿车在停车场左前侧剐蹭，车漆受损但无人员受伤」
  - 描述文本写入记录的 `text` 字段，参与嵌入——文→媒体召回变回文→文，分数回归 50%+
  - 入库脚本 `create_knowledge_base.py` 的 `build_media` 分支接入，可配置开关控制是否自动生成

## R6. 切片调优

- [ ] **R6-1** 按文档类型调整切片策略
  - 制度/手册类：按标题层级切（用 `\n## `、`\n### ` 做分隔），避免大段跨标题稀释语义
  - 其他文档：试 `CHUNK_SIZE=300 / CHUNK_OVERLAP=80` 对比当前 `500/50` 的 recall@k
  - 用 R0 评测集量化对比后固化

---

**执行顺序**：R0 → R1 → R2 → R5 → R3 → R4 → R6

**与真实后端切换的冲突点**：R3（混合检索）切到 standalone Milvus 后可考虑用 Milvus 自身的 hybrid search 能力替代自研 RRF；其余项在 Milvus Lite 阶段即可完成验证。
