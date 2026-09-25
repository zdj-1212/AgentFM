# AgentFM · 企业智能客服

基于 **LangGraph 多智能体 + RAG** 的企业智能客服系统。它不是"把问题丢给大模型"的
聊天 Demo，而是一套完整的服务：有用户体系、按用户隔离的数据、人工客服接入流程，
以及运营侧的会话与质量看板。

同一套 Agent 能力提供三种入口：**网页端（Streamlit）**、**命令行（CLI）**、**REST API（FastAPI）**。

```
用户提问 -> 意图路由 -> 知识问答(RAG) / 订单查询(ReAct Agent) / 闲聊 / 综合兜底
                              |
                    答不好？一键转人工 -> 坐席接入 -> 人工回复 -> 交回机器人
```

---

## 这个项目演示了什么

如果只看代码，建议从这几处入手——它们是这个项目里真正花过心思、也踩过坑的地方：

| 看点 | 在哪 |
|---|---|
| **让大模型查订单，但模型不能伪造身份** | `app/agent/tools.py` + `app/agent/state.py` |
| **多租户隔离做两层**：列表过滤 + 归属校验，拿到 session_id 也读不到别人的对话 | `app/services/chat_service.py` |
| **三种角色职责分离**：管理员只读，坐席能回复，互不包含 | `app/services/{admin,agent}_service.py` |
| **转人工的状态机**，含"机器人必须闭嘴但消息仍要入库" | `app/agent/state.py` + `app/services/agent_service.py` |
| **并发认领不靠分布式锁**，靠一条带条件的原子 UPDATE | `app/database/mysql_client.py` |
| **降级可辨识**：依赖挂掉时不只是兜底话术，还带 `error_code` 指明是哪个依赖 | `app/agent/state.py` |
| **Redis 只做"没有也能跑"的加速**，并明确记录了哪些地方**不该**用缓存 | `app/core/redis_client.py` |

---

## 功能一览

| 能力 | 说明 |
|---|---|
| 意图路由 | 一个图把用户输入分成 4 类：知识问答 / 订单查询 / 闲聊 / 综合兜底 |
| RAG 知识问答 | Milvus 向量检索 + 相似度阈值过滤 + 引用溯源（回答里标 `[1] [2]`） |
| 订单物流查询 | ReAct Agent 调用 MySQL 工具，**身份由框架注入，模型无法伪造** |
| 用户体系 | 注册 / 登录 / 退出，**argon2id** 存口令（pwdlib），**HS256 JWT** 令牌（PyJWT） |
| 令牌吊销 | 退出登录立刻让该账号**已签发的全部令牌**失效，不依赖令牌自然过期 |
| 角色分级 | `user` / `agent` / `admin` 三种角色，**职责分离**（见下） |
| 转人工 | 用户一键呼叫坐席：机器人停止作答 → 坐席接入 → 人工回复 → 结束交回机器人 |
| 运营工作台 | 管理员可看各用户的会话数/消息数/好评率，并下钻到任意会话的消息（只读） |
| 答案反馈 | 每条助手回复可标「有用 / 没用」，可补一句原因；反馈进入运营总览，也是做评测集的原料 |
| 数据隔离 | 会话与订单都按 `user_id` 隔离；**拿到别人的 session_id 也读不到** |
| 会话标题 | 首轮提问自动概括成短标题（与主流程并发，不增加等待时间） |
| 降级可辨识 | 依赖挂掉时回复兜底话术，同时带 `error_code` 标明是哪个依赖出了问题 |
| 超时保护 | LLM / Embedding 单次调用有显式超时，挂死的依赖不会把请求拖住 |
| 认证限流 | 登录/注册按来源与账号双维度限流，且**在昂贵的口令哈希之前**拦截 |
| Redis 加速 | 限流计数全局一致；重复问句复用检索结果。都是"没有 Redis 也能跑"的加速 |

---

## 架构与数据流

```
  Streamlit 网页 ─┐
  CLI (main.py) ──┼─> chat_service ──> LangGraph 工作流
  REST API ───────┘   (会话/历史/       ├ classify  LLM 结构化分类
                      标题/反馈)        ├ route     按意图分派
                                       ├ rag       Milvus 检索 -> LLM 生成
                                       ├ order     ReAct Agent + MySQL 工具
                                       ├ chitchat  直接 LLM
                                       └ general   兜底 Agent（检索 + 工具）
                                              │
                                       finalize 落库 + 并发生成会话标题
                                              │
                              MySQL（用户/会话/消息/订单）  Milvus（知识库向量）
                              Redis（限流计数/检索缓存，可选）
```

分层职责：

- **入口层**（`streamlit_app.py` / `main.py` / `api_server.py`）：只负责传参与展示，
  业务逻辑一律下沉到 service 层；
- **服务层**（`app/services/`）：会话、认证、权限、转人工。**所有数据隔离与权限判定都在这一层**，
  不依赖入口层"记得过滤"；
- **Agent 层**（`app/agent/`）：LangGraph 图与提示词，只关心"怎么答"，不关心"谁能看"；
- **数据层**（`app/database/`）：ORM 与 DAO，含轻量幂等迁移（不引 alembic）。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph 1.x + LangChain 1.x（`create_agent` ReAct Agent） |
| 模型 | 任何 OpenAI 兼容接口（通义千问 / DeepSeek / Ollama / vLLM 均可） |
| 向量库 | Milvus 3.0（COSINE + IVF_FLAT） |
| 数据库 | MySQL 9.x（SQLAlchemy 2.0 ORM） |
| 缓存/限流 | Redis 6.x（**可选**） |
| 服务 | FastAPI（REST）+ Streamlit（网页）+ 命令行 |
| 认证与加密 | pwdlib（argon2id）+ PyJWT（HS256） |
| 依赖管理 | [uv](https://docs.astral.sh/uv/) |

---

## 快速开始

### 前置条件

- Python **3.12+** 与 [uv](https://docs.astral.sh/uv/)
- Docker（用于一键拉起 MySQL + Milvus + Redis；也可自行安装，见「不用 Docker」）
- 一个 OpenAI 兼容的 **LLM + Embedding** API Key

### 1. 准备配置

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

编辑 `.env`，**至少**填这两项：

```ini
LLM_API_KEY=sk-xxxx
EMBEDDING_API_KEY=sk-xxxx
```

然后**必须**改掉令牌签名密钥——这一步不是"建议"，不改就跑不起来：

```bash
python -c "import secrets;print(secrets.token_urlsafe(48))"
```

把输出填给 `AUTH_SECRET_KEY`。`.env.example` 里那串默认值是公开的，
谁拿到它都能伪造任意用户的登录令牌，所以**应用启动时会检测到并直接拒绝运行**
（报错信息里会写清楚怎么改）。想先跳过这项检查快速跑通，可以临时设 `DEBUG=true`。

### 2. 起依赖服务

```bash
docker compose up -d
docker compose ps        # 等 5 个服务都 healthy（Milvus 首次启动约 1 分钟）
```

会拉起 5 个容器：`mysql`、`redis`、`etcd`、`minio`、`milvus`。其中 **etcd 与 minio 是 Milvus
自身的依赖**（元数据与对象存储），应用不直接使用。

- **Redis 可选**：只用 `mysql` + `milvus` 也能完整跑通，只是限流退化为单机精度、检索不走缓存
  （要显式关掉就设 `REDIS_ENABLED=false`）。
- 端口默认与 `.env` 对齐（`3306` / `19530`），起完不需要改配置。端口被占用时：
  `MYSQL_PORT=13306 MILVUS_PORT=29530 docker compose up -d`，并同步改 `.env`。
- 已经有 MySQL / Milvus，把 `.env` 指向现有服务即可，跳过这一步。

### 3. 初始化数据

```bash
uv sync                              # 安装依赖
uv run python -m scripts.init_db     # 建库建表 + 补列迁移 + 演示账号/订单
uv run python -m scripts.ingest_kb   # 知识库语料向量化入库
```

两个脚本都是**幂等**的，重复执行没问题。`init_db` 会创建 4 个演示账号与 5 笔订单。

### 4. 启动

```bash
# 网页端（推荐先试这个）
uv run streamlit run app/streamlit_app.py

# 命令行
uv run python main.py

# REST API（接口文档 http://127.0.0.1:8000/docs）
uv run uvicorn app.api_server:app --host 0.0.0.0 --port 8000
```

### 演示账号

| 用户名 | 密码 | 角色 | 名下有订单 | 能看到的 |
|---|---|---|---|---|
| 张三 | `123456` | **管理员** | 2 笔 | 自己的会话 + 运营工作台（全站数据，只读） |
| 李四 | `123456` | 普通用户 | 2 笔 | 只有自己的会话与订单 |
| 王五 | `123456` | 普通用户 | 1 笔 | 同上 |
| 客服小李 | `123456` | **坐席** | 无 | 待接入队列 + 自己正在接待的会话 |

> **这些账号只用于本地演示。** `init_db()` 会在应用启动时再跑一遍（见 `api_server.py` 的
> `lifespan`），所以只要种子逻辑还在，这几个账号就会被自动重建——**公开发布或部署前，
> 请先摘掉这套种子账号**（`mysql_client.py` 的 `DEMO_USERS` / `DEMO_PASSWORD`）。
> 详见文末「安全须知」。

### 上手走一遍：看角色与隔离的区别

1. 用 **张三** 登录 → 侧边栏显示「管理员」，多出「运营工作台」页签；问"我的订单有哪些？"能查到 2 笔
2. 在对话里给一条回复标「有用 / 没用」→ 回到运营工作台，好评率随之变化
3. 换 **李四** 登录 → 没有运营页签，会话列表是空的，查订单也查不到张三那两笔
4. 转人工：用 **李四** 问一句 → 侧边栏点「转人工」→ 页面开始每 3 秒自动刷新；
   再用 **客服小李** 登录 → 「坐席工作台」里出现这条会话 → 点「接入」→ 回一句话
   → 回到李四的窗口，坐席的回复会自动出现（气泡上写着「坐席」）

---

## 设计与取舍

### 身份注入：工具拿得到，模型看不见

订单查询最容易被做错的地方是**把用户名当工具参数交给大模型生成**——那样模型可以被
一句"查张三的订单"诱导去查别人的数据。

本项目用 LangChain 的 runtime context 解决：

```python
@tool
def query_orders_by_user(runtime: ToolRuntime[UserContext, Any]) -> str:
    user_id = _current_user_id(runtime)   # 框架注入，模型看不到这个参数
```

调用侧 `agent.invoke(..., context=UserContext(user_id=...))`。实测该参数**不会**出现在
工具的 args schema 里，模型既看不到也伪造不了。副作用是 Agent 实例与用户无关，
可以全局单例缓存，不必按用户建实例。

### 数据隔离做两层

只做"列表过滤"是不够的——`session_id` 一旦泄露，不带归属校验就能读到别人的对话。
所以 `chat_service` 里：

- `list_sessions` 只返回该用户的会话（第一层）；
- `get_history` / `ask` 会校验会话归属，不属于本人直接抛 `PermissionError`（第二层）。

API 层对跨用户的记录返回 **404 而不是 403**，避免泄露"这个 id 确实存在，只是不属于你"。

### 角色：运营视角走独立入口，而不是放宽隔离

管理员要能看所有人的数据，很容易写成"在原有查询上加个 `if is_admin: 不加 user_id 过滤`"。
那种写法的问题是把两种信任级别混在同一条代码路径上——日后任何一次改动都可能让
普通用户也走到"不加过滤"的分支。

这里改成两个互不重叠的入口：

- `chat_service` 的每条路径**都**带 `user_id`，是"只看自己"的世界，**不认角色**；
- `admin_service` 是"能看所有人"的世界，每个方法的**第一行**都是 `require_admin(user)`。

于是有了两条被断言锁住的性质：管理员的 `/sessions` 仍然只返回他自己的会话
（升级角色不会让普通接口开洞），而管理员也不能给别人的回复打分（运营台是只读的）。

注册接口则**永远不接受角色字段**——`RegisterRequest` 里没有这个字段，`create_user()`
也不接受 `role` 参数，角色只能由种子数据或后台改库设置。否则任何人都能一键自封管理员。
冒烟测试里专门有一条用例往注册请求里塞 `role=admin` 来验证它被忽略。

### 转人工：三个角色各管一段，机器人会主动闭嘴

状态机放在会话上（不是消息上）——一条会话任一时刻只处于一种状态，比在每条消息上
打标记更不容易出现自相矛盾：

```
bot ──[用户点「转人工」]──> pending ──[坐席接入]──> assigned ──[坐席结束]──> closed
                                                                        |
                                      机器人可作答（bot 与 closed 都算）<--+
```

三个关键点：

**1. 角色是"各管一段"，不是层层包含。** `user` 提问、`agent` 接入与回复、`admin` 看运营总览。
**admin 不是 agent 的超集**——管理员掌握的是查看与分析能力，坐席掌握的是对用户说话的能力；
合成一个角色后，一个被盗的管理员账号就能直接向任意用户发消息。所以 `require_agent`
刻意**不放过管理员**，冒烟测试里有一条专门断言"管理员仍不能代坐席回复"。

**2. 机器人必须真的停下来。** 转人工之后 `chat_service.ask` 直接返回、不调用模型。
但**用户补发的消息仍然入库**——坐席需要看到用户转人工之后补充的订单号之类的信息，
如果因为"机器人不答"就把消息丢掉，用户会觉得"我说了但坐席看不到"。
机器人恢复作答的时机是坐席点「结束」（状态回到 `closed`）。

**3. 认领靠数据库的原子条件更新，不加分布式锁。** 两个坐席同时点「接入」时，
更新语句带上 `WHERE handoff_status = 'pending'`，只有一个人能改到行，另一个拿到 403。
这个手法和 `token_version` 的原子自增是同一个思路：把并发控制交给一条语句，
而不是"先查再写"（那样两个人都查到 pending，都会以为自己抢到了）。

人工回复在 `messages.role` 里记成 `'agent'`，与机器人的 `'assistant'` 区分开：
前端据此决定气泡上写「坐席」还是机器人、以及要不要显示评价按钮
（对人工回复打分没有意义，打分对象是机器人的回答质量）。

### Redis：只做"没有也能跑"的加速，不当依赖

Redis 在这个项目里只承担两件事，且**都是拿不到就退化**的：

| 用途 | 有 Redis | 没有 Redis |
|---|---|---|
| 认证限流计数 | 多进程/多实例共享，重启不清零 | 退回进程内计数（单机精度） |
| 检索结果缓存 | 重复问句省掉一次 embedding + 向量检索（实测 158ms → 0.8ms） | 每次实打实检索 |

所以 Redis 挂了最坏情况是"限流退回单机水平 + 检索慢一点"，登录和问答都不会不可用。
实现上有两个刻意的选择：

- **连接失败有冷却期**：连不上不是每次请求都重试（那会让每次登录都白等一个连接超时），
  而是进入 30 秒冷却，期间直接用退化路径，并且只警告一次；连接中途失效会立刻断开丢弃。
- **用 key 前缀而不是切 DB**：`agentfm:` 前缀与同一个 Redis 上别人的数据隔离，
  而且比 `SELECT` 切库通用（Redis Cluster 不支持切库）。

**刻意没有用 Redis 的地方**（这些看着像能优化，实际上会引入错误）：

- **令牌吊销状态**：本来每次鉴权就要读一次用户行（顺便确认账号还在），换成缓存反而会引入
  "已吊销的令牌在缓存过期前仍然有效"的漏洞窗口——安全判定不该有陈旧副本。
- **助手回复本身**：RAG 的提示词里带了对话历史，同一个问题在不同上下文里的正确回答并不相同，
  缓存会答非所问；而订单类回答里含用户私有数据，跨用户共享会直接泄露。
  相比之下**检索结果**是安全的：知识库对所有用户是同一份，与提问者无关。

### 口令与令牌交给成熟库（以及为什么曾经不是）

这两块最初是**标准库手写**的（PBKDF2 + 自己拼的 HMAC 签名令牌）。后来换成 pwdlib + PyJWT，
原因是手写的两个最关键的决定都得自己拍：

- **迭代次数是拍脑袋定的**（26 万次 PBKDF2-SHA256），而 OWASP 对它的现行建议是 **60 万次**，
  也就是这个值本来就偏低，而且需要有人持续跟进硬件发展；
- **密钥长度、算法白名单、时钟偏移**这类细节，写了不一定写全。

换掉之后：口令用 **argon2id**（每次哈希占 64 MiB 内存，防护来自内存硬度而不是堆迭代次数，
参数由库维护）；令牌用 **PyJWT**，它还会主动警告过短的 HMAC 密钥——正好是本项目原先
只在文档里提醒过的那件事。

> **选库提醒：不要选 passlib。** 它 2020 年后没有再发布，且与 bcrypt 5.x 已不兼容——
> 实测哈希一个普通口令会直接抛 `ValueError`。pwdlib 正是为替代它而出现的。

**存量哈希怎么迁移**：库里存的是哈希不是明文，**没法批量重算**，所以做成了"登录时就地升级"
——旧格式仍能验，验过之后重写为 argon2id。迁完之后，那段兼容校验逻辑**已经删除**
（保留它等于让一套废弃算法长期留在攻击面上）。现在只认 argon2id；碰到非 argon2 的哈希
会打一条明确的 error 日志并拒绝登录——这是为了让"从旧备份恢复的库"能一眼看出问题，
而不是让用户看到一句莫名的"密码明明是对的"。

### 会话标题与主流程并发

标题用 LLM 概括首轮提问生成，但**不额外增加用户等待时间**：标题请求丢进线程池，
与耗时数秒的图调用并行跑，图跑完再取结果。落盘放在 `finally`，所以即使 Agent 失败，
标题也会写上，不会永远停在"新会话"。

---

## API 一览

启动 `uvicorn` 后可在 `/docs` 看到交互式文档。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（无需登录） |
| POST | `/auth/register` | 注册，返回登录令牌 |
| POST | `/auth/login` | 登录，返回登录令牌 |
| GET | `/auth/me` | 当前登录用户信息 |
| POST | `/auth/logout` | 退出登录（吊销该账号全部令牌） |
| POST | `/chat` | 对话（自动建会话或续接指定会话） |
| GET | `/sessions` | 当前用户的会话列表 |
| GET | `/sessions/{sid}/messages` | 某会话历史 |
| POST | `/sessions/{sid}/messages/{mid}/feedback` | 给助手回复打「有用 / 没用」 |
| POST | `/sessions/{sid}/handoff` | 请求转人工 |
| GET | `/agent/queue` | 待接入队列 + 我名下的会话（坐席） |
| GET | `/agent/sessions/{sid}/messages` | 会话消息（坐席） |
| POST | `/agent/sessions/{sid}/claim` | 接入会话（坐席） |
| POST | `/agent/sessions/{sid}/reply` | 以人工身份回复（坐席） |
| POST | `/agent/sessions/{sid}/close` | 结束人工服务（坐席） |
| GET | `/admin/overview` | 运营总览（管理员） |
| GET | `/admin/feedback/summary` | 反馈汇总（管理员） |
| GET | `/admin/users/{uid}/sessions` | 指定用户的会话（管理员） |
| GET | `/admin/sessions/{sid}/messages` | 任意会话的消息（管理员） |

除 `/health` 与 `/auth/*` 外都需要 `Authorization: Bearer <token>`。
**用户身份只从令牌解析，绝不信任请求体里的用户名字段**——否则任何调用方都能自称是别人。

---

## 目录结构

```
config/settings.py          全局配置（所有配置项的唯一出处）
app/
  agent/                    LangGraph 工作流
    graph.py                图的组装与编译
    nodes.py                各节点：分类 / RAG / 订单 / 闲聊 / 综合 / 落库
    tools.py                ReAct 工具（查订单、查物流、查知识库）
    state.py                AgentState + Intent + ErrorCode + UserContext + HandoffStatus
    prompts.py              全部提示词（集中管理，便于调优）
  services/
    auth_service.py         注册/登录/口令哈希(argon2id)/JWT 签发与校验/角色判定
    chat_service.py         对话主流程、会话标题生成、答案反馈、转人工（用户侧）
    agent_service.py        坐席能力（转人工的人工侧：队列/接入/回复/结束）
    admin_service.py        管理员能力（跨用户只读：总览/会话/消息/反馈汇总）
  database/
    mysql_client.py         ORM 模型 + DAO + 幂等迁移
    milvus_client.py        集合管理 + 向量检索
  core/                     LLM、Embedding、日志、限流、Redis
  knowledge/                切分、入库、检索 + corpus/ 语料
  api_server.py             FastAPI 入口
  streamlit_app.py          Streamlit 入口
main.py                     CLI 入口
scripts/                    init_db / ingest_kb / smoke_test
docker-compose.yml          依赖服务编排（MySQL + Milvus + Redis）
```

---

## 配置说明

配置只有一个出处：`config/settings.py`，同名环境变量可覆盖（优先级：环境变量 > `.env` > 默认值）。
常用项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL` | `qwen-plus` | 模型名；换 DeepSeek 用 `deepseek-chat` |
| `LLM_TIMEOUT_SECONDS` | `60` | 单次 LLM 调用超时。**别留空**，否则吃 SDK 的 600 秒默认值 |
| `LLM_MAX_RETRIES` | `1` | 单次调用失败后的重试次数 |
| `EMBEDDING_TIMEOUT_SECONDS` | `30` | 向量化超时（同上，不设则 600 秒） |
| `RETRIEVE_TOP_K` | `4` | 检索返回条数 |
| `RETRIEVE_SCORE_THRESHOLD` | `0.45` | 余弦相似度阈值，低于它的切片被丢弃 |
| `HISTORY_WINDOW` | `6` | 携带最近几轮历史 |
| `TITLE_AUTO_SUMMARY` | `true` | 关掉可省一次 LLM 调用（标题退化为截断问句） |
| `AUTH_SECRET_KEY` | 开发默认值 | **必须改成随机长字符串**；仍是默认值/为空时应用拒绝启动 |
| `AUTH_TOKEN_TTL_HOURS` | `72` | 登录令牌有效期 |
| `AUTH_RATE_LIMIT_IP_PER_MINUTE` | `20` | 同一来源每分钟的认证尝试上限 |
| `AUTH_RATE_LIMIT_USER_FAILURES` | `5` | 同一账号的失败次数上限（窗口 5 分钟） |
| `TRUST_PROXY_HEADERS` | `false` | 是否信 `X-Forwarded-For` 取真实 IP，仅在可信反代之后才可开 |
| `REDIS_ENABLED` | `true` | 关掉则限流纯进程内、检索不缓存 |
| `REDIS_PASSWORD` | 空 | Redis 有密码就必须填，否则会静默退化成"没有 Redis" |
| `RETRIEVE_CACHE_TTL_SECONDS` | `600` | 检索缓存有效期（重新入库时会主动清缓存） |
| `REACT_MAX_ITERATIONS` | `6` | 工具循环上限，防止模型反复调工具 |

---

## 测试

```bash
uv run python -m scripts.smoke_test
```

端到端冒烟测试，12 个阶段：MySQL/建表/演示数据 → 认证 → 数据隔离 → Milvus →
检索相关性 → 认证限流 → 令牌吊销 → Redis 加速层 → 角色与反馈 → 转人工 →
降级可辨识 → 工作流四种路由 + 标题。

它会**真实**连接数据库、Milvus 与 LLM，不 mock 外部依赖。测试过程中创建的临时账号与会话
会自行清理，可以反复执行而不污染演示数据。

测试里有不少断言是针对"看起来实现了、其实没生效"的坑，例如：

- 注册请求里塞 `role=admin` 必须被忽略；
- 拿到别人的 `session_id` 必须读不到历史；
- 转人工后机器人必须停止作答，但用户补发的消息**必须**仍入库；
- 两个坐席并发认领同一会话，只能有一个成功；
- 管理员不能代坐席回复（运营台只读）；
- 不相关的问题检索必须 0 命中（确认相似度阈值真的在起作用）；
- Redis 不可用时限流必须退回进程内且**仍然生效**。

---

## 排错

**界面能开但一问三不知（总答"知识库中暂无相关信息"）**
先确认 Milvus 起来了、语料已入库：

```bash
docker compose ps
uv run python -m scripts.ingest_kb
```

**改了切分方式或换了 embedding 模型后，检索结果很怪**
Milvus 集合是**追加式**的，重复执行 `ingest_kb` 不会覆盖而是新增重复切片；
换 embedding 模型后向量维度与语义空间也都变了。这种情况必须重建：

```bash
uv run python -m scripts.ingest_kb --reset
```

**`EMBEDDING_DIM` 与模型不匹配**
集合的向量维度由它决定，必须与 embedding 模型输出一致
（`text-embedding-v3` 与 `bge-m3` 是 1024，`bge-small-zh-v1.5` 是 512）。不一致会在入库时报错。

**Redis 明明开着却没生效**
检查 `.env` 里的 `REDIS_PASSWORD` 是否填对。填错的表现是**静默退化**——
日志里只有一句"Redis 不可用"，不会报错。启动后可用
`from app.core.rate_limit import auth_limiter; auth_limiter.backend()` 确认当前生效的后端。

**`uv sync` 很慢**
依赖默认走官方 PyPI 源。国内网络可以临时指定镜像加速，不需要改动仓库：

```bash
uv sync --default-index https://mirrors.aliyun.com/pypi/simple/
```

**不用 Docker**
自行安装 MySQL 9.x、Milvus 3.0（Milvus 还需要 etcd + MinIO）与可选 Redis，
监听在 `127.0.0.1:3306` / `127.0.0.1:19530` / `127.0.0.1:6379`，
或改 `.env` 指向别处。`docker-compose.yml` 里的版本与参数可以当参考。

---

## 已知限制

这是一个**演示项目**，以下限制是清楚知道的，不是疏漏：

**安全相关**

- **演示账号与默认密钥**：`123456` 的四个演示账号、以及 `AUTH_SECRET_KEY` 的开发默认值，
  都只适合本地。详见「安全须知」。
- **限流精度取决于是否配了 Redis**：配了则全局一致；没配（或 Redis 不可用）时退回进程内计数，
  多 worker 部署下实际放行量是配置值 × 进程数，且重启清零。严格限流仍建议在网关层再加一道。
- **超时只约束单次调用**：LLM / Embedding 的单次调用有超时，但图里是多次串行调用，
  整体最坏耗时仍约为 `步数 × 超时 × (重试+1)`，没有请求级的墙钟预算。
- **退出登录是"退出所有设备"**：令牌版本按用户维度计，所以退出会把该账号的全部令牌一起作废，
  而不是只作废发起退出的那一个。单令牌精确吊销需要维护已吊销令牌名单，会引入清理与
  多进程一致性问题，本项目没做。
- **令牌没有随机成分**：payload 只有 `sub/u/ver/iat/exp`，同一秒内两次登录会签出完全相同的令牌。
  不影响吊销（按用户版本判断，不依赖令牌唯一性），但如果将来要做"单设备登出"，
  必须先给令牌加上唯一 id（`jti`）。
- **没有做密码轮换 / 强度校验**：只有长度下限，不查弱口令字典、不强制定期更换。

**功能与工程**

- **转人工是"单人接待"**：一条会话同一时刻只能有一个坐席接入，没有转交/协助/主管介入，
  也没有超时自动回收（用户一直等不到坐席就会一直等）。
- **坐席队列没有实时推送**：坐席要手动点「刷新」才能看到新的待接入会话。
  这是刻意的——回复框和列表在同一页，定时刷新会把坐席正在打字的草稿清掉。
  用户侧则是自动刷新的，因为那边没有输入草稿会被影响。
- **反馈没有防刷**：同一个人可以对同一条回复反复改评价（保留最后一次），没有做频次限制或去重统计。
  当质量信号用够了，当指标用需要再加工。
- **没有评测集**：没有 golden Q/A 与 recall@k 度量。冒烟测试里的检索断言是防回归用的，
  不等于质量评估。
- **没有 CI / 代码检查 / 单元测试**：测试入口只有 `scripts/smoke_test.py`（端到端冒烟，
  需要真实依赖在线）；没有 pytest 单元测试、没有 CI、没有 ruff / black / mypy。
  纯逻辑的部分（评分、状态机分支）值得补单元测试。
- **向量化与检索未做混合检索/重排**：目前是纯向量检索 + 阈值过滤。语料规模小的时候够用。

---

## 安全须知

**这个仓库默认配置只适合本地演示，直接部署到公网是不安全的。**

1. **改掉 `AUTH_SECRET_KEY`**。它默认是一个已公开的固定字符串，谁拿到这个值就能伪造任意用户的
   登录令牌。生成随机值：`python -c "import secrets;print(secrets.token_urlsafe(48))"`
   （HMAC-SHA256 建议至少 32 字节，密钥偏短会打印一条警告。）
   **应用会主动检查这一项：密钥仍是默认值或为空时直接拒绝启动**，不会出现"服务正常跑着、
   但其实谁都能伪造令牌"的情况。仅本地开发可设 `DEBUG=true` 跳过。
2. **摘掉演示账号**。`init_db()` 会在应用启动时自动重建 `张三 / 李四 / 王五 / 客服小李`
   四个口令为 `123456` 的账号，其中张三还是管理员。上线前请修改
   `mysql_client.py` 的 `DEMO_USERS` / `DEMO_PASSWORD`，或给种子逻辑加开关。
3. **别用默认的数据库口令**。`MYSQL_PASSWORD=123456` 同样只适合本地。
4. **上 TLS**。本项目自身不处理传输加密，uvicorn 跑的是明文 HTTP。生产环境请在
   反向代理（nginx / Caddy）终止 TLS，或用 `uvicorn --ssl-certfile/--ssl-keyfile`。
5. **限流不是万能的**：没有 Redis 时是单机精度，且没有做账号锁定——重要的场景请在网关层
   再加一道。

`.env` 已被 `.gitignore` 排除，请保持这样：**不要把真实密钥提交进仓库**。

---

## License

[MIT](LICENSE) © 2026 zdj-1212
