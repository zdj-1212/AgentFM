# AgentFM · 企业智能客服

基于 **LangGraph 多智能体 + RAG** 的企业智能客服系统。带完整的用户体系（登录/注册）、
按用户隔离的会话与订单数据，以及自动概括首轮提问的会话标题。

提供三种入口：**网页端（Streamlit）**、**命令行（CLI）**、**REST API（FastAPI）**。

---

## 功能一览

| 能力 | 说明 |
|---|---|
| 意图路由 | 一个图把用户输入分成 4 类：知识问答 / 订单查询 / 闲聊 / 综合兜底 |
| RAG 知识问答 | Milvus 向量检索 + 相似度阈值过滤 + 引用溯源（回答里标 `[1] [2]`） |
| 订单物流查询 | ReAct Agent 调用 MySQL 工具，**身份由框架注入，模型无法伪造** |
| 用户体系 | 注册 / 登录 / 退出，**argon2id** 存口令（pwdlib），**HS256 JWT** 令牌（PyJWT） |
| 令牌吊销 | 退出登录立刻让该账号**已签发的全部令牌**失效，不依赖令牌自然过期 |
| 角色分级 | `user` / `admin` 两种角色；管理员多一个运营视角，普通用户界面完全不变 |
| 运营工作台 | 管理员可看各用户的会话数/消息数/好评率，并下钻到任意会话的消息（只读） |
| 答案反馈 | 每条助手回复可 👍/👎，点踩可补一句原因；反馈进入运营总览，也是将来做评测集的原料 |
| 数据隔离 | 会话与订单都按 `user_id` 隔离；**拿到别人的 session_id 也读不到** |
| 会话标题 | 首轮提问自动概括成短标题（与主流程并发，不增加等待时间） |
| 降级可辨识 | 依赖挂掉时回复兜底话术，同时带 `error_code` 标明是哪个依赖出了问题 |
| 超时保护 | LLM / Embedding 单次调用有显式超时，挂死的依赖不会把请求拖住 |
| 认证限流 | 登录/注册按来源与账号双维度限流，且**在昂贵的口令哈希之前**拦截 |
| Redis 加速 | 限流计数全局一致；重复问句复用检索结果。都是"没有 Redis 也能跑"的加速 |

---

## 技术栈

- **编排**：LangGraph 1.x + LangChain 1.x（`create_agent` ReAct Agent）
- **模型**：任何 OpenAI 兼容接口（通义千问 / DeepSeek / Ollama / vLLM 均可）
- **向量库**：Milvus 3.0（COSINE，IVF_FLAT）
- **数据库**：MySQL 9.x（SQLAlchemy 2.0 ORM）
- **缓存/限流**：Redis（**可选**，没有也能跑：限流退回进程内、检索不走缓存）
- **服务**：FastAPI（REST）+ Streamlit（网页）+ 命令行（`main.py`）
- **认证与加密**：pwdlib（argon2id 口令哈希）+ PyJWT（HS256 令牌）
- **依赖管理**：uv

---

## 快速开始

### 0. 前置条件

- Python **3.12+** 与 [uv](https://docs.astral.sh/uv/)
- Docker（用于起 MySQL + Milvus + Redis；也可以自己装，见下文"不用 Docker"）
- 一个 OpenAI 兼容的 LLM / Embedding API Key

### 1. 准备配置

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

编辑 `.env`，**至少**填这两项：

```ini
LLM_API_KEY=sk-你的key
EMBEDDING_API_KEY=sk-你的key
```

> ⚠️ 部署到公网前还必须改 `AUTH_SECRET_KEY`。它默认是开发用的固定字符串，
> 谁拿到这个值就能伪造任意用户的登录令牌。生成一个随机值：
> `python -c "import secrets;print(secrets.token_urlsafe(48))"`
>
> 顺带一提：HMAC-SHA256 建议密钥至少 **32 字节**，而默认值只有 28 字节。
> 启动时如果密钥偏短会打印一条警告（PyJWT 自己本来会**每个请求**都告警一次，
> 那会把日志刷满，所以改成启动时统一提醒一次）。

### 2. 起依赖服务（MySQL + Milvus + Redis）

```bash
docker compose up -d
docker compose ps        # 等 5 个服务都变成 healthy（Milvus 首次启动约 1 分钟）
```

会拉起 5 个容器：`mysql`、`redis`、`etcd`、`minio`、`milvus`。其中 **etcd 和 minio 是 Milvus
自身的依赖**（分别存元数据和对象存储），不是应用直接用的。

**Redis 是可选的**：只用 `mysql` + `milvus` 也能完整跑通，只是限流变成单机精度、
检索不走缓存（想显式关掉就设 `REDIS_ENABLED=false`）。
如果已经有自己的 Redis，把 `.env` 里的 `REDIS_HOST/PORT/PASSWORD` 指过去即可，
**记得填 `REDIS_PASSWORD`**——密码填错的表现是"静默退化成没有 Redis"，
日志里只有一句"Redis 不可用"，不会报错。

端口默认与 `.env` 对齐（`3306` / `19530`），所以起完不用改任何配置。

> **已经有 MySQL / Milvus 的话，跳过这一步**，直接把 `.env` 里的 `MYSQL_*` / `MILVUS_URI`
> 指向现有服务即可。本机若已在别处跑着 Milvus（比如另一个 compose 项目），
> 再执行本文件起容器会**端口冲突**。

端口被占用时可以覆盖：

```bash
MYSQL_PORT=13306 MILVUS_PORT=29530 docker compose up -d
# 然后同步把 .env 里的 MYSQL_PORT / MILVUS_URI 改掉
```

### 3. 初始化数据并安装依赖

```bash
uv sync                              # 安装 Python 依赖
uv run python -m scripts.init_db     # 建库建表 + 补列迁移 + 演示账号/订单
uv run python -m scripts.ingest_kb   # 把知识库语料向量化灌进 Milvus
```

两个脚本都是**幂等**的，重复执行没问题。`init_db` 会创建 3 个演示账号与 5 笔订单。

### 4. 启动

```bash
# 网页端（推荐先试这个）
uv run streamlit run app/streamlit_app.py

# 命令行
uv run python main.py

# REST API
uv run uvicorn app.api_server:app --host 0.0.0.0 --port 8000
# 接口文档： http://127.0.0.1:8000/docs
```

### 演示账号

| 用户名 | 密码 | 角色 | 名下有订单 | 能看到的 |
|---|---|---|---|---|
| 张三 | `123456` | **管理员** | 2 笔 | 自己的会话 + 运营工作台（所有人的数据） |
| 李四 | `123456` | 普通用户 | 2 笔 | 只有自己的会话与订单 |
| 王五 | `123456` | 普通用户 | 1 笔 | 同上 |

登录页上也写了这几个账号，方便直接体验。**注意**：`init_db()` 在应用启动时也会跑一遍
（见 `api_server.py` 的 `lifespan`），所以只要种子逻辑还在，演示账号就会被自动重建。
真要上线，记得把这套种子账号摘掉或加开关（`mysql_client.py` 的 `DEMO_USERS`）。

试试这几步，能最快看出角色与隔离的区别：

1. 用 **张三** 登录 → 侧边栏显示 `管理员`，多出「运营工作台」页签；问"我的订单有哪些？"能查到 2 笔
2. 在对话里点一条回复的 👍/👎（点踩还能补原因）→ 回到运营工作台，好评率随之变化
3. 换 **李四** 登录 → 没有运营页签，会话列表是空的，查订单也查不到张三那两笔

---

## 目录结构

```
config/settings.py          全局配置（所有配置项唯一出处）
app/
  agent/                    LangGraph 工作流
    graph.py                图的组装与编译
    nodes.py                各节点：分类 / RAG / 订单 / 闲聊 / 综合 / 落库
    tools.py                ReAct 工具（查订单、查物流、查知识库）
    state.py                AgentState + Intent + ErrorCode + UserContext
    prompts.py              全部提示词（集中管理，便于调优）
  services/
    auth_service.py         注册/登录/口令哈希(argon2id)/JWT 签发与校验/角色判定
    chat_service.py         对话主流程、会话标题生成、答案反馈
    admin_service.py        管理员能力（跨用户只读：总览/会话/消息/反馈汇总）
  database/
    mysql_client.py         ORM 模型 + DAO + 幂等迁移
    milvus_client.py        集合管理 + 向量检索
  core/                     LLM、Embedding、日志
  knowledge/                切分、入库、检索 + corpus/ 语料
  api_server.py             FastAPI 入口
  streamlit_app.py          Streamlit 入口
main.py                     CLI 入口
scripts/                    init_db / ingest_kb / smoke_test
docker-compose.yml          依赖服务编排（MySQL + Milvus + Redis）
```

---

## 设计要点

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

注册接口则**永远不接受角色字段** —— `RegisterRequest` 里没有这个字段，
`create_user()` 也不接受 `role` 参数，角色只能由种子数据或后台改库设置。
否则任何人都能一键自封管理员。冒烟测试里专门有一条用例往注册请求里塞 `role=admin` 来验证它被忽略。

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
原因是手写那两个最关键的决定都得自己拍：

- **迭代次数是我自己定的**（26 万次 PBKDF2-SHA256），而 OWASP 对它的现行建议是 **60 万次**，
  也就是这个值本来就偏低，而且需要有人持续跟进硬件发展。
- **密钥长度、算法白名单、时钟偏移**这类细节，写了不一定写全。

换掉之后：口令用 **argon2id**（每次哈希占 64 MiB 内存，防护来自内存硬度而不是堆迭代次数，
参数由库维护）；令牌用 **PyJWT**，它还会主动警告过短的 HMAC 密钥——正好是本项目原先只在
文档里提醒过的那件事。

> **顺带提醒：不要选 passlib。** 它 2020 年后没有再发布，且与 bcrypt 5.x 已不兼容——
> 实测哈希一个普通口令会直接抛 `ValueError`。pwdlib 正是为替代它而出现的。

**迁移是怎么做的（已完成）**：

库里存的是哈希不是明文，**没法批量重算**，所以当时做成"登录时就地升级"——
旧格式仍能验，验过之后重写为 argon2id。因为口令是已知的演示口令，
最后是直接跑了一遍升级把 4 个演示账号全部迁完，没有让你手动逐个登录。

迁完之后，那段兼容校验逻辑**已经删除**（保留它等于让一套废弃算法长期留在攻击面上）。
现在 `auth_service` 只认 argon2id；碰到非 argon2 的哈希会打一条明确的 error 日志并拒绝登录
——这是为了让"从旧备份恢复的库"能一眼看出问题，而不是让用户看到一句莫名的"密码明明是对的"。

### 会话标题与主流程并发

标题用 LLM 概括首轮提问生成，但**不额外增加用户等待时间**：标题请求丢进线程池，
与耗时数秒的图调用并行跑，图跑完再取结果。落盘放在 `finally`，所以即使 Agent 失败，
标题也会写上，不会永远停在"新会话"。

---

## 配置说明（`config/settings.py`）

配置只有一个出处：`config/settings.py`，同名环境变量可覆盖（环境变量优先级高于 `.env`）。
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
| `AUTH_TOKEN_TTL_HOURS` | `72` | 登录令牌有效期 |
| `AUTH_RATE_LIMIT_IP_PER_MINUTE` | `20` | 同一来源每分钟的认证尝试上限 |
| `AUTH_RATE_LIMIT_USER_FAILURES` | `5` | 同一账号的失败次数上限（窗口 5 分钟） |
| `TRUST_PROXY_HEADERS` | `false` | 是否信 `X-Forwarded-For` 取真实 IP，仅在可信反代之后才可开 |
| `REDIS_ENABLED` | `true` | 关掉则限流纯进程内、检索不缓存 |
| `REDIS_PASSWORD` | 空 | 你的 Redis 有密码就必须填，否则会静默退化成"没有 Redis" |
| `RETRIEVE_CACHE_TTL_SECONDS` | `600` | 检索缓存有效期（重新入库时会主动清缓存） |
| `REACT_MAX_ITERATIONS` | `6` | 工具循环上限，防止模型反复调工具 |

---

## 测试

```bash
uv run python -m scripts.smoke_test
```

端到端冒烟测试，11 个阶段：MySQL/建表/演示数据 → 认证 → 数据隔离 → Milvus →
**检索相关性** → **认证限流** → **令牌吊销** → **Redis 加速层** → **角色与反馈** →
**降级可辨识** → 工作流四种路由 + 标题。

会真实调试库、Milvus 和 LLM，**不 mock**。过程中创建的临时账号与会话会自行清理，
可以反复跑而不污染演示数据。

```bash
uv run python -m scripts.smoke_test      # 全量
```

单独验证某一类问题时，也可以直接在代码里调用其中某个 `check_*` 函数。

---

## 排错

**界面能开但一问三不知（老答"知识库中暂无相关信息"）**
先确认 Milvus 起来了（`docker compose ps`，或 `curl localhost:19530`），再确认语料已入库：

```bash
uv run python -m scripts.ingest_kb
```

**改了切分/换 embedding 模型后检索结果很怪**
Milvus 集合是**追加式**的，重复 `ingest_kb` 不会覆盖而是新增重复切片，
且换 embedding 模型后向量维度/语义空间都变了。这种情况下必须重建：

```bash
uv run python -m scripts.ingest_kb --reset    # 先 drop 再重建
```

**`EMBEDDING_DIM` 与模型不匹配**
集合的向量维度由 `EMBEDDING_DIM` 决定，必须与 embedding 模型输出一致
（`text-embedding-v3` → 1024，`bge-small-zh-v1.5` → 512）。不一致会在入库时报错。

**不想用 Docker**
自己装好 MySQL 9.x 与 Milvus 3.0（Milvus 还需要 etcd + MinIO），
把它们监听到 `127.0.0.1:3306` / `127.0.0.1:19530`，或改 `.env` 里的 `MYSQL_*` / `MILVUS_URI`。
`docker-compose.yml` 可以当作版本与参数的参考。

---

## 已知限制

- **限流精度取决于是否配了 Redis**：配了 Redis 时限流计数是全局一致的（多进程/重启都算数）；
  没配（或 Redis 不可用）则退回进程内计数，此时多 worker 部署下实际放行量是
  配置值 × 进程数，且重启清零。要严格限流仍建议在网关层再加一道。
- **超时只约束单次调用**：LLM/Embedding 的单次调用有超时（见配置表），但图里是多次
  串行调用，整体最坏耗时仍约为 `步数 × 超时 × (重试+1)`，没有请求级的墙钟预算。
- **退出登录是"退出所有设备"**：令牌版本按用户维度计，所以退出会把该账号的全部令牌一起
  作废，而不是只作废发起退出的那一个。单令牌精确吊销需要维护已吊销令牌名单，
  会引入清理与多进程一致性问题，本项目没做。
- **令牌没有随机成分**：payload 只有 `sub/u/ver/iat/exp`，同一秒内两次登录会签出完全相同的
  令牌。不影响吊销（按用户版本判断，不依赖令牌唯一性），但如果将来要做"单设备登出"，
  就必须先给令牌加上唯一 id（`jti`）。
- **没有做密码轮换 / 强度校验**：只有长度下限，不查弱口令字典、不强制定期更换。
- **没有转人工**：有管理员视角（能看、能分析），但没有会话分配/认领/接管的工作流，
  也没有"用户一键转人工"的入口。提示词里让用户拨打客服热线，系统内没有对应流程。
- **运营台是只读的**：管理员不能代用户提问、不能改用户数据。那类写操作会显著放大
  管理员账号被拿下的后果，需要配套审计日志，本项目不做。
- **反馈没有防刷**：同一个人可以对同一条回复反复改评价（保留最后一次），没有做频次限制
  或去重统计。当质量信号用时够，当指标用时需要再加工。
- **无评测集**：没有 golden Q/A 与 recall@k 度量。冒烟测试里的检索断言是防回归用的，
  不等于质量评估。
- **无 CI / 代码检查**：未配置 CI、pre-commit、ruff/black/mypy。
