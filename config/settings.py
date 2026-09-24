"""
全局配置中心
=============
统一从环境变量 / .env 文件读取配置，整个项目只在这里定义配置项。

设计要点（企业级项目的配置规范）：
1. 使用 pydantic-settings：自动完成类型校验、默认值、环境变量注入；
2. 所有连接参数（MySQL / Milvus / LLM / Embedding）集中管理，避免散落各处；
3. 提供 get_settings() 单例缓存，避免重复读取 .env 文件。
"""
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings,SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"
class Settings(BaseSettings):
    """项目全局配置。字段名与 .env 中的键名一一对应（大小写不敏感）。"""
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,env_file_encoding="utf-8",extra="ignore"
    )

    APP_NAME:str ="AgentFM"
    DEBUG:bool =False
    # ---------------- MySQL ----------------
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "123456"
    MYSQL_DATABASE: str = "agentfm"

    @property
    def mysql_url(self)->str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )

    # ---------------- Redis（可选加速层） ----------------
    # Redis 不是必需依赖：拿不到时认证限流退回进程内实现、检索缓存停用，功能不受影响。
    # 启用后：限流计数在多进程/重启后保持一致；语义相同的问题可复用检索结果。
    REDIS_ENABLED: bool = True
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""  # 你的 Redis 有密码就必须在这里填，否则连不上会静默退化
    REDIS_DB: int = 0
    # key 前缀：与同一个 Redis 上别人的数据隔离（比切 DB 通用，Redis Cluster 也适用）
    REDIS_KEY_PREFIX: str = "agentfm:"
    # 检索结果缓存的有效期（秒）。知识库更新时会主动清缓存，所以这里只是兜底上限。
    RETRIEVE_CACHE_TTL_SECONDS: int = 600

    # ---------------- Milvus ----------------
    MILVUS_URI: str = "http://127.0.0.1:19530"
    MILVUS_COLLECTION: str = "agentfm_knowledge"
    # 必须与所选用 embedding 模型的输出维度一致，否则建集合/入库会失败：
    #   text-embedding-v3      -> 1024
    #   BAAI/bge-m3            -> 1024
    #   BAAI/bge-small-zh-v1.5 -> 512
    EMBEDDING_DIM: int = 1024

    # ---------------- 大模型 LLM ----------------
    # 提供方：openai_compatible
    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1024
    # 单次 LLM 调用的超时与重试。不设的话走 openai SDK 的默认值（600 秒超时、2 次重试），
    # 也就是说一次卡住的调用能把用户的请求挂住十几分钟。本项目实际调用通常 1~3 秒，
    # 60 秒已相当宽松。
    # 注意：图里是**多次**串行调用（ReAct 循环最多 REACT_MAX_ITERATIONS*4 步），
    # 所以单次超时只约束单次，整体最坏耗时约为 步数 × 超时 × (重试+1)。
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 1

    # ---------------- Embedding ----------------
    # 模式：api（OpenAI 兼容接口）
    EMBEDDING_MODE: str = "api"
    EMBEDDING_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_MODEL: str = "text-embedding-v3"
    CHECK_EMBEDDING_CTX_LENGTH:bool=False
    # 向量化超时与重试（同样别留空，否则吃 SDK 的 600 秒默认值）
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0
    EMBEDDING_MAX_RETRIES: int = 1

    # ---------------- 检索参数 ----------------
    RETRIEVE_TOP_K: int = 4
    RETRIEVE_SCORE_THRESHOLD: float = 0.45  # 余弦相似度阈值

    # ---------------- 对话记忆 ----------------
    HISTORY_WINDOW: int = 6  # 携带最近几轮历史

    # ---------------- 用户认证 ----------------
    # 令牌签名密钥：生产环境务必通过 .env 覆盖为随机长字符串
    AUTH_SECRET_KEY: str = "agentfm-dev-secret-change-me"
    AUTH_TOKEN_TTL_HOURS: int = 72  # 登录令牌有效期（小时）
    PASSWORD_MIN_LENGTH: int = 6

    # ---------------- 认证限流 ----------------
    # 登录/注册都要跑一次昂贵的口令哈希（argon2id），不限流则既是撞库入口也是纯 CPU 的 DoS 面。
    # 计数在进程内存里：多 worker 部署时每个进程各算各的，严格限流请放到网关层。
    AUTH_RATE_LIMIT_ENABLED: bool = True
    AUTH_RATE_LIMIT_IP_PER_MINUTE: int = 20  # 同一来源每分钟的认证尝试上限（不论成败）
    AUTH_RATE_LIMIT_USER_FAILURES: int = 5  # 同一账号在窗口内的失败次数上限
    AUTH_RATE_LIMIT_USER_WINDOW_SECONDS: int = 300  # 上一条的统计窗口（秒）
    # 是否信任 X-Forwarded-For 里的客户端 IP。
    # 只有"请求必然经过可信反向代理、且代理会覆写该头"时才可开启——
    # XFF 是客户端能自己伪造的，贸然信任等于给攻击者一把绕过限流的钥匙。
    TRUST_PROXY_HEADERS: bool = False

    # ---------------- 会话标题 ----------------
    TITLE_AUTO_SUMMARY: bool = True  # 用 LLM 概括首轮提问作为标题；关闭则直接截断问句
    TITLE_MAX_CHARS: int = 20  # 标题最大字数
    TITLE_TIMEOUT_SECONDS: float = 8.0  # 标题概括的最长等待时间，超时用截断问句兜底

    # ---------------- ReAct Agent ----------------
    # 工具循环最大步数，防止模型无限调用工具
    REACT_MAX_ITERATIONS: int = 6

@lru_cache(maxsize=1)
def get_settings()->Settings:
    return Settings()

settings=get_settings()

if __name__ == '__main__':
    print(ENV_FILE_PATH)