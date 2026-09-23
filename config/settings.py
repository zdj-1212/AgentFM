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
    MYSQL_DATABASE: str = "agentfm2"

    @property
    def mysql_url(self)->str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4"
        )

    # ---------------- Milvus ----------------
    MILVUS_URI: str = "http://127.0.0.1:19530"
    MILVUS_COLLECTION: str = "agentfm_knowledge"
    EMBEDDING_DIM: int = 512  # 必须与 embedding 模型输出维度一致

    # ---------------- 大模型 LLM ----------------
    # 提供方：openai_compatible
    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1024

    # ---------------- Embedding ----------------
    # 模式：api（OpenAI 兼容接口）
    EMBEDDING_MODE: str = "api"
    EMBEDDING_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_MODEL: str = "text-embedding-v3"
    CHECK_EMBEDDING_CTX_LENGTH:bool=False

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