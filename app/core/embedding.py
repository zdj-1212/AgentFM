from abc import ABC,abstractmethod
from functools import lru_cache
from typing import List
from config.settings import settings
from app.core.logger import get_logger

logger=get_logger(__name__)

class BaseEmbedding(ABC):
    """向量化统一接口"""

    @abstractmethod
    def embed_documents(self,texts:List[str])->List[List[float]]:
        """批量向量化（入库用）"""

    @abstractmethod
    def embed_query(self,text:str)->List[float]:
        """单条查询向量化（检索用）。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度，创建 Milvus 集合时必须与之一致。"""

class ApiEmbedding(BaseEmbedding):
    """基于 OpenAI 兼容接口的向量化（如通义 text-embedding-v3）"""
    def __init__(self):
        from langchain_openai import OpenAIEmbeddings

        self._client=OpenAIEmbeddings(
            model=settings.EMBEDDING_API_MODEL,
            base_url=settings.EMBEDDING_API_BASE,
            api_key=settings.EMBEDDING_API_KEY,
            check_embedding_ctx_length =settings.CHECK_EMBEDDING_CTX_LENGTH
        )

    def embed_documents(self,texts:List[str]) ->List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self,text:str) ->List[float]:
        return self._client.embed_query(text)

    @property
    def dim(self) -> int:
        return settings.EMBEDDING_DIM

@lru_cache(maxsize=1)
def get_embedder()->BaseEmbedding:
    """全局唯一 embedder 单例。"""
    if settings.EMBEDDING_MODE=="api":
        return ApiEmbedding()

    raise ValueError(f"不支持的 EMBEDDING_MODE: {settings.EMBEDDING_MODE}")
