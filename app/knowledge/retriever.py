"""
检索器（Retriever）
====================
把"向量化 + Milvus 检索 + 上下文拼装"封装为统一的 RAG 检索入口。

对外提供两个能力：
- retrieve(query)：纯检索，返回命中的知识块列表（带分数与来源）；
- build_context(query)：检索并拼装成可直接喂给 LLM 的上下文文本（含引用编号）。

检索结果带 Redis 缓存（可选，无 Redis 时自动跳过）：一次检索包含一次外部
embedding 调用 + 一次向量检索，而知识库对**所有用户是同一份**，同样的问法
反复出现时完全可以复用。
"""

import hashlib
import json
from typing import Dict,List,Optional

from app.core import redis_client
from app.core.embedding import get_embedder
from app.core.logger import get_logger
from app.database import milvus_client
from config.settings import settings

logger=get_logger(__name__)

# 缓存 key 里必须带上这些"会改变检索结果"的因素：换了 embedding 模型、
# 切了集合、改了 top_k 或阈值，都应当视为不同的查询，否则会拿到不适用的旧结果。
def _cache_key(query: str, top_k: int, threshold: float) -> str:
    raw = "|".join([
        settings.MILVUS_COLLECTION,
        settings.EMBEDDING_API_MODEL,
        str(top_k),
        str(threshold),
        query.strip(),
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return redis_client.key("retrieve", digest)


def invalidate_cache() -> int:
    """
    清空检索缓存（知识库重新入库后必须调用，否则会继续返回旧内容）。
    返回清理的 key 数量。
    """
    client = redis_client.get_redis()
    if client is None:
        return 0
    try:
        # scan_iter 而不是 KEYS：KEYS 会阻塞 Redis
        keys = list(client.scan_iter(match=redis_client.key("retrieve", "*"), count=500))
        if keys:
            client.delete(*keys)
        logger.info("已清理检索缓存 %s 条", len(keys))
        return len(keys)
    except Exception as e:
        redis_client.mark_down()
        logger.warning("清理检索缓存失败（不影响检索）: %s", e)
        return 0


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """执行向量检索，返回命中的知识块（按相似度降序）。优先读缓存。"""
    effective_top_k = top_k or settings.RETRIEVE_TOP_K
    threshold = settings.RETRIEVE_SCORE_THRESHOLD
    cache_key = _cache_key(query, effective_top_k, threshold)

    client = redis_client.get_redis()
    if client is not None:
        try:
            cached = client.get(cache_key)
            if cached is not None:
                logger.info("检索命中缓存: %s", query[:20])
                return json.loads(cached)
        except Exception as e:
            redis_client.mark_down()
            logger.warning("读检索缓存失败，改为直接检索: %s", e)

    embedder = get_embedder()
    query_vector = embedder.embed_query(query)
    hits = milvus_client.search(query_vector, top_k=top_k)

    if client is not None:
        try:
            client.setex(
                cache_key, settings.RETRIEVE_CACHE_TTL_SECONDS, json.dumps(hits, ensure_ascii=False)
            )
        except Exception as e:
            redis_client.mark_down()
            logger.warning("写检索缓存失败（不影响检索）: %s", e)

    return hits

def build_context(query: str, top_k: int = None) -> tuple[str, List[Dict]]:
    """
    检索并把结果拼装为带引用编号的上下文。
    返回 (context_text, hits)：
      context_text: 供 LLM 阅读的检索片段（[1] 文档标题 内容...）
      hits:         原始命中列表（供溯源展示）
    """
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return "", []

    parts = []
    for i, hit in enumerate(hits, start=1):
        # 引用编号 + 来源标题 + 正文，让 LLM 回答时能对应 [1][2]...
        parts.append(f"[{i}]（来源：{hit['title']}）{hit['text']}")

    context = "\n\n".join(parts)
    logger.info("检索到 %s 条相关内容", len(hits))
    # logger.info(f"传入模型的文档：{context}")
    return context, hits