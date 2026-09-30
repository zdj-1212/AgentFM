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
from functools import lru_cache
from typing import Dict,List,Optional

from app.core import redis_client
from app.core.embedding import get_embedder
from app.core.logger import get_logger
from app.database import milvus_client
from config.settings import settings

logger=get_logger(__name__)

# 缓存 key 里必须带上这些"会改变检索结果"的因素：换了 embedding 模型、
# 切了集合、改了 top_k/阈值/检索模式，都应当视为不同的查询，否则会拿到不适用的旧结果。
# 检索模式尤其容易漏：同一句话在"纯向量"和"混合检索"下结果并不相同，
# 不区分就会出现"把开关关掉后，返回的还是上次混合检索的缓存"这种诡异现象。
def _cache_key(query: str, top_k: int, threshold: float) -> str:
    mode = "hybrid" if _hybrid_active() else "dense"
    if mode == "hybrid":
        mode = f"{mode}:{settings.RERANK_STRATEGY}:{settings.RRF_K}"
    raw = "|".join([
        settings.MILVUS_COLLECTION,
        settings.EMBEDDING_API_MODEL,
        str(top_k),
        str(threshold),
        mode,
        query.strip(),
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return redis_client.key("retrieve", digest)


@lru_cache(maxsize=1)
def _hybrid_active() -> bool:
    """
    本次进程是否真的走混合检索。

    两个条件同时满足才走：配置打开，且**当前集合的 schema 支持**（含 BM25 稀疏字段）。
    老集合没有稀疏字段，Milvus 又不支持给已有集合加 Function，所以只能重建；
    这里不静默降级成"混合检索假装成功"，而是明确退回纯向量并提示重建方式。
    """
    if not settings.HYBRID_ENABLED:
        return False
    if milvus_client.supports_hybrid():
        return True
    logger.warning(
        "集合缺少 BM25 稀疏字段，本次退回纯向量检索。"
        "要启用混合检索请重建集合（语料可重新入库）：uv run python -m scripts.ingest_kb --reset"
    )
    return False


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


def _search(query: str, query_vector: List[float], top_k: int) -> List[Dict]:
    """
    实际的检索逻辑（缓存之外的部分）。

    混合检索时先做**语义门槛**再融合，顺序不能反：

     1. 先用稠密向量 + 相似度阈值判断"这个问题到底在不在知识库的覆盖范围里"；
        一条都不过阈值就直接返回空，保住"知识库中暂无相关信息"这个回答；
     2. 门槛过了，再用稠密 + BM25 做混合检索并用 ranker 重排。

    为什么门槛必须由稠密分支来把：**BM25 没有绝对分数基准**，它总能返回若干
    "词面最像"的片段——哪怕问的是"今天北京天气怎么样"。如果让稀疏分支的结果
    直接进最终结果，那个"不知道就说不知道"的闸门就废了，模型会被喂进无关上下文
    然后编出答案。稠密分数有绝对含义（余弦相似度），适合当这个闸门。
    """
    if not _hybrid_active():
        return milvus_client.search(query_vector, top_k=top_k)

    # 1) 语义门槛：只要 1 条即可判断"有没有相关内容"，不用取满
    if not milvus_client.search(query_vector, top_k=1):
        logger.info("稠密分支无结果过阈值，判定为知识库未覆盖，跳过混合检索")
        return []

    # 2) 混合召回 + 重排
    return milvus_client.hybrid_search(query_vector, query, top_k=top_k)


def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """执行检索，返回命中的知识块（已排序）。优先读缓存。"""
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
    hits = _search(query, query_vector, effective_top_k)

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