"""
检索器（Retriever）
====================
把"向量化 + Milvus 检索 + 上下文拼装"封装为统一的 RAG 检索入口。

对外提供两个能力：
- retrieve(query)：纯检索，返回命中的知识块列表（带分数与来源）；
- build_context(query)：检索并拼装成可直接喂给 LLM 的上下文文本（含引用编号）。
"""

from typing import Dict,List

from app.core.embedding import get_embedder
from app.core.logger import get_logger
from app.database import milvus_client

logger=get_logger(__name__)

def retrieve(query: str, top_k: int = None) -> List[Dict]:
    """执行向量检索，返回命中的知识块（按相似度降序）。"""
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)
    hits = milvus_client.search(query_vector, top_k=top_k)
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