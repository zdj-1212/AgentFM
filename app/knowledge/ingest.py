"""
知识库入库流水线
=================
把 app/knowledge/corpus/ 下的企业文档，经过「读取 -> 切分 -> 向量化 -> 写入 Milvus」
变成可供 RAG 检索的向量库。

用法（在项目根目录执行）：
    uv run python -m scripts.ingest_kb            # 增量入库
    uv run python -m scripts.ingest_kb --reset    # 清空后重新全量入库
"""
import argparse
import re
from pathlib import Path

from tenacity import retry

from app.core.embedding import get_embedder
from app.core.logger import get_logger
from app.database import milvus_client
from app.knowledge.chunker import build_chunk_meta,chunk_text

logger=get_logger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

SUPPORTED_EXTS ={".md",".txt"}

def _parse_title(filename:str)->str:
    """从文件名提取文档标题，如 '01_售后政策.md' -> '售后政策'。"""
    stem=Path(filename).stem
    title =re.sub(r"^\d+[_-]\s*", "", stem)
    return title

def load_corpus()->list[dict]:
    """读取语料目录下所有文档并切分，返回待入库的文档块列表。"""
    all_docs = []
    for file in sorted(CORPUS_DIR.glob("*")):
        if file.suffix.lower() not in SUPPORTED_EXTS:
            continue
        text= file.read_text(encoding="utf-8")
        title=_parse_title(file.name)
        chunks=chunk_text(text)

        for idx,chunk in enumerate(chunks):
            all_docs.append(build_chunk_meta(source=file.name,title=title,chunk_index=idx,text=chunk))

        logger.info("文档 %s：共 %s 字，切分为 %s 块", file.name, len(text), len(chunks))
    return all_docs

def ingest(reset: bool =False)->int:
    """执行入库。reset=True 时先清空集合再全量写入。"""
    if reset:
        logger.warning("--reset 模式：清空 Milvus 集合后重新入库")
        milvus_client.drop_collection()

    # 1) 读取并切分语料
    docs=load_corpus()
    if not docs:
        logger.warning("语料目录为空，未生成任何文档块")
        return 0
    # 2) 批量向量化
    embedder=get_embedder()
    texts=[d["text"] for d in docs]
    logger.info("开始向量化 %s 个文档块...", len(texts))
    vectors = embedder.embed_documents(texts)
    # 3) 向量与元数据合并后写入 Milvus
    for doc, vec in zip(docs,vectors):
        doc["vector"]=vec
    inserted = milvus_client.insert_documents(docs)

    logger.info(
        "入库完成：新增 %s 条，集合当前总量 %s 条",
        inserted,
        milvus_client.count(),
    )

    # 知识库变了，缓存里基于旧内容的检索结果必须作废，否则会继续拿旧切片回答
    from app.knowledge.retriever import invalidate_cache

    invalidate_cache()
    return inserted


