"""
Milvus 向量数据库客户端
========================
Milvus 是业界主流的开源向量数据库，本模块封装：
- 集合（Collection）的创建 / 删除 / 自检
- 向量写入（入库）
- 向量检索（ANN 近似最近邻搜索）

技术要点：
- 使用 pymilvus 的轻量客户端 MilvusClient（比旧版 Collection API 更简洁、推荐）；
- 度量方式用 COSINE 余弦相似度：返回的 distance[0,2] 越小越相似；
- 检索时通过 score 阈值过滤低质量结果，防止"强行回答"。
"""
from functools import lru_cache
from typing import Dict,List,Optional

from pymilvus import MilvusClient,DataType,MilvusException

from config.settings import settings
from app.core.logger import get_logger

logger=get_logger(__name__)

@lru_cache(maxsize=1)
def get_milvus()->MilvusClient:
    """全局唯一 MilvusClient 单例（连接是懒建立 + 缓存的）。"""
    logger.info("连接 Milvus: %s", settings.MILVUS_URI)
    return MilvusClient(uri=settings.MILVUS_URI)

def collection_exists()->bool:
    return get_milvus().has_collection(settings.MILVUS_COLLECTION)

def create_collection()->None:
    """
    创建知识库向量集合。
    字段设计：
      - id          主键（自增）
      - text        原始文本（召回后直接作为 LLM 上下文）
      - title       所属文档标题
      - source      来源文件/章节（用于引用溯源）
      - chunk_index 第几个切片
      - vector      float 向量，维度 = embedding 维度
    """
    dim =settings.EMBEDDING_DIM
    client=get_milvus()
    collection=settings.MILVUS_COLLECTION
    if client.has_collection(collection):
        logger.warning(f"集合 {collection} 已存在，跳过创建")
        return

    schema=client.create_schema(auto_id=True,enable_dynamic_field=False)
    schema.add_field("id",DataType.INT64,is_primary=True,auto_id=True)
    schema.add_field("text",DataType.VARCHAR,max_length=8192)
    schema.add_field("title",DataType.VARCHAR,max_length=512)
    schema.add_field("source",DataType.VARCHAR,max_length=512)
    schema.add_field("chunk_index",DataType.INT64)
    schema.add_field("vector",DataType.FLOAT_VECTOR,dim=dim)

    index_params=client.prepare_index_params()
    index_params.add_index(
        field_name="vector",index_type="IVF_FLAT",metric_type="COSINE",params={"nlist": 1024}
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params
    )
    logger.info(f"Milvus 集合创建完成: {collection} (dim={dim})")

def drop_collection()->None:
    """删除集合（重新入库前使用，危险操作请谨慎）。"""
    if collection_exists():
        get_milvus().drop_collection(settings.MILVUS_COLLECTION)
        logger.info(f"已删除集合：{settings.MILVUS_COLLECTION}")

def insert_documents(doc:List[Dict])->int:
    """
        批量写入向量数据。
        入参 docs: [{text, title, source, chunk_index, vector}, ...]
        返回写入条数。
    """
    create_collection()
    data=[
        {
            "text":d["text"],
            "title":d["title"],
            "source":d["source"],
            "chunk_index":d["chunk_index"],
            "vector":d["vector"],
        }
        for d in doc
    ]
    res=get_milvus().insert(collection_name=settings.MILVUS_COLLECTION,data=data)
    logger.info(f"已经写入{len(data)}条向量到 {settings.MILVUS_COLLECTION}")
    # 必须 flush + load：刚写入的数据还在未封存的增量段里，Milvus 不会立刻把它算进
    # query/search 的结果。少了这一步，紧接着的 count() 会返回 0、search() 一条都搜不到，
    # 表现为"首次入库明明说成功了，知识库却像是空的"。
    flush_and_load()
    return res.get("insert_count",len(data))

def search(
        query_vector:List[float],
        top_k:int=None,
        score_threshold:float=None
)->List[Dict]:
    """
    向量检索：返回按相似度降序排列的命中文档。
    每个结果形如：{text, title, source, chunk_index, score}
    score 为余弦相似度（0~1，越大越相关）。
    """
    top_k=top_k or settings.RETRIEVE_TOP_K
    score_threshold=score_threshold if score_threshold is not None else settings.RETRIEVE_SCORE_THRESHOLD

    if not collection_exists():
        logger.warning("集合不存在，返回空结果（请先运行 scripts/ingest_kb.py 入库）")
        return []
    client=get_milvus()

    try:
        results = client.search(
            collection_name=settings.MILVUS_COLLECTION,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "title", "source", "chunk_index"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )
    except MilvusException as e:
        # 判断是不是集合未加载
        if "collection not loaded" in str(e).lower():
            logger.warning("集合未加载，执行load_collection")
            client.load_collection(settings.MILVUS_COLLECTION)
            results = client.search(
                collection_name=settings.MILVUS_COLLECTION,
                data=[query_vector],
                limit=top_k,
                output_fields=["text", "title", "source", "chunk_index"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            )
        else:
            # 其他milvus异常直接抛出
            raise e


    hits=[]
    # MilvusClient.search 返回: [[{id, distance, entity:{...}}, ...]]
    for hit in results[0]:
        # 注意：集合的 metric_type 是 COSINE，Milvus 对 COSINE 返回的 distance
        # 本身就是余弦相似度（越大越相关），**不是**距离，所以这里直接取用。
        # 曾经写成 score=1.0-float(hit["distance"])——那会把相关性整个反过来：
        # 最相关的切片被阈值滤掉，留下最不相关的，且分数含义也不再是"相似度"。
        score=float(hit["distance"])

        if score < score_threshold:
            continue
        entity=hit["entity"]
        hits.append(
            {
                "text": entity["text"],
                "title": entity.get("title", ""),
                "source": entity.get("source", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "score": round(score, 4),
            }
        )

    return hits

def count()->int:
    """
    返回集合中的向量总数。
    用 query + count(*) 聚合（实时准确）；get_collection_stats 的行数统计
    存在最终一致性延迟，不适合刚写入后立即判断。
    """
    if not collection_exists():
        return 0
    res=get_milvus().query(
        collection_name=settings.MILVUS_COLLECTION,
        filter="id >= 0",
        output_fields=["count(*)"]
    )
    return int(res[0]["count(*)"]) if res else 0

def flush_and_load():
    if not collection_exists():
        return 0
    client=get_milvus()
    client.flush(collection_name=settings.MILVUS_COLLECTION)
    client.load_collection(collection_name=settings.MILVUS_COLLECTION)
    return "success"
